"""
Dynamics models wrapped for Neuromancer.

SSMGrad is a pure predictor: (y, u, d) -> y_next. It does NOT compute
Jacobians in its forward pass — gradient sign constraints are handled
externally by SampledGradientPenalty (see gradient_sampling.py).

get_dynamics() wraps SSMGrad in a Neuromancer Node + SystemPreview:

  - The Node uses input_keys=['yn', 'U', 'D'] and output_keys=['yn'].
    Because 'yn' appears in BOTH input and output, the System creates
    an autoregressive feedback loop: at each timestep, the model reads
    its own previous prediction and produces the next one.

  - SystemPreview with start_iter=in_steps-1 skips the first timesteps
    (the model needs in_steps of history). The input_map on the Node
    handles temporal windowing: at each step i, the model receives
    data[:, i-past:i+1, :] flattened (or as a 3D window for TCN).

  - The data loader provides 'yn' initialized to the first in_steps
    of ground truth (see data_utils.get_sw_dict). The System then
    appends model predictions to 'yn' via cat() at each rollout step.

  - 'Y' (ground truth) is only used by the loss for comparison and
    by the System to determine rollout length (nstep_key='Y').
"""

import torch
import torch.nn as nn
from typing import Callable, Optional
from neurotemplate.modeling.config import ModelingConfig

from neurotemplate._typing import beartype

from neuromancer.system import Node, SystemPreview
from neuromancer.modules import blocks

from neurotemplate.modeling.heads import HeadStack


class GRUBlock(nn.Module):
    """
    GRU-based dynamics block. Stateful: hidden state persists across rollout
    steps within a single sequence, then resets between sequences via reset().

    Handles two input shapes:
      - [batch, features]  — normal training/inference path
      - [features]         — under vmap (jacrev); starts from zeros, no state stored
    """

    def __init__(self, insize: int, outsize: int, hidden_size: int, num_layers: int, bias: bool = True):
        super().__init__()
        self.in_features = insize
        self.out_features = outsize
        self.gru = nn.GRU(insize, hidden_size, num_layers=num_layers, batch_first=True, bias=bias)
        self.output = nn.Linear(hidden_size, outsize, bias=bias)
        self._h: Optional[torch.Tensor] = None

    def reset(self) -> None:
        """Clear hidden state. Call before each new sequence."""
        self._h = None

    def reg_error(self) -> torch.Tensor:
        weights = [p.pow(2).mean() for p in self.gru.parameters() if p.dim() > 1]
        return torch.mean(torch.stack(weights))

    def _manual_gru_step(self, x: torch.Tensor) -> torch.Tensor:
        """
        Manual single-step GRU forward unrolled into vmap-safe primitives
        (matmul/sigmoid/tanh). nn.GRU's fused CuDNN kernel cannot be
        batched by vmap, so jacrev under vmap needs this path.

        Variable names follow the PyTorch nn.GRU equation:
            r = sigmoid(W_ir x + b_ir + W_hr h + b_hr)   # reset gate
            z = sigmoid(W_iz x + b_iz + W_hz h + b_hz)   # update gate
            n = tanh   (W_in x + b_in + r*(W_hn h + b_hn))  # candidate
            h = (1 - z) * n + z * h_prev
        See https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html

        Args:
            x: [features] (1D, under vmap)
        Returns:
            [outsize] output
        """
        h = torch.zeros(self.gru.hidden_size, device=x.device)

        for layer in range(self.gru.num_layers):
            w_ih = getattr(self.gru, f'weight_ih_l{layer}')  # [3*H, input]
            w_hh = getattr(self.gru, f'weight_hh_l{layer}')  # [3*H, H]
            b_ih = getattr(self.gru, f'bias_ih_l{layer}')     # [3*H]
            b_hh = getattr(self.gru, f'bias_hh_l{layer}')     # [3*H]

            gi = x @ w_ih.T + b_ih    # [3*H]
            gh = h @ w_hh.T + b_hh    # [3*H]

            i_r, i_z, i_n = gi.chunk(3)
            h_r, h_z, h_n = gh.chunk(3)

            r = torch.sigmoid(i_r + h_r)
            z = torch.sigmoid(i_z + h_z)
            n = torch.tanh(i_n + r * h_n)
            h = (1 - z) * n + z * h
            x = h  # input to next layer

        return self.output(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            # Under vmap: manual GRU math (fused nn.GRU kernel isn't vmap-safe)
            return self._manual_gru_step(x)

        # Batched: [batch, features] → treat as single timestep
        out, self._h = self.gru(x.unsqueeze(1), self._h)    # [batch, 1, hidden]
        return self.output(out[:, 0, :])                     # [batch, outsize]


class SSMGrad(nn.Module):
    """
    State space model for discrete-time dynamics:
        y_{k+1} = f(y_k, u_k, d_k)  [+ y_k if residual]

    Two modes:
        Separable:   y_{k+1} = fy(y_k) + fu(u_k) + fd(d_k)
        Inseparable: y_{k+1} = fy(cat(y_k, u_k, d_k))

    This is a pure predictor — no Jacobians computed here.
    Gradient penalties use model.compute_output externally via jacrev.
    """

    def __init__(
        self,
        fy: Callable,
        fu: Optional[Callable],
        fd: Optional[Callable],
        ny: int,
        nu: int,
        nd: int,
        in_steps: int,
        residual: bool,
        separable: bool,
        head_stack: Optional[HeadStack] = None,
    ):
        super().__init__()
        self.fy, self.fu, self.fd = fy, fu, fd
        self.ny, self.nu, self.nd = ny, nu, nd
        self.in_steps = in_steps
        self.residual = residual
        self.head_stack = head_stack

        self.is_tcn = not separable and hasattr(fy, 'tcn')

        if separable:
            self.compute_output = self.compute_separable
        else:
            self.compute_output = self.compute_inseparable

    def compute_separable(self, y, u, d):
        return self.fy(y) + self.fu(u) + self.fd(d)

    def compute_inseparable(self, y, u, d):
        return self.fy(torch.cat([y, u, d], dim=-1))

    def _last_y(self, y: torch.Tensor) -> torch.Tensor:
        """Extract the most recent in_steps slice as [batch, ny]."""
        if y.ndim == 2:
            in_steps, ny = self.in_steps, self.ny
            return y[:, (in_steps - 1) * ny : in_steps * ny]
        return y[:, -1]

    def forward(self, y, u, d):
        """
        Forward pass: predict next state from current inputs.

        Args:
            y: [batch, in_steps*ny] or [batch, in_steps, ny] (TCN)
            u: [batch, in_steps*nu] or [batch, in_steps, nu]
            d: [batch, in_steps*nd] or [batch, in_steps, nd]

        Returns:
            [batch, ny] next state — activated (post-head) if a head_stack
            is attached, otherwise raw model output.
        """
        out = self.compute_output(y, u, d)

        if self.head_stack is None:
            # Pre-heads path: residual + identity, unchanged behaviour.
            if self.residual:
                out = out + self._last_y(y)
            return out

        # Heads path: residual is applied per-channel (only to channels
        # whose head opts in) BEFORE activation, then the stack activates.
        if self.residual:
            out = self.head_stack.apply_residual(out, self._last_y(y))
        return self.head_stack.activate(out)


# ── Model factory functions ──────────────────────────────────────

@beartype
def get_mlp(cfg: ModelingConfig, nm_mapping: dict[str, list[str]]) -> SSMGrad:
    ny = len(nm_mapping['Y'])
    nu = len(nm_mapping['U'])
    nd = len(nm_mapping['D'])
    insize = (ny + nu + nd) * cfg.in_steps

    fy = blocks.MLP(
        insize=insize, outsize=ny, bias=True,
        linear_map=nn.Linear, nonlin=nn.ReLU,
        hsizes=cfg.num_layers * [(2**cfg.log2_hidden_size)],
    )
    return SSMGrad(
        fy, None, None, ny, nu, nd,
        cfg.in_steps, cfg.residual, separable=False,
    )


@beartype
def get_nssm(cfg: ModelingConfig, nm_mapping: dict[str, list[str]]) -> SSMGrad:
    ny = len(nm_mapping['Y'])
    nu = len(nm_mapping['U'])
    nd = len(nm_mapping['D'])
    s = cfg.in_steps

    fy = blocks.MLP(s*ny, ny, bias=True, linear_map=nn.Linear,
                    nonlin=nn.ReLU, hsizes=cfg.num_layers * [(2**cfg.log2_hidden_size)])
    fu = blocks.MLP(s*nu, ny, bias=True, linear_map=nn.Linear,
                    nonlin=nn.ReLU, hsizes=cfg.num_layers * [(2**cfg.log2_hidden_size)])
    fd = blocks.MLP(s*nd, ny, bias=True, linear_map=nn.Linear,
                    nonlin=nn.ReLU, hsizes=cfg.num_layers * [(2**cfg.log2_hidden_size)])

    return SSMGrad(
        fy, fu, fd, ny, nu, nd,
        s, cfg.residual, separable=True,
    )


@beartype
def get_lssm(cfg: ModelingConfig, nm_mapping: dict[str, list[str]]) -> SSMGrad:
    ny = len(nm_mapping['Y'])
    nu = len(nm_mapping['U'])
    nd = len(nm_mapping['D'])
    s = cfg.in_steps

    fy = blocks.Linear(s*ny, ny, bias=True, linear_map=nn.Linear)
    fu = blocks.Linear(s*nu, ny, bias=True, linear_map=nn.Linear)
    fd = blocks.Linear(s*nd, ny, bias=True, linear_map=nn.Linear)

    return SSMGrad(
        fy, fu, fd, ny, nu, nd,
        s, cfg.residual, separable=True,
    )


@beartype
def get_tcn(cfg: ModelingConfig, nm_mapping: dict[str, list[str]]) -> SSMGrad:
    ny = len(nm_mapping['Y'])
    nu = len(nm_mapping['U'])
    nd = len(nm_mapping['D'])

    fy = blocks.CausalTCN(
        insize=ny + nu + nd, outsize=ny, bias=True,
        linear_map=nn.Linear, nonlin=nn.ReLU,
        hsizes=cfg.num_layers * [(2**cfg.log2_hidden_size)],
    )
    return SSMGrad(
        fy, None, None, ny, nu, nd,
        cfg.in_steps, cfg.residual, separable=False,
    )


@beartype
def get_gru(cfg: ModelingConfig, nm_mapping: dict[str, list[str]]) -> SSMGrad:
    ny = len(nm_mapping['Y'])
    nu = len(nm_mapping['U'])
    nd = len(nm_mapping['D'])

    fy = GRUBlock(
        insize=ny + nu + nd, outsize=ny,
        hidden_size=(2**cfg.log2_hidden_size), num_layers=cfg.num_layers,
    )
    # GRU is stateful — in_steps=1, no windowing. Temporal memory lives in the hidden state.
    return SSMGrad(fy, None, None, ny, nu, nd, 1, cfg.residual, separable=False)


model_catalog = {
    "mlp": get_mlp,
    "nssm": get_nssm,
    "lssm": get_lssm,
    "tcn": get_tcn,
    "gru": get_gru,
}


# ── Model description (for reportcard) ───────────────────────────

def _count_params(module: Optional[nn.Module]) -> int:
    if module is None:
        return 0
    return sum(p.numel() for p in module.parameters())


def _describe_block(block: Optional[nn.Module]) -> Optional[dict]:
    """Return a small dict describing an SSMGrad sub-block (fy/fu/fd)."""
    if block is None:
        return None
    info = {
        "class": type(block).__name__,
        "param_count": _count_params(block),
    }
    # MLP / Linear blocks expose hsizes via their nhidden attribute or args
    if hasattr(block, "hsizes"):
        info["hsizes"] = list(block.hsizes)
    if hasattr(block, "in_features"):
        info["in_features"] = block.in_features
    if hasattr(block, "out_features"):
        info["out_features"] = block.out_features
    return info


@beartype
def describe_dynamics(dyn_sys: SystemPreview, cfg: ModelingConfig) -> dict:
    """
    Describe the trained dynamics system for the reportcard.

    Returns a dict with:
      - factory: model family name (mlp/nssm/lssm/tcn/gru)
      - separable: whether the SSM is separable (fy + fu + fd)
      - residual: whether y_{k+1} adds y_k as a residual base
      - in_steps / ny / nu / nd: shape parameters
      - param_count: total trainable parameters
      - blocks: per-block descriptors {fy, fu, fd}

    Used by reportcard.render_reportcard. Reads only what's available on
    the constructed module — does not re-introspect cfg.
    """
    # Unwrap: SystemPreview -> Node -> SSMGrad
    raw = dyn_sys.nodes[0].callable

    blocks_info: dict[str, Optional[dict]] = {
        "fy": _describe_block(getattr(raw, "fy", None)),
        "fu": _describe_block(getattr(raw, "fu", None)),
        "fd": _describe_block(getattr(raw, "fd", None)),
    }

    return {
        "factory": cfg.model,
        "separable": raw.fu is not None and raw.fd is not None,
        "residual": bool(raw.residual),
        "in_steps": int(raw.in_steps),
        "ny": int(raw.ny),
        "nu": int(raw.nu),
        "nd": int(raw.nd),
        "param_count": _count_params(raw),
        "blocks": blocks_info,
    }


# ── System construction ──────────────────────────────────────────

@beartype
def get_dynamics(
    cfg: ModelingConfig,
    nm_mapping: dict[str, list[str]],
    head_stack: Optional[HeadStack] = None,
) -> SystemPreview:
    """
    Wrap a dynamics model in a Neuromancer SystemPreview for autoregressive rollout.

    The Node reads ['yn', 'U', 'D'] and writes ['yn']. Because 'yn' is both
    input and output, the System feeds predictions back as input at the next
    timestep — this is the autoregressive loop.

    SystemPreview handles temporal windowing via input_map: at each rollout
    step i, the Node receives a window of in_steps past values for each input,
    flattened to [batch, in_steps*features] (or kept 3D for TCN).
    """
    if cfg.model not in model_catalog:
        raise ValueError(
            f"Unknown model '{cfg.model}'. "
            f"Available: {list(model_catalog.keys())}"
        )

    model = model_catalog[cfg.model](cfg, nm_mapping)

    # Heads are attached post-construction so each factory stays focused
    # on its own block topology. The first-pass model output width is
    # still ny (every head has width 1) — once OrdinalHead lands we will
    # need to thread raw_width back into the factories.
    if head_stack is not None:
        if head_stack.raw_width != model.ny:
            raise ValueError(
                f"head_stack.raw_width={head_stack.raw_width} does not match "
                f"model.ny={model.ny}; first-pass factories assume width-1 heads."
            )
        model.head_stack = head_stack

    # Temporal windowing: each input gets a window of in_steps past values.
    # GRU is stateful — hidden state carries across steps, so no windowing ever.
    # For in_steps=1, no windowing needed (current timestep only).
    if cfg.in_steps > 1 and cfg.model != "gru":
        input_map = {
            key: {
                "past": cfg.in_steps - 1,
                "future": 0,
                "pad_mode": "nearest",
                # MLP/NSSM/LSSM want flattened [batch, steps*features]
                # TCN wants 3D [batch, steps, features]
                "flatten": cfg.model != "tcn",
            }
            for key in ["yn", "U", "D"]
        }
    else:
        input_map = {}

    node = Node(
        model,
        ['yn', 'U', 'D'],   # input: reads predicted state + exogenous
        ['yn'],              # output: next predicted state (feedback loop)
        name=cfg.model.upper() + " Node",
        input_map=input_map,
    )

    system = SystemPreview(
        [node],
        name=cfg.model.upper() + " System",
        nstep_key="Y",               # rollout length from ground truth
        start_iter=cfg.in_steps - 1,  # skip initial history steps
    )

    # GRU: reset hidden state before each system forward pass (i.e. each batch).
    # This gives a clean computational graph per sequence while carrying state
    # across rollout steps within a sequence.
    if cfg.model == "gru":
        gru_block = model.fy
        system.register_forward_pre_hook(lambda m, inp: gru_block.reset())

    return system
