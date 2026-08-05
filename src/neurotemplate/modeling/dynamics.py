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
    data[:, i-past:i+1, :] flattened.

  - The data loader provides 'yn' initialized to the first in_steps
    of ground truth (see data_utils.get_sw_dict). The System then
    appends model predictions to 'yn' via cat() at each rollout step.

  - 'Y' (ground truth) is only used by the loss for comparison and
    by the System to determine rollout length (nstep_key='Y').

cfg.in_steps is the model's context length, and every family reads it the
same way: as a flattened input window of in_steps past values per input key,
gathered by the Node's input_map. A stateful block (GRU) additionally keeps a
hidden state so its memory can run past in_steps during a rollout, but it
still takes the same window — it just skips the steps already folded into that
state (see GRUBlock). The only thing statefulness costs the caller is a reset
at each sequence boundary, which make_reset_init() wires into the System.

Because cfg.in_steps is also the sliding-window stride
(data_utils.get_sw_dict), keeping it meaningful for every family is what
makes two models trained at the same in_steps see the same training set.
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
    GRU-based dynamics block. Reads the same flattened in_steps history window
    every other block reads, but folds each timestep into its hidden state
    exactly once:

      - Cold (no hidden state): consume the whole window, so the state
        summarises the full context before the first prediction.
      - Warm: the window's earlier steps are already in the hidden state, so
        consume only its newest step.

    The invariant is that the hidden state summarises every timestep this block
    has already consumed, and the window supplies whatever it has not. That is
    what lets memory run past in_steps during a rollout while a single cold
    call is still correct on its own — instantiate this block anywhere, hand it
    one window, and you get a properly warmed prediction.

    Call reset() at a sequence boundary (a new batch); the next call re-warms
    itself from its own window.

    Handles two input shapes:
      - [batch, in_steps*features]  — normal training/inference path
      - [in_steps*features]         — under vmap (jacrev); always cold, no
                                      state stored
    """

    def __init__(self, ny: int, nu: int, nd: int, in_steps: int, outsize: int,
                 hidden_size: int, num_layers: int, bias: bool = True):
        super().__init__()
        self.widths = (ny, nu, nd)
        self.in_steps = in_steps
        self.in_features = in_steps * (ny + nu + nd)
        self.out_features = outsize
        self.gru = nn.GRU(ny + nu + nd, hidden_size, num_layers=num_layers,
                          batch_first=True, bias=bias)
        self.output = nn.Linear(hidden_size, outsize, bias=bias)
        self._h: Optional[torch.Tensor] = None

    def reset(self) -> None:
        """Clear hidden state. Call before each new sequence."""
        self._h = None

    def reg_error(self) -> torch.Tensor:
        weights = [p.pow(2).mean() for p in self.gru.parameters() if p.dim() > 1]
        return torch.mean(torch.stack(weights))

    def _to_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """
        [..., in_steps*(ny+nu+nd)] -> [..., in_steps, ny+nu+nd].

        SSMGrad hands us cat([y_window, u_window, d_window]), and each window is
        time-major within itself, so the flat vector groups by key rather than
        by timestep. A GRU has to read it per timestep, so regroup it.
        """
        lead = x.shape[:-1]
        windows = x.split([self.in_steps * w for w in self.widths], dim=-1)
        return torch.cat(
            [w.reshape(*lead, self.in_steps, width)
             for w, width in zip(windows, self.widths)],
            dim=-1,
        )

    def _manual_gru(self, seq: torch.Tensor) -> torch.Tensor:
        """
        Replay a whole window from a zero hidden state using vmap-safe
        primitives (matmul/sigmoid/tanh). nn.GRU's fused CuDNN kernel cannot be
        batched by vmap, so jacrev under vmap needs this path.

        Variable names follow the PyTorch nn.GRU equation:
            r = sigmoid(W_ir x + b_ir + W_hr h + b_hr)   # reset gate
            z = sigmoid(W_iz x + b_iz + W_hz h + b_hz)   # update gate
            n = tanh   (W_in x + b_in + r*(W_hn h + b_hn))  # candidate
            h = (1 - z) * n + z * h_prev
        See https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html

        Args:
            seq: [in_steps, ny+nu+nd] (1D input, under vmap)
        Returns:
            [outsize] output
        """
        # One hidden state per layer, all starting at zero — the sampled
        # gradient penalty measures d(prediction)/d(window) from cold, the
        # same object it measures for a stateless block.
        hidden = [torch.zeros(self.gru.hidden_size, device=seq.device)
                  for _ in range(self.gru.num_layers)]

        for step in range(self.in_steps):
            x = seq[step]
            for layer in range(self.gru.num_layers):
                w_ih = getattr(self.gru, f'weight_ih_l{layer}')  # [3*H, input]
                w_hh = getattr(self.gru, f'weight_hh_l{layer}')  # [3*H, H]
                b_ih = getattr(self.gru, f'bias_ih_l{layer}')     # [3*H]
                b_hh = getattr(self.gru, f'bias_hh_l{layer}')     # [3*H]

                gi = x @ w_ih.T + b_ih             # [3*H]
                gh = hidden[layer] @ w_hh.T + b_hh  # [3*H]

                i_r, i_z, i_n = gi.chunk(3)
                h_r, h_z, h_n = gh.chunk(3)

                r = torch.sigmoid(i_r + h_r)
                z = torch.sigmoid(i_z + h_z)
                n = torch.tanh(i_n + r * h_n)
                hidden[layer] = (1 - z) * n + z * hidden[layer]
                x = hidden[layer]  # input to next layer

        return self.output(hidden[-1])  # last layer, last step

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq = self._to_sequence(x)

        if x.ndim == 1:
            # Under vmap: manual GRU math (fused nn.GRU kernel isn't vmap-safe)
            return self._manual_gru(seq)

        # Cold: fold the whole window in. Warm: every step but the newest is
        # already in the hidden state.
        steps = seq if self._h is None else seq[:, -1:, :]
        out, self._h = self.gru(steps, self._h)   # [batch, len(steps), hidden]
        return self.output(out[:, -1, :])          # [batch, outsize]


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
        in_steps, ny = self.in_steps, self.ny
        return y[:, (in_steps - 1) * ny : in_steps * ny]

    def forward(self, y, u, d):
        """
        Forward pass: predict next state from current inputs.

        Args:
            y: [batch, in_steps*ny]
            u: [batch, in_steps*nu]
            d: [batch, in_steps*nd]

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
def get_gru(cfg: ModelingConfig, nm_mapping: dict[str, list[str]]) -> SSMGrad:
    ny = len(nm_mapping['Y'])
    nu = len(nm_mapping['U'])
    nd = len(nm_mapping['D'])

    fy = GRUBlock(
        ny, nu, nd, cfg.in_steps, outsize=ny,
        hidden_size=(2**cfg.log2_hidden_size), num_layers=cfg.num_layers,
    )
    # Same in_steps window as every other family. The GRU is additionally
    # stateful, so its memory outlives the window during a rollout — but the
    # window is what warms it, so no caller-side priming is needed.
    return SSMGrad(
        fy, None, None, ny, nu, nd,
        cfg.in_steps, cfg.residual, separable=False,
    )


class _SetpointPassthrough(nn.Module):
    """Dummy dynamics: next zone temp = current cooling setpoint.

    Given the inseparable [y | u | d] row, returns the U (setpoint) block as
    the next state. `bias` is an ordinary output offset kept at zero — it only
    exists so the sys-id optimizer has a trainable parameter to hold; mint the
    dummy checkpoint with lr=0 and it stays zero, so the prediction is exactly
    the setpoint.
    """

    def __init__(self, ny: int, nu: int):
        super().__init__()
        self.ny = ny
        self.nu = nu
        self.bias = nn.Parameter(torch.zeros(ny))

    def forward(self, cat_yud: torch.Tensor) -> torch.Tensor:
        setpoint = cat_yud[:, self.ny:self.ny + self.nu]
        return setpoint + self.bias


@beartype
def get_dummy(cfg: ModelingConfig, nm_mapping: dict[str, list[str]]) -> SSMGrad:
    """Content-free surrogate that predicts zone temp = cooling setpoint.

    Used to test whether the DPC policy can be trained against a trivially
    invertible model (no real sys-id needed). Requires ny == nu (each zone
    temp is driven by its own setpoint) and must run in physical units
    (norm_type=None) so setpoint and zone temp share units.
    """
    ny = len(nm_mapping['Y'])
    nu = len(nm_mapping['U'])
    nd = len(nm_mapping['D'])
    if ny != nu:
        raise ValueError(
            f"dummy model needs one setpoint per zone temp: ny ({ny}) must "
            f"equal nu ({nu}). Y={nm_mapping['Y']}, U={nm_mapping['U']}."
        )
    # in_steps=1: a passthrough only needs the current setpoint (no windowing).
    return SSMGrad(
        _SetpointPassthrough(ny, nu), None, None, ny, nu, nd,
        in_steps=1, residual=False, separable=False,
    )


model_catalog = {
    "mlp": get_mlp,
    "nssm": get_nssm,
    "lssm": get_lssm,
    "gru": get_gru,
    "dummy": get_dummy,
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
      - factory: model family name (mlp/nssm/lssm/gru)
      - separable: whether the SSM is separable (fy + fu + fd)
      - residual: whether y_{k+1} adds y_k as a residual base
      - in_steps: context length (cfg.in_steps) — see below
      - ny / nu / nd: shape parameters
      - param_count: total trainable parameters
      - blocks: per-block descriptors {fy, fu, fd}

    Used by reportcard.render_reportcard.
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
        # Context length. Matches cfg.in_steps for every real family; only the
        # dummy passthrough narrows it to 1, since it reads just the setpoint.
        "in_steps": int(raw.in_steps),
        "ny": int(raw.ny),
        "nu": int(raw.nu),
        "nd": int(raw.nd),
        "param_count": _count_params(raw),
        "blocks": blocks_info,
    }


# ── System construction ──────────────────────────────────────────

def make_reset_init(model: SSMGrad):
    """
    Build the SystemPreview init_func for a stateful model, or None.

    A hidden state summarises every step the block has already consumed, so it
    must be cleared at a sequence boundary — otherwise state, and the autograd
    graph hanging off it, leaks from one batch into the next. This is the whole
    of what statefulness asks of the caller: the block re-warms itself from its
    own input window on the first call of the rollout, so nothing here needs to
    know the rollout's indices or key names.

    Gradients still flow through that warmup: it is an ordinary forward pass, so
    the hidden state at the first scored step is a function of the block's
    weights, which is exactly what we want the loss to shape.

    Returns None for stateless blocks, leaving SystemPreview's identity init.

    Note: unrelated to ModelingConfig.warmup, which counts early-stopping epochs.
    """
    # A reset() on the block is what makes it stateful — GRUBlock.reset is the
    # single source of truth, so a new stateful block gets this for free.
    reset = getattr(model.fy, "reset", None)
    if reset is None:
        return None

    def reset_init(data):
        reset()
        return data

    return reset_init


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
    step i, the Node receives a window of model.in_steps past values for each
    input, flattened to [batch, model.in_steps*features].
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
                f"model.ny={model.ny}; factories (currently) assume width-1 heads."
            )
        model.head_stack = head_stack

    # Temporal windowing: the gathered window must be exactly as wide as the
    # block's declared input width, and model.in_steps is that width. Only the
    # passthrough (dummy) block declares 1, so it gets no window.
    if model.in_steps > 1:
        input_map = {
            key: {
                "past": model.in_steps - 1,
                "future": 0,
                "pad_mode": "nearest",
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

    return SystemPreview(
        [node],
        name=cfg.model.upper() + " System",
        nstep_key="Y",               # rollout length from ground truth
        start_iter=cfg.in_steps - 1,  # skip initial history steps
        # Stateful blocks only: clear the hidden state so it cannot leak
        # between batches. None for stateless blocks.
        init_func=make_reset_init(model),
    )
