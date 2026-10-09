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

import copy
import warnings

import torch
import torch.nn as nn
from typing import Callable, Optional
from neurotemplate.modeling.config import ModelingConfig

from neurotemplate._typing import beartype

from neuromancer.system import Node, SystemPreview
# Aliased: this module's own per-block Block type would otherwise collide
# with neuromancer's `blocks` module name (MLP/Linear builders).
from neuromancer.modules import blocks as nm_blocks

from neurotemplate.modeling import windows
from neurotemplate.modeling.heads import HeadStack, PreStack
from neurotemplate.modeling.bounded import bounded_parameters
from neurotemplate.modeling.rc import TAU_KEYS, RCBlock, time_constant_boxes
from neurotemplate.modeling.blocks import Block, build_blocks


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

        # SSMGrad hands us a key-major window (see `windows`). A GRU reads
        # time-major, and the reorder is a fixed permutation of positions —
        # it depends only on (widths, in_steps), never on the data — so
        # compute it once here instead of on every call.
        self.register_buffer(
            "_seq_perm",
            torch.tensor(windows.time_major(self.widths, in_steps), dtype=torch.long),
            persistent=False,
        )
        self._last_step_slices = windows.last_step(self.widths, in_steps)

    # The stateful-block protocol (see `neurotemplate.export.state`). nn.GRU
    # keeps its state layer-first, [num_layers, batch, hidden]; the protocol
    # is batch-first and flat, so the two methods own that reshape.

    def rollout_state(self) -> Optional[torch.Tensor]:
        """The hidden state, [batch, num_layers*hidden], or None if cold."""
        if self._h is None:
            return None
        return self._h.transpose(0, 1).reshape(self._h.shape[1], -1)

    def set_rollout_state(self, state: Optional[torch.Tensor]) -> None:
        """Restore a hidden state, or None to force a cold start."""
        if state is None:
            self._h = None
            return
        layers, hidden = self.gru.num_layers, self.gru.hidden_size
        self._h = state.reshape(-1, layers, hidden).transpose(0, 1).contiguous()

    def _to_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """[..., in_steps*(ny+nu+nd)] -> [..., in_steps, ny+nu+nd], key-major
        reordered to time-major via the precomputed permutation."""
        lead = x.shape[:-1]
        seq = x.index_select(x.ndim - 1, self._seq_perm)
        return seq.reshape(*lead, self.in_steps, sum(self.widths))

    def _last_step(self, x: torch.Tensor) -> torch.Tensor:
        """[..., in_steps*(ny+nu+nd)] -> [..., ny+nu+nd]: just the newest
        timestep, without touching the rest of the window."""
        return torch.cat([x[..., s] for s in self._last_step_slices], dim=-1)

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
        if x.ndim == 1:
            # Under vmap: manual GRU math (fused nn.GRU kernel isn't vmap-safe).
            # Always cold, so it needs the reordered window regardless.
            return self._manual_gru(self._to_sequence(x))

        if self._h is None:
            # Cold: fold the whole window in.
            steps = self._to_sequence(x)
        else:
            # Warm: every step but the newest is already in the hidden state —
            # no need to touch the rest of the window.
            steps = self._last_step(x).unsqueeze(1)  # [batch, 1, ny+nu+nd]

        out, self._h = self.gru(steps, self._h)   # [batch, len(steps), hidden]
        return self.output(out[:, -1, :])          # [batch, outsize]


class BlockStack(nn.Module):
    """
    One independent sub-block per block, over the one window they all share.

    Each sub-block sees its own Y and U columns, its own D subset (all of D
    by default), and anything it `reads` from another block (see `blocks`
    for what a block owns), predicts only its own Y channels, and never
    reads another block's state on the same step. The wrapper does two
    things and nothing else: gather each block's slice out of the shared
    flattened window, and put the per-block outputs back in Y order.

    Both are fixed permutations of positions, so both are `index_select`
    against a buffer computed at construction — which keeps the whole thing
    vmap-safe, and so keeps the sampled gradient penalty working. This holds
    even with `reads`: every input, including a read one, comes from the
    shared history window rather than another block's same-step output, so
    there is no evaluation-order dependency between blocks to break it.

    Args:
        blocks: the partition, from `blocks.build_blocks`.
        widths: `(ny, nu, nd)` of the full mapping.
        in_steps: history window length.
        make_block: `(block) -> nn.Module` for one block's sub-block. The
            block carries its own shape (`ny`, `nu`, `nd`, `ny_in`) and
            channel names. Called once per block, so every block gets its
            own parameters.
    """

    def __init__(self, blocks: list[Block], widths: tuple[int, int, int],
                 in_steps: int, make_block: Callable[[Block], nn.Module]):
        super().__init__()
        ny = widths[0]
        self.block_specs = list(blocks)
        self.in_features = in_steps * sum(widths)
        self.out_features = ny

        self.blocks = nn.ModuleList([make_block(b) for b in blocks])

        # Every block's gather indices live in one buffer, so they follow the
        # module onto its device together; `_spans` says which stretch of it
        # belongs to which block.
        gathers, self._spans, cursor = [], [], 0
        for block in blocks:
            indices = windows.gather_columns(
                widths, in_steps, (block.y_in_idx, block.u_idx, block.d_idx))
            gathers += indices
            self._spans.append(slice(cursor, cursor + len(indices)))
            cursor += len(indices)
        self.register_buffer(
            "_gather", torch.tensor(gathers, dtype=torch.long), persistent=False)

        # Sub-block outputs concatenate in block order; this puts them back
        # in the order nm_mapping["Y"] lists, which is the order the loss and
        # the autoregressive feedback both assume.
        block_order = [i for block in blocks for i in block.y_idx]
        self.register_buffer(
            "_to_y_order",
            torch.tensor(
                [block_order.index(i) for i in range(ny)], dtype=torch.long
            ),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dim = x.ndim - 1                    # the feature axis, batched or not
        out = torch.cat(
            [block(x.index_select(dim, self._gather[span]))
             for block, span in zip(self.blocks, self._spans)],
            dim=-1,
        )
        return out.index_select(dim, self._to_y_order)



class SSMGrad(nn.Module):
    """
    State space model for discrete-time dynamics:
        y_{k+1} = f(y_k, u_k, d_k)  [+ y_k if residual]

    Two modes:
        Separable:   y_{k+1} = fy(y_k) + fu(u_k) + fd(d_k)
        Inseparable: y_{k+1} = fy(cat(y_k, u_k, d_k))

    This is a pure predictor — no Jacobians computed here. The sampled
    gradient penalty differentiates `pre_activation` externally via jacrev.

    An optional `pre_stack` (see `heads.PreStack`) rewrites declared U
    columns from the U and Y windows before `compute_output` sees them, so
    every family consumes the derived feature and none the raw column. It
    runs inside the autoregressive loop: the Y it reads is recorded data in
    the seeded history and the model's own prediction past it.
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
        pre_stack: Optional[PreStack] = None,
    ):
        super().__init__()
        self.fy, self.fu, self.fd = fy, fu, fd
        self.ny, self.nu, self.nd = ny, nu, nd
        self.in_steps = in_steps
        self.residual = residual
        self.head_stack = head_stack
        self.pre_stack = pre_stack

        if separable:
            self.compute_output = self.compute_separable
        else:
            self.compute_output = self.compute_inseparable

    def compute_separable(self, y, u, d):
        return self.fy(y) + self.fu(u) + self.fd(d)

    def compute_inseparable(self, y, u, d):
        return self.fy(torch.cat([y, u, d], dim=-1))

    def _last_y(self, y: torch.Tensor) -> torch.Tensor:
        """The newest step of the Y window, [..., ny]: the window is
        oldest-first, so it is the last ny entries, batched or not."""
        return y[..., -self.ny:]

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
        if self.pre_stack is not None:
            u = self.pre_stack(y, u)
        out = self.pre_activation(y, u, d)
        return out if self.head_stack is None else self.head_stack(out)

    def pre_activation(self, y, u, d):
        """The learned map plus the residual, before any head activates it.

        With heads, the residual goes only to the channels whose head opts in
        (see `HeadStack.apply_residual`). Takes `u` after the pre-stack. The
        sampled gradient penalty differentiates this: every head's activation
        is monotone increasing, so the signs it checks are the same as the
        activated output's, and it stays vmap-safe where a snapping decode
        does not.
        """
        out = self.compute_output(y, u, d)
        if not self.residual:
            return out
        if self.head_stack is None:
            return out + self._last_y(y)
        return self.head_stack.apply_residual(out, self._last_y(y))


# ── Model factory functions ──────────────────────────────────────
#
# Every monolithic factory takes the same (cfg, nm_mapping) so model_catalog
# can dispatch on name alone. These build ONE network over the whole Y
# vector at once; get_dynamics only calls into model_catalog when no block
# is asking to be built separately (see _uses_per_block below) — a family
# that models all outputs jointly has nothing to split, which is exactly
# why `per_block` is refused for it in ModelingConfig.validate().

def _widths(nm_mapping: dict[str, list[str]]) -> tuple[int, int, int]:
    """(ny, nu, nd) — the channel counts every factory starts from."""
    return tuple(len(nm_mapping[key]) for key in ("Y", "U", "D"))


def _mlp(cfg: ModelingConfig, insize: int, outsize: int) -> nm_blocks.MLP:
    """The ReLU MLP every family builds, sized by cfg's hidden width and depth."""
    return nm_blocks.MLP(
        insize=insize, outsize=outsize, bias=True,
        linear_map=nn.Linear, nonlin=nn.ReLU,
        hsizes=cfg.num_layers * [2**cfg.log2_hidden_size],
    )


@beartype
def get_mlp(cfg: ModelingConfig, nm_mapping: dict[str, list[str]]) -> SSMGrad:
    ny, nu, nd = _widths(nm_mapping)
    fy = _mlp(cfg, cfg.in_steps * (ny + nu + nd), ny)
    return SSMGrad(
        fy, None, None, ny, nu, nd,
        cfg.in_steps, cfg.residual, separable=False,
    )


@beartype
def get_nssm(cfg: ModelingConfig, nm_mapping: dict[str, list[str]]) -> SSMGrad:
    ny, nu, nd = _widths(nm_mapping)
    s = cfg.in_steps

    fy, fu, fd = _mlp(cfg, s * ny, ny), _mlp(cfg, s * nu, ny), _mlp(cfg, s * nd, ny)

    return SSMGrad(
        fy, fu, fd, ny, nu, nd,
        s, cfg.residual, separable=True,
    )


@beartype
def get_lssm(cfg: ModelingConfig, nm_mapping: dict[str, list[str]]) -> SSMGrad:
    ny, nu, nd = _widths(nm_mapping)
    s = cfg.in_steps

    fy = nm_blocks.Linear(s*ny, ny, bias=True, linear_map=nn.Linear)
    fu = nm_blocks.Linear(s*nu, ny, bias=True, linear_map=nn.Linear)
    fd = nm_blocks.Linear(s*nd, ny, bias=True, linear_map=nn.Linear)

    return SSMGrad(
        fy, fu, fd, ny, nu, nd,
        s, cfg.residual, separable=True,
    )


@beartype
def get_gru(cfg: ModelingConfig, nm_mapping: dict[str, list[str]]) -> SSMGrad:
    ny, nu, nd = _widths(nm_mapping)
    fy = GRUBlock(
        ny, nu, nd, cfg.in_steps, outsize=ny,
        hidden_size=(2**cfg.log2_hidden_size), num_layers=cfg.num_layers,
    )
    return SSMGrad(
        fy, None, None, ny, nu, nd,
        cfg.in_steps, cfg.residual, separable=False,
    )


#: Families with a monolithic factory: one network over the whole Y vector.
#: `2r2c` has no entry here — an RC network describes *a* block, never all
#: of Y at once (see BLOCK_BUILDERS below and `_uses_per_block`) — so this
#: dict is NOT the model whitelist; see VALID_MODEL_NAMES for that.
model_catalog = {
    "mlp": get_mlp,
    "nssm": get_nssm,
    "lssm": get_lssm,
    "gru": get_gru,
}


# ── Per-block builders ────────────────────────────────────────────
#
# Each builds ONE block's module — the heterogeneous counterpart of the
# monolithic factories above. `nssm`/`lssm` model all of Y jointly
# and have no per-block form, so they are simply absent from this registry;
# PER_BLOCK_FAMILIES (which gates the `per_block` flag) is derived from it
# rather than hand-maintained a second time.

def _mlp_block(cfg: ModelingConfig, block: Block, norm_stats: Optional[dict]) -> nn.Module:
    return _mlp(cfg, cfg.in_steps * (block.ny_in + block.nu + block.nd), block.ny)


def _gru_block(cfg: ModelingConfig, block: Block, norm_stats: Optional[dict]) -> nn.Module:
    return GRUBlock(
        block.ny_in, block.nu, block.nd, cfg.in_steps, outsize=block.ny,
        hidden_size=(2**cfg.log2_hidden_size), num_layers=cfg.num_layers,
    )


def _2r2c_block(cfg: ModelingConfig, block: Block, norm_stats: Optional[dict]) -> nn.Module:
    """One block's RC network. See `rc` for the physics.

    Refuses `reads`: `RCBlock`'s unconditional-stability guarantee holds
    only because its learned forcing map `g` reads `u` and `d` and never
    state — the state matrix is then fixed by three positive rates whatever
    `g` does (see rc.py's module docstring for the eigenvalue argument). A
    `reads` entry would let `g` see another block's Y, which is exactly the
    state-dependence the guarantee forbids, so it is refused here rather
    than silently building a block that looks stable and is not.

    A `demand` declaration is the one state read admitted: it reaches the
    block only as the clamped setpoint error, through the bounded
    `saturating` branch, whose slope the parameterization keeps inside the
    stability headroom — never through `g`.
    """
    if block.reads:
        raise ValueError(
            f"2r2c block {block.name!r} cannot declare `reads`: its "
            "unconditional-stability guarantee holds only because the "
            "learned forcing map g reads u and d and never state. A `reads` "
            "entry would let g see another block's Y — exactly the "
            "state-dependence the guarantee forbids. Use a different family "
            "for this block if it needs to see another block's state."
        )
    if not block.params:
        raise ValueError(
            f"2r2c block {block.name!r} declares no time constants. Give its "
            f"block_mapping entry `params` with {list(TAU_KEYS)}, in hours — a "
            "scalar pins one, [min, max] bounds it. They are this zone's "
            "storage prior, and there is no library default for a building. "
            "With no block_mapping hook there is nowhere to declare them: "
            "`model: 2r2c` needs one."
        )
    return RCBlock(
        block.ny, block.nu, block.nd, cfg.in_steps,
        steps_per_hour=cfg.steps_per_hour,
        hidden_size=(2**cfg.log2_hidden_size),
        num_layers=cfg.num_layers,
        taus=time_constant_boxes(block.params, cfg.steps_per_hour, block.name),
        channel_names=block.y_names,
        u_names=block.u_names, d_names=block.d_names,
        inputs=block.inputs, ambient=block.ambient, norm_stats=norm_stats,
    )


BLOCK_BUILDERS: dict[str, Callable[[ModelingConfig, Block, Optional[dict]], nn.Module]] = {
    "mlp": _mlp_block,
    "gru": _gru_block,
    "2r2c": _2r2c_block,
}

#: Families that need `per_block: true` to opt into the per-block split.
#: `2r2c` is excluded even though BLOCK_BUILDERS has a builder for it: it is
#: per-block unconditionally, so it needs no flag (and
#: ModelingConfig.validate() refuses the combination as redundant).
PER_BLOCK_FAMILIES = tuple(name for name in BLOCK_BUILDERS if name != "2r2c")

#: Every valid `cfg.model` name: the union of model_catalog (monolithic
#: factories) and BLOCK_BUILDERS (per-block-only factories — today just
#: `2r2c`, which has no monolithic form at all). This is the whitelist
#: ModelingConfig.validate()'s check_one_of and get_dynamics's "unknown
#: model" check both read, so a new family only means adding it to one of
#: the two dicts above, never a third hand-maintained list.
VALID_MODEL_NAMES = tuple(dict.fromkeys([*model_catalog, *BLOCK_BUILDERS]))


def _resolve_family(cfg: ModelingConfig, block: Block) -> str:
    """The model family one block builds under: its own declared `model`,
    or `cfg.model` when it names none."""
    return block.model or cfg.model


@beartype
def family_by_channel(
    cfg: ModelingConfig,
    nm_mapping: dict[str, list[str]],
    block_spec: Optional[dict[str, dict]] = None,
) -> dict[str, str]:
    """Which model family produces each Y channel: `{column: family}`.

    Built from the same `build_blocks` + `_resolve_family` pair `get_dynamics`
    uses, so a caller asking "what builds this channel?" can never disagree
    with what actually gets built. That matters for `nt data heads`, which
    warns about head/family conflicts per channel: with mixed families,
    `cfg.model` is the wrong answer for every block that declares its own.

    With no block spec there is one block holding every channel, so every
    column maps to `cfg.model`.
    """
    return {
        name: _resolve_family(cfg, block)
        for block in build_blocks(nm_mapping, block_spec)
        for name in block.y_names
    }


def _build_block(cfg: ModelingConfig, block: Block,
                 norm_stats: Optional[dict]) -> nn.Module:
    """Dispatch one block to its family's per-block builder."""
    family = _resolve_family(cfg, block)
    builder = BLOCK_BUILDERS.get(family)
    if builder is None:
        raise ValueError(
            f"block {block.name!r} requests model {family!r}, which has no "
            f"per-block builder; available: {sorted(BLOCK_BUILDERS)}."
        )
    return builder(cfg, block, norm_stats)


def _build_per_block_model(
    cfg: ModelingConfig, nm_mapping: dict[str, list[str]],
    blocks: list[Block], families: list[str], norm_stats: Optional[dict],
) -> SSMGrad:
    """Heterogeneous dynamics: one sub-module per block, each in whatever
    family it (or cfg.model, absent a block-level override) declares.

    This is the general form `2r2c` always needed on its own — an RC
    network describes *a* block, never the whole Y vector at once — so a
    spec mixing `2r2c` with `mlp` takes exactly the same path, just with
    more than one family among `blocks`. `2r2c`'s residual/steps_per_hour
    requirements are checked once here rather than per block, because
    SSMGrad.residual gates the whole output, not a per-channel one (see
    SSMGrad.forward).
    """
    ny, nu, nd = _widths(nm_mapping)

    if "2r2c" in families:
        if not cfg.residual:
            raise ValueError(
                "2r2c requires residual=True: the RC block predicts the "
                "temperature *increment*, which SSMGrad adds to the current "
                "state. With residual=False the increment would be read as "
                "an absolute temperature."
            )
        if cfg.steps_per_hour is None:
            raise ValueError(
                "2r2c needs cfg.steps_per_hour to place its time constants. "
                "The CLI sets it from the data's sampling frequency; set it "
                "by hand (steps_per_hour = 3600 / timestep_seconds) if you "
                "are building a model outside the CLI."
            )

    fy = BlockStack(blocks, (ny, nu, nd), cfg.in_steps,
                    make_block=lambda block: _build_block(cfg, block, norm_stats))
    return SSMGrad(
        fy, None, None, ny, nu, nd,
        cfg.in_steps, cfg.residual, separable=False,
    )


# ── Model description (for reportcard) ───────────────────────────

def _count_params(module: Optional[nn.Module]) -> int:
    if module is None:
        return 0
    return sum(p.numel() for p in module.parameters())


def _describe_block(block: Optional[nn.Module], cfg: ModelingConfig) -> Optional[dict]:
    """Return a small dict describing an SSMGrad sub-block (fy/fu/fd)."""
    if block is None:
        return None
    info = {
        "class": type(block).__name__,
        "param_count": _count_params(block),
    }
    # neuromancer's MLP has no `hsizes` attribute of its own (only `nhidden`,
    # the layer count) — the per-layer widths have to be read off `linear`,
    # its ModuleList of nn.Linear-likes, one per layer including the output.
    if isinstance(block, nm_blocks.MLP):
        info["hsizes"] = [int(lin.out_features) for lin in block.linear[:-1]]
    # GRUBlock has no hsizes (nn.GRU takes hidden_size/num_layers directly),
    # so report those explicitly or a recurrent block's width never shows up.
    if hasattr(block, "gru"):
        info["hidden_size"] = block.gru.hidden_size
        info["num_layers"] = block.gru.num_layers
    if hasattr(block, "in_features"):
        info["in_features"] = block.in_features
    if hasattr(block, "out_features"):
        info["out_features"] = block.out_features
    if isinstance(block, BlockStack):
        info["blocks"] = {}
        for spec, sub in zip(block.block_specs, block.blocks):
            sub_info = _describe_block(sub, cfg)
            # The family is the interesting fact once blocks can mix —
            # spec.model is None for a block that just inherits cfg.model.
            sub_info["model"] = _resolve_family(cfg, spec)
            info["blocks"][spec.name] = sub_info
    # A block that can say something about itself in its own vocabulary —
    # RCBlock reports the time constants it learned — says it here.
    if hasattr(block, "describe"):
        info.update(block.describe())
    # Every bounded parameter it owns, generically: its box and where the
    # fit landed in it. A stack's live in its blocks' own entries.
    if not isinstance(block, BlockStack):
        bounded = {b.name: b.describe() for b in bounded_parameters(block)}
        if bounded:
            info["bounded"] = bounded
    return info


@beartype
def describe_dynamics(dyn_sys: SystemPreview, cfg: ModelingConfig) -> dict:
    """
    Describe the trained dynamics system for the reportcard.

    Returns a dict with:
      - factory: the model family, or `mixed (a, b)` when the blocks of a
        block spec use different families (each block line names its own)
      - separable: whether the SSM is separable (fy + fu + fd)
      - residual: whether y_{k+1} adds y_k as a residual base
      - in_steps: context length (cfg.in_steps) — see below
      - ny / nu / nd: shape parameters
      - param_count: total trainable parameters
      - blocks: per-block descriptors {fy, fu, fd}
      - pre_stack: the derived-input declarations, or None

    Used by reportcard.render_reportcard.
    """
    # Unwrap: SystemPreview -> Node -> SSMGrad
    raw = dyn_sys.nodes[0].callable

    blocks_info: dict[str, Optional[dict]] = {
        "fy": _describe_block(getattr(raw, "fy", None), cfg),
        "fu": _describe_block(getattr(raw, "fu", None), cfg),
        "fd": _describe_block(getattr(raw, "fd", None), cfg),
    }

    # cfg.model is only the default family under a block spec; the blocks
    # themselves say what was built.
    fy = blocks_info["fy"]
    families = (sorted({sub["model"] for sub in fy["blocks"].values()})
                if fy is not None and "blocks" in fy else [cfg.model])
    factory = families[0] if len(families) == 1 else f"mixed ({', '.join(families)})"

    pre_stack = getattr(raw, "pre_stack", None)
    return {
        "factory": factory,
        "separable": raw.fu is not None and raw.fd is not None,
        "residual": bool(raw.residual),
        "in_steps": int(raw.in_steps),
        "ny": int(raw.ny),
        "nu": int(raw.nu),
        "nd": int(raw.nd),
        "param_count": _count_params(raw),
        "blocks": blocks_info,
        # Derived inputs, `{u_column: {kind, state, ...}}`; None when no
        # column is rewritten.
        "pre_stack": pre_stack.describe() if pre_stack is not None else None,
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
    blocks = stateful_blocks(model)
    if not blocks:
        return None

    def reset_init(data):
        for block in blocks:
            block.set_rollout_state(None)
        return data

    return reset_init


def stateful_blocks(model: nn.Module) -> list[nn.Module]:
    """Every submodule carrying rollout state, in `modules()` order.

    A block is stateful exactly when it implements the protocol
    `rollout_state()` / `set_rollout_state(state)` (RCBlock, GRUBlock): the
    state as one [batch, width] tensor, None when cold. A new stateful family
    implements those two methods and gets both the per-batch reset and ONNX
    export (see `neurotemplate.export.state`) for free.
    """
    return [m for m in model.modules() if hasattr(m, "set_rollout_state")]


def _uses_per_block(cfg: ModelingConfig, blocks: list[Block], families: list[str]) -> bool:
    """Whether get_dynamics should build heterogeneously (BlockStack) rather
    than through the monolithic model_catalog factories.

    Three independent reasons, any one of which is enough: the spec itself
    names a per-block family (mixed families in one graph), the run opted in
    with `cfg.per_block`, or a resolved family is `2r2c`, which is per-block
    unconditionally — an RC network describes *a* block, never the whole Y
    vector at once, so there is no monolithic 2r2c to fall back to.
    """
    return (any(b.model is not None for b in blocks)
            or cfg.per_block or "2r2c" in families)


def _check_2r2c_heads(blocks: list[Block], families: list[str],
                      head_stack: HeadStack) -> None:
    """2r2c predicts an increment; only a residual-compatible head restores
    the state. Scoped to the channels a 2r2c block actually produces, so a
    mixed-family run's other channels are free to use any head — see
    _build_per_block_model for the analogous residual/steps_per_hour checks,
    which are model-wide because SSMGrad's residual flag is not per-channel.
    """
    rc_channels = {
        name for block, family in zip(blocks, families) if family == "2r2c"
        for name in block.y_names
    }
    if not rc_channels:
        return
    blind = [name for name, head in zip(head_stack.channel_names, head_stack.heads)
             if name in rc_channels and not head.residual_compatible]
    if blind:
        raise ValueError(
            f"2r2c predicts increments, so every Y channel it produces needs "
            f"a residual-compatible head; {blind} are not."
        )


def _build_pre_stack(
    blocks: list[Block], nm_mapping: dict[str, list[str]],
    norm_stats: Optional[dict], in_steps: int,
) -> Optional[PreStack]:
    """One `PreStack` over every block's `demand` declarations, or None.

    The stack is block-agnostic — each entry is resolved to a global
    `(U column, Y channel)` pair — which is what lets one wiring point in
    `SSMGrad.forward` serve every family.
    """
    # `branch` is where a 2r2c block routes the feature; the transform itself
    # does not carry it.
    demands = {
        column: {k: v for k, v in entry.items() if k != "branch"}
        for block in blocks
        for column, entry in block.inputs.items() if entry["kind"] == "demand"
    }
    if not demands:
        return None
    if norm_stats is None:
        raise ValueError(
            f"a `demand` declaration ({sorted(demands)}) needs norm_stats to "
            "place the setpoint and the zone in physical units; pass "
            "norm_stats to get_dynamics."
        )
    return PreStack.from_config(
        demands, nm_mapping["Y"], nm_mapping["U"], norm_stats, in_steps)


def _warn_ignored_declarations(blocks: list[Block], families: list[str]) -> None:
    """`linear` and `ambient` only mean something to `2r2c`; every other
    family already sees the column as an ordinary input. Say so rather than
    let the declaration sit inert."""
    for block, family in zip(blocks, families):
        if family == "2r2c":
            continue
        ignored = [c for c, e in block.inputs.items() if e["kind"] == "linear"]
        if block.ambient:
            ignored.append(block.ambient)
        if ignored:
            warnings.warn(
                f"block {block.name!r} builds under {family!r}, which ignores "
                f"the `linear`/`ambient` declaration of {ignored}: only 2r2c "
                "gives those columns a structural term.",
                stacklevel=3,
            )


@beartype
def get_dynamics(
    cfg: ModelingConfig,
    nm_mapping: dict[str, list[str]],
    head_stack: Optional[HeadStack] = None,
    block_spec: Optional[dict[str, dict]] = None,
    norm_stats: Optional[dict[str, dict]] = None,
) -> SystemPreview:
    """
    Wrap a dynamics model in a Neuromancer SystemPreview for autoregressive rollout.

    The Node reads ['yn', 'U', 'D'] and writes ['yn']. Because 'yn' is both
    input and output, the System feeds predictions back as input at the next
    timestep — this is the autoregressive loop.

    SystemPreview handles temporal windowing via input_map: at each rollout
    step i, the Node receives a window of model.in_steps past values for each
    input, flattened to [batch, model.in_steps*features].

    `block_spec` is the optional `block_mapping` hook's result, splitting Y,
    U and D across independent blocks — see `blocks`. None means one block
    holding every channel, which is the right answer for a single-block
    building. `norm_stats` is needed only when the spec declares a column
    (`inputs` / `ambient`): those branches work in physical units.
    """
    if cfg.model not in VALID_MODEL_NAMES:
        raise ValueError(
            f"Unknown model '{cfg.model}'. "
            f"Available: {list(VALID_MODEL_NAMES)}"
        )

    blocks = build_blocks(nm_mapping, block_spec)
    families = [_resolve_family(cfg, b) for b in blocks]

    _warn_ignored_declarations(blocks, families)
    unread = [b.name for b, family in zip(blocks, families)
              if b.params and family != "2r2c"]
    if unread:
        raise ValueError(
            f"block(s) {unread} declare `params`, but only a 2r2c block reads "
            "any; the declaration would sit inert."
        )
    if _uses_per_block(cfg, blocks, families):
        model = _build_per_block_model(cfg, nm_mapping, blocks, families, norm_stats)
    else:
        model = model_catalog[cfg.model](cfg, nm_mapping)
    model.pre_stack = _build_pre_stack(blocks, nm_mapping, norm_stats, cfg.in_steps)

    # Heads are attached post-construction so each factory stays focused
    # on its own block topology. Every head has width 1, so the factories'
    # output width ny is also the stack's raw width.
    if head_stack is not None:
        if head_stack.raw_width != model.ny:
            raise ValueError(
                f"head_stack.raw_width={head_stack.raw_width} does not match "
                f"model.ny={model.ny}; factories (currently) assume width-1 heads."
            )
        _check_2r2c_heads(blocks, families, head_stack)
        # The System owns its own copy: post-training calibration writes the
        # decision thresholds into the stack it trained with, and a caller
        # reusing one stack across seeds or trials must not inherit them.
        model.head_stack = copy.deepcopy(head_stack)

    # Temporal windowing: the gathered window must be exactly as wide as the
    # block's declared input width, and model.in_steps is that width.
    input_map = {
        key: {
            "past": model.in_steps - 1,
            "future": 0,
            "pad_mode": "nearest",
        }
        for key in ["yn", "U", "D"]
    }

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
