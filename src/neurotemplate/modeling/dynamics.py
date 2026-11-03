"""
The built-in dynamics families, behind the model contract, wrapped for
Neuromancer.

Every family is an SSMGrad computing in normalised units, put behind the
physical contract by `contract.Normalized` (see `contract`).

SSMGrad is a pure predictor: (y, u, d) -> y_next. It does NOT compute
Jacobians in its forward pass — gradient sign constraints are handled
externally by SampledGradientPenalty (see gradient_sampling.py).

get_dynamics() wraps SSMGrad in a Neuromancer Node + SystemPreview:

  - The Node uses input_keys=['yn', 'U', 'D'] and output_keys=['yn'].
    Because 'yn' appears in BOTH input and output, the System creates
    an autoregressive feedback loop: at each timestep, the model reads
    its own previous prediction and produces the next one.

  - SystemPreview with start_iter=history_steps-1 skips the first timesteps
    (the model needs history_steps of history). The input_map on the Node
    handles temporal windowing: at each step i, the model receives
    data[:, i-past:i+1, :] flattened.

  - The data loader provides 'yn' initialized to the first history_steps
    of ground truth (see data_utils.get_sw_dict). The System then
    appends model predictions to 'yn' via cat() at each rollout step.

  - 'Y' (ground truth) is only used by the loss for comparison and
    by the System to determine rollout length (nstep_key='Y').

`cfg.history` sets each column's lag, the steps before the current one the
model reads. The Node hands every key one rectangular window,
`cfg.history_steps = max(lag) + 1` rows deep; each learned map then gathers
the cells it reads out of it, each column at the lag its family consumes (see
`block_lags`): any lag for `mlp`, `nssm` and `lssm`, one lag for every column
a `gru` reads (its window is its warm-up), and for `rc_network` one window of
its Y, D and a commanded heat, and the current step of a setpoint. A uniform
table gathers nothing. A stateful
block (GRU) additionally keeps a hidden state so its memory can run past its
window during a rollout, but it still takes the same window — it just skips
the steps already folded into that state (see gru.GRUBlock). The only thing
statefulness costs the caller is a reset at each sequence boundary, which
make_reset_init() wires into the System.

Because cfg.history_steps is also the sliding-window stride
(data_utils.get_sw_dict), two models with the same deepest lag see the same
training set.
"""

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

from neurotemplate.hooks import load_modeling_hooks
from neurotemplate.modeling import windows
from neurotemplate.modeling.contract import Normalized, Windowed, contract_model, normalized_core
from neurotemplate.modeling.heads import HeadStack, RegressionHead
from neurotemplate.modeling.prestack import PreStack
from neurotemplate.modeling.bounded import bounded_parameters
from neurotemplate.modeling.gru import GRUBlock
from neurotemplate.modeling.rollout import WindowedSystem
from neurotemplate.modeling.blocks import Block, build_blocks
from neurotemplate.modeling.rc_network import NetworkBlock


# ── Per-column lags ──────────────────────────────────────────────

#: One lag per column of each key, `(Y, U, D)`, in the order the window
#: carries the columns.
Lags = tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]

_KEYS = ("Y", "U", "D")


@beartype
def column_lags(cfg: ModelingConfig, nm_mapping: dict[str, list[str]]) -> Lags:
    """`cfg.history` as one lag per column, in nm_mapping order.

    The table must name exactly the model's columns: a missing one has no
    lag, and an extra one is a declaration nothing reads.
    """
    columns = [c for key in _KEYS for c in nm_mapping[key]]
    missing = [c for c in columns if c not in cfg.history]
    extra = sorted(set(cfg.history) - set(columns))
    if missing or extra:
        raise ValueError(
            "history must list every Y, U and D column of the model and "
            f"nothing else. Missing: {missing}. Not a model column: {extra}."
        )
    return tuple(tuple(cfg.history[c] for c in nm_mapping[key]) for key in _KEYS)


def _check_one_lag(what: str, lags: dict[str, int], reason: str) -> None:
    if len(set(lags.values())) > 1:
        raise ValueError(f"{what} needs one lag for {reason}; history gives {lags}.")


@beartype
def block_lags(family: str, block: Block, history: dict[str, int]) -> Lags:
    """The lags `block` consumes, per column of its window: `y_in_names`,
    `u_names`, `d_names`.

    `mlp` reads every column at its declared lag. `gru` folds its window
    into its hidden state row by row, so every column it reads shares one
    lag. `rc_network` filters one window of its Y and D, and of its heat
    when the heat is a U, and reads its setpoints at the current step only,
    whatever they are declared at.
    """
    y, u, d = (tuple(history[c] for c in names)
               for names in (block.y_in_names, block.u_names, block.d_names))
    if family == "gru":
        _check_one_lag(
            f"gru block {block.name!r}",
            {c: history[c] for c in block.y_in_names + block.u_names + block.d_names},
            "every column it reads: its window is its warm-up")
    if family == "rc_network":
        window = [c for c in block.y_in_names + block.u_names + block.d_names
                  if c not in block.u_names or c == block.heat]
        _check_one_lag(f"rc_network block {block.name!r}",
                       {c: history[c] for c in window},
                       "its zone, its heat and its D: the window its filter runs over")
        return y, tuple(history[c] if c == block.heat else 0 for c in block.u_names), d
    return y, u, d


def _check_lags_consumed(blocks: list[Block], lags: list[Lags],
                         history: dict[str, int]) -> None:
    """Every column's declared lag is what some block reading it consumes.

    A deeper declaration would sit inert: a U column read only by
    rc_network blocks as a setpoint is read at its current step, whatever its lag says. A D column
    no block reads (per-block D subsets that all leave it out) is listed at
    0: the control side may still use it, so it stays a model column.
    """
    consumed: dict[str, dict[str, int]] = {c: {} for c in history}
    for block, (y, u, d) in zip(blocks, lags):
        names = block.y_in_names + block.u_names + block.d_names
        for column, lag in zip(names, y + u + d):
            consumed[column][block.name] = lag
    for column, readers in consumed.items():
        if not readers and history[column] != 0:
            raise ValueError(
                f"history[{column!r}] is {history[column]}, but no block reads "
                "it. If nothing on the control side needs it either (the "
                "policy, a loss weight, power_metric, the plots), drop it with "
                "exclude_columns; otherwise list it at 0."
            )
        deepest = max(readers.values(), default=0)
        if deepest != history[column]:
            raise ValueError(
                f"history[{column!r}] is {history[column]}, but no block reads "
                f"it that far back; the lag each reader consumes: {readers}. A "
                f"rc_network block reads a setpoint at the current step only. Set it to "
                f"{deepest}."
            )


def _check_demand_lags(blocks: list[Block], history: dict[str, int]) -> None:
    """A demand setpoint's lag is at most its state's.

    The pre-stack computes the demand on the full window, so a setpoint read
    further back than its zone would carry zone history older than the
    zone's own declared lag into the block.
    """
    for block in blocks:
        for column, entry in block.inputs.items():
            state = entry.get("state")
            if entry["kind"] == "demand" and history[column] > history[state]:
                raise ValueError(
                    f"history[{column!r}] is {history[column]}, deeper than its "
                    f"demand state history[{state!r}] = {history[state]}. The "
                    "demand feature reads the state at every step it covers; "
                    f"give {column!r} a lag of at most {history[state]}."
                )


def _window_keep(columns: tuple[int, ...], lags: tuple[int, ...],
                 history_steps: int) -> tuple[tuple[int, int, int], ...]:
    """`windows.window_gather` rows for columns read at `lags` out of a
    window `history_steps` deep: each keeps its newest `lag + 1` rows."""
    newest = history_steps - 1
    return tuple((j, newest - lag, newest) for j, lag in zip(columns, lags))


def _key_gather(lags: tuple[int, ...], history_steps: int) -> Optional[torch.Tensor]:
    """The gather narrowing one key's window to its columns' lags, or None
    when every column reads the whole window."""
    indices = windows.window_gather(
        (len(lags),), (history_steps,),
        (_window_keep(tuple(range(len(lags))), lags, history_steps),))
    if indices == list(range(len(lags) * history_steps)):
        return None
    return torch.tensor(indices, dtype=torch.long)


def _window_size(lags: tuple[int, ...]) -> int:
    """Cells of one key read at `lags`."""
    return sum(lag + 1 for lag in lags)


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

    Each block's gather also narrows in time: a column arrives at the lag the
    block's family consumes (see `block_lags`), so an rc_network block gets
    one step of U.

    Args:
        blocks: the partition, from `blocks.build_blocks`.
        widths: `(ny, nu, nd)` of the full mapping.
        history_steps: rows of the shared window.
        lags: per block, the lags it consumes, from `block_lags`.
        make_block: `(block, lags) -> nn.Module` for one block's sub-block.
            The block carries its own shape (`ny`, `nu`, `nd`, `ny_in`) and
            channel names. Called once per block, so every block gets its
            own parameters.
    """

    def __init__(self, blocks: list[Block], widths: tuple[int, int, int],
                 history_steps: int, lags: list["Lags"],
                 make_block: Callable[[Block, "Lags"], nn.Module]):
        super().__init__()
        ny = widths[0]
        self.block_specs = list(blocks)
        self.block_lags = list(lags)
        self.in_features = history_steps * sum(widths)
        self.out_features = ny

        self.blocks = nn.ModuleList([make_block(b, l) for b, l in zip(blocks, lags)])

        # Every block's gather indices live in one buffer, so they follow the
        # module onto its device together; `_spans` says which stretch of it
        # belongs to which block.
        gathers, self._spans, cursor = [], [], 0
        for block, consumed in zip(blocks, lags):
            indices = windows.window_gather(
                widths, (history_steps,) * 3,
                tuple(_window_keep(idx, key_lags, history_steps) for idx, key_lags
                      in zip((block.y_in_idx, block.u_idx, block.d_idx), consumed)))
            gathers += indices
            self._spans.append(slice(cursor, cursor + len(indices)))
            cursor += len(indices)
        self.register_buffer(
            "_gather", torch.tensor(gathers, dtype=torch.long), persistent=False)

        # Sub-block outputs concatenate in execution order; this puts them
        # back in the order nm_mapping["Y"] lists, which is the order the
        # loss and the autoregressive feedback both assume.
        exec_order = [i for block in blocks for i in block.y_idx]
        self.register_buffer(
            "_to_y_order",
            torch.tensor(
                [exec_order.index(i) for i in range(ny)], dtype=torch.long
            ),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dim = x.ndim - 1                    # the feature axis, batched or not
        outs = [block(x.index_select(dim, self._gather[span]))
                for block, span in zip(self.blocks, self._spans)]
        return torch.cat(outs, dim=-1).index_select(dim, self._to_y_order)



class SSMGrad(nn.Module):
    """
    State space model for discrete-time dynamics:
        y_{k+1} = f(y_k, u_k, d_k)  [+ y_k if residual]

    Two modes:
        Separable:   y_{k+1} = fy(y_k) + fu(u_k) + fd(d_k)
        Inseparable: y_{k+1} = fy(cat(y_k, u_k, d_k))

    This is a pure predictor — no Jacobians computed here. The sampled
    gradient penalty differentiates `pre_activation` externally via jacrev.

    An optional `pre_stack` (see `prestack.PreStack`) rewrites declared U
    columns from the U and Y windows before `compute_output` sees them, so
    every family consumes the derived feature and none the raw column. It
    runs inside the autoregressive loop: the Y it reads is recorded data in
    the seeded history and the model's own prediction past it.

    `lags`, for a monolithic family, narrows each key's window to its
    columns' lags before the learned maps see it, after the pre-stack; the
    residual still reads the newest step of the full Y window. None when
    `fy` is a `BlockStack`, which gathers per block.
    """

    def __init__(
        self,
        fy: Callable,
        fu: Optional[Callable],
        fd: Optional[Callable],
        ny: int,
        nu: int,
        nd: int,
        history_steps: int,
        residual: bool,
        separable: bool,
        head_stack: Optional[HeadStack] = None,
        pre_stack: Optional[PreStack] = None,
        lags: Optional[Lags] = None,
    ):
        super().__init__()
        self.fy, self.fu, self.fd = fy, fu, fd
        self.ny, self.nu, self.nd = ny, nu, nd
        self.history_steps = history_steps
        self.residual = residual
        self.head_stack = head_stack
        self.pre_stack = pre_stack

        # One gather per key whose columns do not all read the whole window.
        self._gathered = []
        for key, key_lags in zip(_KEYS, lags or ()):
            gather = _key_gather(key_lags, history_steps)
            if gather is not None:
                self.register_buffer(f"_gather_{key}", gather, persistent=False)
                self._gathered.append(key)

        if separable:
            self.compute_output = self.compute_separable
        else:
            self.compute_output = self.compute_inseparable

    def compute_separable(self, y, u, d):
        return self.fy(y) + self.fu(u) + self.fd(d)

    def compute_inseparable(self, y, u, d):
        return self.fy(torch.cat([y, u, d], dim=-1))

    def _narrow(self, key: str, x: torch.Tensor) -> torch.Tensor:
        """One key's window, narrowed to its columns' lags if it gathers."""
        if key not in self._gathered:
            return x
        return x.index_select(x.ndim - 1, getattr(self, f"_gather_{key}"))

    def _last_y(self, y: torch.Tensor) -> torch.Tensor:
        """The newest step of the Y window, [..., ny]: the window is
        oldest-first, so it is the last ny entries, batched or not."""
        return y[..., -self.ny:]

    def forward(self, y, u, d):
        """
        Forward pass: predict next state from current inputs.

        Args:
            y: [batch, history_steps*ny]
            u: [batch, history_steps*nu]
            d: [batch, history_steps*nd]

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
        out = self.compute_output(
            self._narrow("Y", y), self._narrow("U", u), self._narrow("D", d))
        if not self.residual:
            return out
        if self.head_stack is None:
            return out + self._last_y(y)
        return self.head_stack.apply_residual(out, self._last_y(y))


# ── Model factory functions ──────────────────────────────────────
#
# Every monolithic factory takes the same (cfg, nm_mapping, lags) so model_catalog
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
def get_mlp(cfg: ModelingConfig, nm_mapping: dict[str, list[str]], lags: Lags) -> SSMGrad:
    ny, nu, nd = _widths(nm_mapping)
    fy = _mlp(cfg, sum(_window_size(key_lags) for key_lags in lags), ny)
    return SSMGrad(
        fy, None, None, ny, nu, nd,
        cfg.history_steps, cfg.residual, separable=False, lags=lags,
    )


@beartype
def get_nssm(cfg: ModelingConfig, nm_mapping: dict[str, list[str]], lags: Lags) -> SSMGrad:
    ny, nu, nd = _widths(nm_mapping)
    fy, fu, fd = (_mlp(cfg, _window_size(key_lags), ny) for key_lags in lags)
    return SSMGrad(
        fy, fu, fd, ny, nu, nd,
        cfg.history_steps, cfg.residual, separable=True, lags=lags,
    )


@beartype
def get_lssm(cfg: ModelingConfig, nm_mapping: dict[str, list[str]], lags: Lags) -> SSMGrad:
    ny, nu, nd = _widths(nm_mapping)
    fy, fu, fd = (nm_blocks.Linear(_window_size(key_lags), ny, bias=True,
                                   linear_map=nn.Linear)
                  for key_lags in lags)
    return SSMGrad(
        fy, fu, fd, ny, nu, nd,
        cfg.history_steps, cfg.residual, separable=True, lags=lags,
    )


@beartype
def get_gru(cfg: ModelingConfig, nm_mapping: dict[str, list[str]], lags: Lags) -> SSMGrad:
    # block_lags has checked every column shares one lag, so the window is
    # history_steps deep and gathers nothing.
    ny, nu, nd = _widths(nm_mapping)
    fy = GRUBlock(
        ny, nu, nd, cfg.history_steps, outsize=ny,
        hidden_size=(2**cfg.log2_hidden_size), num_layers=cfg.num_layers,
    )
    return SSMGrad(
        fy, None, None, ny, nu, nd,
        cfg.history_steps, cfg.residual, separable=False, lags=lags,
    )


#: Families with a monolithic factory: one network over the whole Y vector.
#: `rc_network` has no entry here — it describes *a* zone, never all of Y
#: at once (see BLOCK_BUILDERS below and `_uses_per_block`) — so this
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

def _mlp_block(cfg: ModelingConfig, block: Block, lags: Lags,
               norm_stats: Optional[dict]) -> nn.Module:
    return _mlp(cfg, sum(_window_size(key_lags) for key_lags in lags), block.ny)


def _gru_block(cfg: ModelingConfig, block: Block, lags: Lags,
               norm_stats: Optional[dict]) -> nn.Module:
    # One lag for every column (block_lags); a block always reads its own Y.
    return GRUBlock(
        block.ny_in, block.nu, block.nd, lags[0][0] + 1, outsize=block.ny,
        hidden_size=(2**cfg.log2_hidden_size), num_layers=cfg.num_layers,
    )


def _rc_network_block(cfg: ModelingConfig, block: Block, lags: Lags,
                      norm_stats: Optional[dict]) -> nn.Module:
    """One thermal zone's declared network, behind a thermostat when its heat
    is a Y. See `rc_network`."""
    problems = []
    if block.heat is None or block.ambient is None:
        problems.append("declare its `heat` (a Y or U column, kW) and `ambient` (a D column)")
    if block.reads or block.inputs:
        problems.append("`reads` and `inputs` have no meaning here: the network and the "
                        "thermostat say where every column goes")
    if block.heat in block.u_names:
        if len(block.y_names) != 1 or block.u_names != (block.heat,):
            problems.append(f"with the heat a U, Y must be the zone temperature and U the "
                            f"heat alone; got Y {list(block.y_names)}, U {list(block.u_names)}")
    elif len(block.y_names) != 2:
        problems.append(f"Y must be the zone temperature and the heat; got {list(block.y_names)}")
    if problems:
        raise ValueError(f"rc_network block {block.name!r}: " + "; ".join(problems))
    return NetworkBlock(
        block.name, block.y_names, block.u_names, block.d_names, block.heat, block.ambient,
        block.params, window=lags[0][0] + 1, dt_h=1.0 / cfg.steps_per_hour,
        norm_stats=norm_stats)


BLOCK_BUILDERS: dict[str, Callable[[ModelingConfig, Block, Lags, Optional[dict]],
                                    nn.Module]] = {
    "mlp": _mlp_block,
    "gru": _gru_block,
    "rc_network": _rc_network_block,
}

#: Families that need `per_block: true` to opt into the per-block split.
#: `rc_network` is excluded even though BLOCK_BUILDERS has a builder for it:
#: it describes one thermal zone and is per-block unconditionally, so it
#: needs no flag (and ModelingConfig.validate() refuses the combination).
PER_BLOCK_FAMILIES = tuple(name for name in BLOCK_BUILDERS if name != "rc_network")

#: Every valid `cfg.model` name: the union of model_catalog (monolithic
#: factories) and BLOCK_BUILDERS (per-block-only factories — today just
#: `rc_network`, which has no monolithic form at all). This is the whitelist
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


def _build_block(cfg: ModelingConfig, block: Block, lags: Lags,
                 norm_stats: Optional[dict]) -> nn.Module:
    """Dispatch one block to its family's per-block builder."""
    family = _resolve_family(cfg, block)
    builder = BLOCK_BUILDERS.get(family)
    if builder is None:
        raise ValueError(
            f"block {block.name!r} requests model {family!r}, which has no "
            f"per-block builder; available: {sorted(BLOCK_BUILDERS)}."
        )
    return builder(cfg, block, lags, norm_stats)


def _build_per_block_model(
    cfg: ModelingConfig, nm_mapping: dict[str, list[str]],
    blocks: list[Block], families: list[str], lags: list[Lags],
    norm_stats: Optional[dict],
) -> SSMGrad:
    """Heterogeneous dynamics: one sub-module per block, each in whatever
    family it (or cfg.model, absent a block-level override) declares.

    This is the general form `rc_network` always needs on its own — it
    describes *a* zone, never the whole Y vector at once — so a spec mixing
    it with `mlp` takes exactly the same path, just with more than one
    family among `blocks`. Its residual/steps_per_hour requirements are
    checked once here rather than per block, because
    SSMGrad.residual gates the whole output, not a per-channel one (see
    SSMGrad.forward).
    """
    ny, nu, nd = _widths(nm_mapping)

    if "rc_network" in families:
        if not cfg.residual:
            raise ValueError(
                "rc_network requires residual=True: the block predicts the "
                "*increment*, which SSMGrad adds to the current state. With "
                "residual=False the increment would be read as an absolute value."
            )
        if cfg.steps_per_hour is None:
            raise ValueError(
                "rc_network needs cfg.steps_per_hour to discretise its network. "
                "The CLI sets it from the data's sampling frequency; set it "
                "by hand (steps_per_hour = 3600 / timestep_seconds) if you "
                "are building a model outside the CLI."
            )

    fy = BlockStack(
        blocks, (ny, nu, nd), cfg.history_steps, lags,
        make_block=lambda block, consumed: _build_block(cfg, block, consumed,
                                                        norm_stats))
    return SSMGrad(
        fy, None, None, ny, nu, nd,
        cfg.history_steps, cfg.residual, separable=False,
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
        for spec, sub, (y, u, d) in zip(block.block_specs, block.blocks,
                                        block.block_lags):
            sub_info = _describe_block(sub, cfg)
            # The family is the interesting fact once blocks can mix —
            # spec.model is None for a block that just inherits cfg.model.
            sub_info["model"] = _resolve_family(cfg, spec)
            # What it reads, which for rc_network is not what history declares.
            sub_info["lags"] = dict(zip(
                spec.y_in_names + spec.u_names + spec.d_names, y + u + d))
            info["blocks"][spec.name] = sub_info
    # A block that can say something about itself in its own vocabulary —
    # NetworkBlock reports the time constants it learned — says it here.
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
      - history: each column's lag, and history_steps, the window depth
      - ny / nu / nd: shape parameters
      - param_count: total trainable parameters
      - blocks: per-block descriptors {fy, fu, fd}
      - pre_stack: the derived-input declarations, or None

    Used by reportcard.render_reportcard.
    """
    raw = normalized_core(dyn_sys)
    if raw is None:
        return _describe_custom_model(contract_model(dyn_sys), dyn_sys.nodes[0].callable.widths,
                                    cfg)

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
        "history": dict(cfg.history),
        "history_steps": int(raw.history_steps),
        "ny": int(raw.ny),
        "nu": int(raw.nu),
        "nd": int(raw.nd),
        "param_count": _count_params(raw),
        "blocks": blocks_info,
        # Derived inputs, `{u_column: {kind, state, ...}}`; None when no
        # column is rewritten.
        "pre_stack": pre_stack.describe() if pre_stack is not None else None,
    }


def _describe_custom_model(model: nn.Module, widths: tuple[int, int, int],
                         cfg: ModelingConfig) -> dict:
    """`describe_dynamics` for a model from `custom_models()`: what it says
    about itself (`describe()`, its bounded parameters), and its size."""
    ny, nu, nd = widths
    return {
        "factory": cfg.model,
        "separable": False,
        "residual": False,
        "history": dict(cfg.history),
        "history_steps": cfg.history_steps,
        "ny": ny,
        "nu": nu,
        "nd": nd,
        "param_count": _count_params(model),
        "blocks": {"fy": _describe_block(model, cfg), "fu": None, "fd": None},
        "pre_stack": None,
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
    `rollout_state()` / `set_rollout_state(state)` (NetworkBlock, GRUBlock): the
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
    with `cfg.per_block`, or a resolved family is `rc_network`, which is
    per-block unconditionally: it describes *a* zone, so there is no
    monolithic one to fall back to.
    """
    return (any(b.model is not None for b in blocks)
            or cfg.per_block or "rc_network" in families)


def _check_increment_heads(blocks: list[Block], families: list[str],
                           head_stack: HeadStack) -> None:
    """rc_network predicts increments; only a residual-compatible head
    restores the value. Scoped to the channels such a block produces, so a
    mixed-family run's other channels are free to use any head — see
    _build_per_block_model for the analogous residual/steps_per_hour checks,
    which are model-wide because SSMGrad's residual flag is not per-channel.
    """
    rc_channels = {
        name for block, family in zip(blocks, families) if family == "rc_network"
        for name in block.y_names
    }
    if not rc_channels:
        return
    blind = [name for name, head in zip(head_stack.channel_names, head_stack.heads)
             if name in rc_channels and not head.residual_compatible]
    if blind:
        raise ValueError(
            f"rc_network predicts increments, so every Y channel it produces needs "
            f"a residual-compatible head; {blind} are not."
        )


def _build_pre_stack(
    blocks: list[Block], nm_mapping: dict[str, list[str]],
    norm_stats: Optional[dict], history_steps: int,
) -> Optional[PreStack]:
    """One `PreStack` over every block's `demand` declarations, or None.

    The stack is block-agnostic — each entry is resolved to a global
    `(U column, Y channel)` pair — which is what lets one wiring point in
    `SSMGrad.forward` serve every family.
    """
    demands = {
        column: {k: entry[k] for k in ("kind", "state", "sign")}
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
        demands, nm_mapping["Y"], nm_mapping["U"], norm_stats, history_steps)


def _warn_ignored_declarations(blocks: list[Block], families: list[str]) -> None:
    """`ambient`, `heat` and `solar` only mean something to `rc_network`;
    every other family already sees those columns as ordinary inputs. Say
    so rather than let the declaration sit inert."""
    for block, family in zip(blocks, families):
        if family == "rc_network":
            continue
        ignored = [c for c in (block.ambient, block.heat, block.solar) if c]
        if ignored:
            warnings.warn(
                f"block {block.name!r} builds under {family!r}, which ignores "
                f"the `ambient`/`heat`/`solar` declaration of {ignored}: only "
                "rc_network gives those columns a place in its physics.",
                stacklevel=3,
            )


@beartype
def get_dynamics(
    cfg: ModelingConfig,
    nm_mapping: dict[str, list[str]],
    norm_stats: dict[str, dict],
    head_stack: Optional[HeadStack] = None,
    block_spec: Optional[dict[str, dict]] = None,
) -> SystemPreview:
    """
    Wrap a dynamics model in a Neuromancer SystemPreview for autoregressive rollout.

    The Node reads ['yn', 'U', 'D'] and writes ['yn']. Because 'yn' is both
    input and output, the System feeds predictions back as input at the next
    timestep — this is the autoregressive loop.

    SystemPreview handles temporal windowing via input_map: at each rollout
    step i, the Node receives a window of `cfg.history_steps` values for each
    input, flattened to [batch, history_steps*features]; the learned maps
    narrow it to each column's lag in `cfg.history`.

    `block_spec` is the optional `block_mapping` hook's result, splitting Y,
    U and D across independent blocks — see `blocks`. None means one block
    holding every channel, which is the right answer for a single-block
    building. `norm_stats` is the run's: the family computes in the units
    it normalises to, and `head_stack`, declared in physical units, is
    rescaled into them.

    A `cfg.model` that is not a built-in family is looked up in the hooks
    file's `custom_models()` (see `contract`).
    """
    if cfg.model not in VALID_MODEL_NAMES:
        model = _custom_model(cfg, nm_mapping, norm_stats, head_stack, block_spec)
        return _rollout_system(model, model, cfg, nm_mapping)

    blocks = build_blocks(nm_mapping, block_spec)
    families = [_resolve_family(cfg, b) for b in blocks]

    _warn_ignored_declarations(blocks, families)
    unread = [b.name for b, family in zip(blocks, families)
              if b.params and family != "rc_network"]
    if unread:
        raise ValueError(
            f"block(s) {unread} declare `params`, but only an rc_network block "
            "reads any; the declaration would sit inert."
        )
    lags = column_lags(cfg, nm_mapping)
    _check_demand_lags(blocks, cfg.history)
    if _uses_per_block(cfg, blocks, families):
        per_block_lags = [block_lags(family, block, cfg.history)
                          for block, family in zip(blocks, families)]
        _check_lags_consumed(blocks, per_block_lags, cfg.history)
        model = _build_per_block_model(cfg, nm_mapping, blocks, families,
                                       per_block_lags, norm_stats)
    else:
        # One network reads every column, at its declared lag; only a GRU
        # constrains them.
        if cfg.model == "gru":
            _check_one_lag("gru", dict(cfg.history),
                           "every column: its window is its warm-up")
        model = model_catalog[cfg.model](cfg, nm_mapping, lags)
    model.pre_stack = _build_pre_stack(blocks, nm_mapping, norm_stats,
                                       cfg.history_steps)

    # Heads are attached post-construction so each factory stays focused
    # on its own block topology. Every head has width 1, so the factories'
    # output width ny is also the stack's raw width.
    if head_stack is not None:
        if head_stack.raw_width != model.ny:
            raise ValueError(
                f"head_stack.raw_width={head_stack.raw_width} does not match "
                f"model.ny={model.ny}; factories (currently) assume width-1 heads."
            )
        _check_increment_heads(blocks, families, head_stack)
        # A new stack, in the family's units; post-training calibration
        # writes its decision thresholds into this one, never the caller's.
        model.head_stack = head_stack.normalized(norm_stats)

    return _rollout_system(Normalized(model, nm_mapping, norm_stats), model, cfg, nm_mapping)


def _rollout_system(contract: nn.Module, stateful: nn.Module, cfg: ModelingConfig,
                    nm_mapping: dict[str, list[str]]) -> SystemPreview:
    """The contract model as the autoregressive rollout; `stateful` is where
    its rollout state lives, reset at every sequence boundary."""
    # Temporal windowing: one rectangular window per key, history_steps deep;
    # the model narrows it to each column's lag.
    input_map = {
        key: {
            "past": cfg.history_steps - 1,
            "future": 0,
            "pad_mode": "nearest",
        }
        for key in ["yn", "U", "D"]
    }

    node = Node(
        Windowed(contract, _widths(nm_mapping), cfg.history_steps),
        ['yn', 'U', 'D'],   # input: reads predicted state + exogenous
        ['yn'],              # output: next predicted state (feedback loop)
        name=cfg.model.upper() + " Node",
        input_map=input_map,
    )

    return WindowedSystem(
        [node],
        name=cfg.model.upper() + " System",
        nstep_key="Y",               # rollout length from ground truth
        start_iter=cfg.history_steps - 1,  # skip initial history steps
        # Stateful blocks only: clear the hidden state so it cannot leak
        # between batches. None for stateless blocks.
        init_func=make_reset_init(stateful),
    )


def _custom_model(cfg: ModelingConfig, nm_mapping: dict[str, list[str]],
                norm_stats: dict[str, dict], head_stack: Optional[HeadStack],
                block_spec: Optional[dict[str, dict]]) -> nn.Module:
    """`cfg.model` built by the hooks file's `custom_models()`: one module
    over every column, behind the contract as it is (see `contract`).

    What a built-in family adds around its learned maps is the family's own,
    so the library's per-column machinery is refused here rather than
    silently ignored: blocks, heads, and the gradient penalty (`build_loss`).
    """
    custom = getattr(load_modeling_hooks(cfg), "custom_models", lambda: {})()
    if cfg.model not in custom:
        raise ValueError(
            f"Unknown model {cfg.model!r}. Built-in families: "
            f"{list(VALID_MODEL_NAMES)}; the hooks file's custom_models(): "
            f"{sorted(custom)}.")
    if block_spec:
        raise ValueError(
            f"model {cfg.model!r} is one module over every column, so "
            "block_mapping has nothing to split: return None from it. Blocks "
            "are the built-in families' structure.")
    if head_stack is not None and any(type(h) is not RegressionHead for h in head_stack.heads):
        raise ValueError(
            f"model {cfg.model!r} emits its own outputs, so data_head_config "
            "must give every channel a RegressionHead. Bound or level a "
            "channel inside the model; `heads` has the activations to use.")
    column_lags(cfg, nm_mapping)   # the history table names exactly the columns
    model = custom[cfg.model](cfg, nm_mapping, norm_stats)
    if not isinstance(model, nn.Module):
        raise TypeError(f"custom_models()[{cfg.model!r}] returned "
                        f"{type(model).__name__}, not a torch.nn.Module.")
    return model
