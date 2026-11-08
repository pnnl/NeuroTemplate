"""
Building a dynamics model from a config and the block hook, and wrapping it for
the rollout.

Every built-in model is one shape: a `stepper.Stepper` around `Blocks`, one block
per group of outputs, each block's core built by its family (`FAMILIES`). No
`block_mapping` hook means one block holding every column, so a single network
over all of Y and a split are the same code, and any family can sit in any block.

`get_dynamics()` wraps the model in a Neuromancer Node + SystemPreview:

  - The Node uses input_keys=['yn', 'U', 'D'] and output_keys=['yn'].
    Because 'yn' appears in BOTH input and output, the System creates
    an autoregressive feedback loop: at each timestep, the model reads
    its own previous prediction and produces the next one.

  - SystemPreview with start_iter=history_steps-1 skips the first timesteps
    (the model needs history_steps of history). The input_map on the Node
    handles temporal windowing: at each step i, the model receives
    data[:, i-past:i+1, :] flattened; `contract.Windowed` undoes the flattening.

  - The data loader provides 'yn' initialized to the first history_steps
    of ground truth (see data_utils.get_sw_dict). The System then
    appends model predictions to 'yn' via cat() at each rollout step.

  - 'Y' (ground truth) is only used by the loss for comparison and
    by the System to determine rollout length (nstep_key='Y').

`cfg.past` sets each column's lag, the steps before the current one a learned
map reads: any lag for `mlp`, `nssm` and `lssm`, one lag for every column a `gru`
reads (its window is its warm-up). A thermal block's window is not in the table:
an `rc_network` reads the whole window, `cfg.context` deep, for its filter,
and a `thermostat` the current row. A stateful block additionally keeps state so
its memory can run past its window during a rollout, but it still takes the same
window. The only thing statefulness costs the caller is a reset at each sequence
boundary, which make_reset_init() wires into the System.

Training windows start `cfg.train_stride` apart.
"""

import warnings

import torch
import torch.nn as nn
from typing import Callable, Optional
from neurotemplate.modeling.config import ModelingConfig

from neurotemplate._typing import beartype

from neurotemplate.nm.system import Node, SystemPreview

from neurotemplate.hooks import load_modeling_hooks
from neurotemplate.modeling.contract import Windowed, contract_model, stateful_blocks
from neurotemplate.modeling.families import Mlp, Separable
from neurotemplate.modeling.heads import HeadStack, RegressionHead
from neurotemplate.modeling.hinge import Hinges
from neurotemplate.modeling.bounded import bounded_parameters
from neurotemplate.modeling.gru import GRUBlock
from neurotemplate.modeling.rollout import WindowedSystem
from neurotemplate.modeling.blocks import Block, build_blocks
from neurotemplate.modeling.rc_network import ThermostatBlock, ZoneBlock
from neurotemplate.modeling.stepper import Stepper


# ── Per-column lags ──────────────────────────────────────────────

#: One lag per column of each key, `(Y, U, D)`, in the order the window
#: carries the columns.
Lags = tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]

_KEYS = ("Y", "U", "D")

#: The physical families: each describes one part of a zone, reads its declaration
#: from the block's `params`, and returns the value itself.
THERMAL_FAMILIES = ("rc_network", "thermostat")

@beartype
def column_lags(cfg: ModelingConfig, nm_mapping: dict[str, list[str]]) -> Lags:
    """`cfg.past` as one lag per column, in steps, in nm_mapping order. A table names
    only the model's columns: an extra one is a declaration nothing reads."""
    columns = [c for key in _KEYS for c in nm_mapping[key]]
    extra = sorted(set(cfg.past) - set(columns)) if isinstance(cfg.past, dict) else []
    if extra:
        raise ValueError(f"past names {extra}, which are not Y, U or D columns of the model.")
    lags = cfg.lags_for(columns)
    return tuple(tuple(lags[c] for c in nm_mapping[key]) for key in _KEYS)


def _check_one_lag(what: str, lags: dict[str, int], reason: str) -> None:
    if len(set(lags.values())) > 1:
        raise ValueError(f"{what} needs one lag for {reason}; `past` gives {lags}.")


@beartype
def block_lags(family: str, block: Block, history: dict[str, int], history_steps: int) -> Lags:
    """The lags `block` consumes, per column of its window: `y_in_names`,
    `u_names`, `d_names`.

    `mlp` reads every column at its declared lag. `gru` folds its window
    into its hidden state row by row, so every column it reads shares one
    lag. `rc_network` filters the whole window and `thermostat` reads the
    current row, whatever the table says.
    """
    names = (block.y_in_names, block.u_names, block.d_names)
    if family == "rc_network":
        return tuple(tuple(history_steps - 1 for _ in key) for key in names)
    if family == "thermostat":
        return tuple(tuple(0 for _ in key) for key in names)
    if family == "gru":
        _check_one_lag(
            f"gru block {block.name!r}",
            {c: history[c] for c in block.y_in_names + block.u_names + block.d_names},
            "every column it reads: its window is its warm-up")
    return tuple(tuple(history[c] for c in key) for key in names)


def _check_lags_consumed(blocks: list[Block], families: list[str], lags: list[Lags],
                         history: dict[str, int]) -> None:
    """Every column's declared lag is what some learned block reading it
    consumes: a deeper declaration would sit inert. A thermal block's reads are
    not in the table, so a column only they read is listed at 0. A D column no
    block reads (per-block D subsets that all leave it out) is listed at 0 too:
    the control side may still use it, so it stays a model column.
    """
    consumed: dict[str, dict[str, int]] = {c: {} for c in history}
    for block, family, (y, u, d) in zip(blocks, families, lags):
        if family in THERMAL_FAMILIES:
            continue
        names = block.y_in_names + block.u_names + block.d_names
        for column, lag in zip(names, y + u + d):
            consumed[column][block.name] = lag
    for column, readers in consumed.items():
        if not readers and history[column] != 0:
            raise ValueError(
                f"past[{column!r}] is {history[column]} steps, but no learned block reads "
                "it. A thermal block's window is `context`, not this table. If "
                "nothing on the control side needs it either (the policy, a loss weight, "
                "power_metric, the plots), take it out of `columns` in data.yaml; otherwise "
                "list it at 0."
            )
        deepest = max(readers.values(), default=0)
        if deepest != history[column]:
            raise ValueError(
                f"past[{column!r}] is {history[column]} steps, but no block reads "
                f"it that far back; the lag each reader consumes, in steps: {readers}. Set "
                f"it to {deepest} steps."
            )


def _check_hinge_lags(blocks: list[Block], history: dict[str, int]) -> None:
    """A hinged setpoint's lag is at most its zone's.

    The hinge is computed on the full window, so a setpoint read further back than its
    zone would carry zone history older than the zone's own declared lag into the block.
    """
    for block in blocks:
        for column, entry in block.inputs.items():
            target = entry["target"]
            if history[column] > history[target]:
                raise ValueError(
                    f"past[{column!r}] is {history[column]} steps, deeper than its hinge's "
                    f"target past[{target!r}], {history[target]} steps. The hinge reads the "
                    f"target at every step it covers; give {column!r} a lag of at most "
                    f"{history[target]} steps."
                )


# ── The families ─────────────────────────────────────────────────

def _mlp_core(cfg, block, lags, norm_stats):
    return Mlp(lags, cfg.history_steps, block.ny, 2**cfg.log2_hidden_size, cfg.num_layers)


def _nssm_core(cfg, block, lags, norm_stats):
    return Separable(lags, cfg.history_steps, block.ny, 2**cfg.log2_hidden_size, cfg.num_layers)


def _lssm_core(cfg, block, lags, norm_stats):
    return Separable(lags, cfg.history_steps, block.ny)


def _gru_core(cfg, block, lags, norm_stats):
    # One lag for every column (block_lags); a block always reads its own Y.
    return GRUBlock(block.ny_in + block.nu + block.nd, lags[0][0] + 1, block.ny,
                    2**cfg.log2_hidden_size, cfg.num_layers)


def _rc_network_core(cfg, block, lags, norm_stats):
    if cfg.steps_per_hour is None:
        raise ValueError(
            "rc_network needs cfg.steps_per_hour to discretise its network. The CLI sets "
            "it from the data's sampling frequency; set it by hand (steps_per_hour = "
            "3600 / timestep_seconds) if you are building a model outside the CLI.")
    return ZoneBlock(block, cfg.history_steps, 1.0 / cfg.steps_per_hour, norm_stats)


def _thermostat_core(cfg, block, lags, norm_stats):
    return ThermostatBlock(block, norm_stats)


#: Every built-in family: `(cfg, block, lags, norm_stats) -> core`.
FAMILIES: dict[str, Callable] = {
    "mlp": _mlp_core,
    "nssm": _nssm_core,
    "lssm": _lssm_core,
    "gru": _gru_core,
    "rc_network": _rc_network_core,
    "thermostat": _thermostat_core,
}

#: Every valid built-in `cfg.model` name.
VALID_MODEL_NAMES = tuple(FAMILIES)


def fill_context(cfg: ModelingConfig, nm_mapping: dict[str, list[str]],
                 block_spec: Optional[dict[str, dict]]) -> None:
    """`context`, when unset: a day when a block is an rc_network, whose filter places its
    hidden nodes from it, and nothing otherwise. A day is plenty for buildings measured
    so far, and it costs records, so a model that keeps no such state is not charged it."""
    if cfg.context is None:
        filtered = cfg.model in VALID_MODEL_NAMES and any(
            resolve_family(cfg, block)[0] == "rc_network"
            for block in build_blocks(nm_mapping, block_spec))
        cfg.context = "24h" if filtered else 0


def resolve_family(cfg: ModelingConfig, block: Block) -> tuple[str, str]:
    """`(family, where it came from)` for one block: the config's `block_models`,
    then the block's own `model`, then the config's `model`."""
    if block.name in cfg.block_models:
        return cfg.block_models[block.name], "block_models"
    if block.model is not None:
        return block.model, "block_mapping"
    return cfg.model, "model"


@beartype
def family_lines(cfg: ModelingConfig, nm_mapping: dict[str, list[str]],
                 block_spec: Optional[dict[str, dict]]) -> list[str]:
    """One line per block: the family it takes, and which declaration said so,
    since a family can be set in three places."""
    lines = []
    for block in build_blocks(nm_mapping, block_spec):
        family, source = resolve_family(cfg, block)
        lines.append(f"  block {block.name!r}: {family} (from {source})")
    return lines


class Blocks(nn.Module):
    """
    One core per block, over the windows they all share: the core every built-in
    model is made of.

    Each block sees its own Y and U columns, its own D subset (all of D by
    default), anything it `reads` from another block as of the row before, and
    anything it is `driven_by` as of this step (see `blocks` for what a block
    owns), and predicts only its own Y channels. Blocks run in the order
    `blocks.run_order` gives them. A `driven_by` column's rows mean what a U's
    do: row j held over the step from j to j + 1, its last row what the producer
    just wrote, head and all.
    """

    def __init__(self, blocks: list[Block], families: list[str], lags: list[Lags],
                 cores: list[nn.Module]):
        super().__init__()
        self.specs = list(blocks)
        self.families = list(families)
        self.lags = list(lags)
        self.cores = nn.ModuleList(cores)
        for i, block in enumerate(blocks):
            for key, idx in zip(("y", "u", "d"), (block.y_in_idx, block.u_idx, block.d_idx)):
                self.register_buffer(f"_{key}{i}", torch.tensor(idx, dtype=torch.long),
                                     persistent=False)
        position = {c: j for block in blocks for j, c in zip(block.y_idx, block.y_names)}
        self._driven = [tuple((block.y_in_names.index(c), position[c])
                              for c in block.driven_by) for block in blocks]
        run = [j for block in blocks for j in block.y_idx]
        self.register_buffer("_to_y_order", torch.tensor(
            [run.index(j) for j in range(len(run))], dtype=torch.long), persistent=False)
        self._read = {channel for driven in self._driven for _, channel in driven}
        self._outputs = ([], None)

    def reads_outputs_through(self, residual_mask: torch.Tensor,
                              head_stack: Optional[HeadStack]) -> None:
        """What turns a block's `z` into the value a block driven by it reads,
        both owned by the `Stepper` this core is inside."""
        self._outputs = (residual_mask.tolist(), head_stack)

    def _value(self, z: torch.Tensor, newest: torch.Tensor, channel: int) -> torch.Tensor:
        residual, head_stack = self._outputs
        value = z + newest if residual[channel] else z
        if head_stack is None:
            return value
        return head_stack.activate_channel(value, channel)

    def forward(self, y: torch.Tensor, u: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        written: dict[int, torch.Tensor] = {}
        outs = []
        for i, (core, spec) in enumerate(zip(self.cores, self.specs)):
            y_in = y.index_select(2, getattr(self, f"_y{i}"))
            if self._driven[i]:
                columns = list(y_in.unbind(-1))
                for at, channel in self._driven[i]:
                    columns[at] = torch.cat([y[:, 1:, channel], written[channel][:, None]], dim=1)
                y_in = torch.stack(columns, dim=-1)
            out = core(y_in, u.index_select(2, getattr(self, f"_u{i}")),
                       d.index_select(2, getattr(self, f"_d{i}")))
            for at, channel in enumerate(spec.y_idx):
                if channel in self._read:
                    written[channel] = self._value(out[:, at], y[:, -1, channel], channel)
            outs.append(out)
        return torch.cat(outs, dim=-1).index_select(1, self._to_y_order)


# ── Model description (for reportcard) ───────────────────────────

def _count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def _describe_part(part: nn.Module) -> dict:
    """What a core or a model of your own says about itself (`describe()`), its
    size, and every bounded parameter under it: its box and where the fit landed."""
    info = {"class": type(part).__name__, "param_count": _count_params(part)}
    if hasattr(part, "describe"):
        info.update(part.describe())
    bounded = {b.name: b.describe() for b in bounded_parameters(part)}
    if bounded:
        info["bounded"] = bounded
    return info


@beartype
def describe_dynamics(dyn_sys: SystemPreview, cfg: ModelingConfig) -> dict:
    """
    Describe the trained dynamics system for the reportcard.

    Returns a dict with:
      - factory: the model family, or `mixed (a, b)` when the blocks use different
        families (each block names its own), or a custom model's name
      - residual: whether the run's `cfg.predicts` is an increment
      - past: the run's `cfg.past`, and history_steps, the window depth
      - ny / nu / nd: shape parameters
      - param_count: total trainable parameters
      - blocks: per-block descriptors, `{name: info}`; a custom model is one
      - hinges: the setpoint hinges, or None

    Used by model_card.render_reportcard.
    """
    model = contract_model(dyn_sys)
    ny, nu, nd = dyn_sys.nodes[0].callable.widths
    core = getattr(model, "core", None)
    if isinstance(core, Blocks):
        blocks = {}
        for spec, family, lags, part in zip(core.specs, core.families, core.lags, core.cores):
            info = _describe_part(part)
            info["model"] = family
            # What it reads, which for a thermal block is not what `past` declares.
            info["lags"] = dict(zip(spec.y_in_names + spec.u_names + spec.d_names,
                                    lags[0] + lags[1] + lags[2]))
            blocks[spec.name] = info
        families = sorted({info["model"] for info in blocks.values()})
        factory = families[0] if len(families) == 1 else f"mixed ({', '.join(families)})"
    else:
        factory = cfg.model
        blocks = {cfg.model: {**_describe_part(model), "model": cfg.model}}
    hinges = getattr(model, "hinges", None)
    return {
        "factory": factory,
        "residual": cfg.predicts == "increment",
        "past": cfg.past,
        "history_steps": int(cfg.history_steps),
        "ny": int(ny),
        "nu": int(nu),
        "nd": int(nd),
        "param_count": _count_params(model),
        "blocks": blocks,
        "hinges": hinges.describe() if hinges is not None else None,
    }


# ── System construction ──────────────────────────────────────────

def make_reset_init(model: nn.Module):
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
    """
    blocks = stateful_blocks(model)
    if not blocks:
        return None

    def reset_init(data):
        for block in blocks:
            block.set_rollout_state(None)
        return data

    return reset_init


def _check_value_heads(blocks: list[Block], families: list[str], head_stack: HeadStack) -> None:
    """A thermal block's output is the value, in its column's units; only a head
    that reads it as one can sit on it. Scoped to the channels such a block
    produces, so its neighbours are free to use any head."""
    thermal = {name for block, family in zip(blocks, families) if family in THERMAL_FAMILIES
               for name in block.y_names}
    blind = [name for name, head in zip(head_stack.channel_names, head_stack.heads)
             if name in thermal and not head.residual_compatible]
    if blind:
        raise ValueError(f"a thermal block's output is the value in its column's units, so "
                         f"every Y channel it produces needs a head that reads it as one; "
                         f"{blind} are not.")


def _build_hinges(blocks: list[Block], nm_mapping: dict[str, list[str]],
                  norm_stats: dict) -> Optional[Hinges]:
    """One `Hinges` over every block's declarations, or None."""
    declared = {column: entry for block in blocks for column, entry in block.inputs.items()}
    return Hinges(declared, nm_mapping, norm_stats) if declared else None


def _warn_ignored_declarations(blocks: list[Block], families: list[str]) -> None:
    """`ambient`, `scale` and `solar` only mean something to the thermal
    families; every other family already sees those columns as ordinary inputs.
    Say so rather than let the declaration sit inert."""
    for block, family in zip(blocks, families):
        if family in THERMAL_FAMILIES:
            continue
        ignored = [c for c in (block.ambient, block.solar) if c] + list(block.scale or ())
        if ignored:
            warnings.warn(
                f"block {block.name!r} builds under {family!r}, which ignores "
                f"the `ambient`/`scale`/`solar` declaration of {ignored}: only "
                "the thermal families give those columns a place in their physics.",
                stacklevel=3,
            )


def _refuse_declarations(cfg: ModelingConfig, blocks: list[Block], families: list[str]) -> None:
    """A family, block or `params` key that names nothing, and an increment a thermal
    block cannot predict: loud, since each would otherwise change what is built."""
    problems = []
    names = {block.name for block in blocks}
    unknown = sorted(set(cfg.block_models) - names)
    if unknown:
        problems.append(f"block_models names {unknown}; the blocks are {sorted(names)}.")
    for block, family in zip(blocks, families):
        if family not in FAMILIES:
            problems.append(f"block {block.name!r} asks for model {family!r}; built-in "
                            f"families: {sorted(FAMILIES)}.")
        if block.predicts == "increment" and family in THERMAL_FAMILIES:
            problems.append(f"block {block.name!r}: {family} outputs the value itself, so "
                            "`predicts: increment` has no meaning there.")
        stray = sorted(set(block.params) - set(THERMAL_FAMILIES))
        if stray:
            problems.append(f"block {block.name!r} declares `params` for {stray}; `params` "
                            f"maps a family that reads a declaration, {list(THERMAL_FAMILIES)}, "
                            "to that declaration.")
    if problems:
        raise ValueError("block_mapping:\n  - " + "\n  - ".join(problems))


@beartype
def get_dynamics(
    cfg: ModelingConfig,
    nm_mapping: dict[str, list[str]],
    norm_stats: dict[str, dict],
    head_stack: Optional[HeadStack] = None,
    block_spec: Optional[dict[str, dict]] = None,
) -> SystemPreview:
    """
    The model `cfg` names, wrapped in a Neuromancer SystemPreview for
    autoregressive rollout.

    The Node reads ['yn', 'U', 'D'] and writes ['yn']. Because 'yn' is both
    input and output, the System feeds predictions back as input at the next
    timestep — this is the autoregressive loop. At each rollout step i, the Node
    receives a window of `cfg.history_steps` rows of each input.

    `block_spec` is the optional `block_mapping` hook's result, splitting Y,
    U and D across blocks — see `blocks`. None means one block holding every
    channel. `norm_stats` is the run's: a family computes in the units it
    normalises to, and `head_stack`, declared in physical units, is rescaled
    into them.

    A `cfg.model` that is not a built-in family is looked up in the hooks
    file's `custom_models()` (see `contract`).
    """
    if cfg.model not in VALID_MODEL_NAMES:
        return _rollout_system(_custom_model(cfg, nm_mapping, norm_stats, head_stack, block_spec),
                               cfg, nm_mapping)

    blocks = build_blocks(nm_mapping, block_spec)
    families = [resolve_family(cfg, block)[0] for block in blocks]
    _refuse_declarations(cfg, blocks, families)
    _warn_ignored_declarations(blocks, families)
    column_lags(cfg, nm_mapping)   # a `past` table names only the model's columns
    declared = cfg.lags_for([c for key in _KEYS for c in nm_mapping[key]])
    _check_hinge_lags(blocks, declared)
    lags = [block_lags(family, block, declared, cfg.history_steps)
            for block, family in zip(blocks, families)]
    # A table says each column on purpose; one time for all is not a claim about each.
    if isinstance(cfg.past, dict):
        _check_lags_consumed(blocks, families, lags, declared)
    if head_stack is not None:
        _check_value_heads(blocks, families, head_stack)

    core = Blocks(blocks, families, lags,
                  [FAMILIES[family](cfg, block, block_lag, norm_stats)
                   for block, family, block_lag in zip(blocks, families, lags)])
    flags = [False] * len(nm_mapping["Y"])
    for block, family in zip(blocks, families):
        for channel in block.y_idx:
            flags[channel] = (family not in THERMAL_FAMILIES
                              and (block.predicts or cfg.predicts) == "increment")
    model = Stepper(core, nm_mapping, norm_stats, cfg.history_steps, residual=flags,
                    head_stack=head_stack,
                    hinges=_build_hinges(blocks, nm_mapping, norm_stats))
    core.reads_outputs_through(model.residual_mask, model.head_stack)
    return _rollout_system(model, cfg, nm_mapping)


def _rollout_system(model: nn.Module, cfg: ModelingConfig,
                    nm_mapping: dict[str, list[str]]) -> SystemPreview:
    """The contract model as the autoregressive rollout; its rollout state is
    reset at every sequence boundary."""
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
        Windowed(model, tuple(len(nm_mapping[key]) for key in _KEYS), cfg.history_steps),
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
        init_func=make_reset_init(model),
    )


def _custom_model(cfg: ModelingConfig, nm_mapping: dict[str, list[str]],
                  norm_stats: dict[str, dict], head_stack: Optional[HeadStack],
                  block_spec: Optional[dict[str, dict]]) -> nn.Module:
    """`cfg.model` built by the hooks file's `custom_models()`: one module over
    every column, behind the contract as it is (see `contract`).

    Blocks are built-in families' structure, so a spec is refused. A model that
    bounds an output says so by owning a `head_stack`; a head declared
    in `output_heads` that it does not own would be silently ignored, so that
    is refused too.
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
            "are the built-in families' structure; wire parts in its forward.")
    column_lags(cfg, nm_mapping)   # a `past` table names only the model's columns
    model = custom[cfg.model](cfg, nm_mapping, norm_stats)
    if not isinstance(model, nn.Module):
        raise TypeError(f"custom_models()[{cfg.model!r}] returned "
                        f"{type(model).__name__}, not a torch.nn.Module.")
    declared = head_stack is not None and any(type(h) is not RegressionHead
                                              for h in head_stack.heads)
    if declared and getattr(model, "head_stack", None) is None:
        raise ValueError(
            f"output_heads declares heads, but model {cfg.model!r} owns none. Give it a "
            "`head_stack` (a `stepper.Stepper` takes one), or make every head a "
            "RegressionHead.")
    return model
