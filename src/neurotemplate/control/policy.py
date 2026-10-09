"""
DPC control policy.

A bounded network wrapped in a Neuromancer Node:

    [yn-window, D-window(, R-window)]  ->  U

R is the objective's reference columns (control.losses), present only when it
names one.

Output is squashed into the actuator limits (normalized U space) by
blocks.MLP_bounds. Windowing uses the same input_map convention as
modeling/dynamics.get_dynamics so the policy composes with the frozen dynamics Node.

`policy_catalog` selects the network family the same way `model_catalog`
selects a dynamics family on the sys-id side:

  - "mlp"    — bounded MLP, `policy_layers` hidden layers of width
               2**log2_policy_hidden
  - "linear" — the same bounded map with no hidden layers, i.e. a single
               saturated affine feedback law u = sat(K @ x + b) over the
               flattened feature window. This is the structure of an explicit
               MPC law inside its unconstrained region, and its gain matrix is
               small enough to read.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from neuromancer.system import Node
from neuromancer.modules import blocks

from neurotemplate.common.data_utils import NormStats
from neurotemplate.control.losses import resolve_objective
from neurotemplate.control.scenarios import horizons, scored_steps
from neurotemplate.hooks import load_control_hooks
from neurotemplate.modeling import windows

from neurotemplate._typing import beartype


# Nonlinearities selectable from YAML. Kept small on purpose — these are the
# activations that actually get used for control policies.
POLICY_NONLIN = {
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "silu": nn.SiLU,
}

# Positions of D and R among the policy net's positional inputs, fixed by the
# input_keys build_policy constructs: ["yn", "D"], then "R" when there is one.
_D_ARG_INDEX = 1
_R_ARG_INDEX = 2


class STEQuantize(nn.Module):
    """Snap to a setpoint grid in the forward pass; identity in the backward.

    Lets the policy learn against a continuous surrogate while emitting only
    grid-feasible setpoints (e.g. half-degree thermostat steps). The
    `(round(.) - u).detach()` trick zeros the gradient contribution of the
    snap, so ∂out/∂u = 1.

    Snap targets the grid `{k*step - offset : integer k}` in the same space
    as the input. For the policy this is normalized U: caller passes
    `step = step_phys / stretch` and `offset = shift / stretch` so the
    *physical* grid lands at multiples of `step_phys`.

    `step` and `offset` may be scalars or 1-D tensors of shape [nu]
    (per-channel grid, broadcast over batch/time).
    """

    def __init__(self, step, offset=0.0):
        super().__init__()
        if isinstance(step, torch.Tensor):
            if (step <= 0).any():
                raise ValueError(f"STEQuantize step entries must be > 0, got {step}")
            self.register_buffer("step", step.float())
        else:
            if step <= 0:
                raise ValueError(f"STEQuantize step must be > 0, got {step}")
            self.step = float(step)

        if isinstance(offset, torch.Tensor):
            self.register_buffer("offset", offset.float())
        else:
            self.offset = float(offset)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        snapped = ((u + self.offset) / self.step).round() * self.step - self.offset
        return u + (snapped - u).detach()


class _QuantizedPolicy(nn.Module):
    """Compose the bounded MLP with STEQuantize while keeping the variadic
    forward signature the Neuromancer Node expects."""

    def __init__(self, base: nn.Module, quantize: STEQuantize):
        super().__init__()
        self.base = base
        self.quantize = quantize

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        return self.quantize(self.base(*args))


class _PreviewGather(nn.Module):
    """Narrow previewed windows to the (column, timestep) pairs declared.

    SystemPreview's input_map gives one window per key, so a key's window is
    sized to the deepest preview any of its columns declared. A column that
    asked for less would otherwise see future it never requested. This applies
    the narrowing gather — one per positional input, keyed by its index — in
    front of whatever net build_policy constructed.

    A fixed permutation computed once at construction, applied with
    `index_select` against registered buffers — same discipline as
    `BlockStack` (modeling/dynamics.py): no data-dependent control flow.
    """

    def __init__(self, base: nn.Module, gathers: dict[int, torch.Tensor]):
        super().__init__()
        self.base = base
        self.indices = sorted(gathers)
        for i in self.indices:
            self.register_buffer(f"_gather_{i}", gathers[i], persistent=False)

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        args = list(args)
        for i in self.indices:
            args[i] = args[i].index_select(-1, getattr(self, f"_gather_{i}"))
        return self.base(*args)


def _current(x: torch.Tensor, block: int, width: int) -> torch.Tensor:
    """The `block`-th step out of a flattened `(batch, window*width)` window."""
    return x[:, block * width:(block + 1) * width]


class _ReferenceRule(nn.Module):
    """Adapt `control_user_defined.reference_control` to the policy's inputs.

    The policy is handed *flattened* windows — `(batch, window, dim)` reshaped
    to `(batch, window*dim)`, oldest-first, see `SystemPreview.get_mapped_data`
    — and that flattening convention is library business which must not leak
    into a use case's rule. This slices the current step back out and hands the
    hook plain `[b, n]` tensors, and each reference column by name.

    Deliberately memoryless: every rule-based controller in the record is a
    function of the reference and the current state, and it is what the
    deploy-side callback can supply with no new plumbing. A rule that needs its
    own integrator is out of scope — say so rather than adding a
    state-carrying path for it.
    """

    def __init__(self, rule, nm_mapping: dict[str, list[str]],
                 norm_stats: NormStats, h_ctrl: int, r_columns: list[str]):
        super().__init__()
        self.rule = rule
        self.nm_mapping = nm_mapping
        self.norm_stats = norm_stats
        self.r_columns = r_columns
        # Which block of each flattened window holds the current step. yn and
        # D look `h_ctrl - 1` steps back, so it is the last of their history;
        # R has no history, so it is the first.
        self.ny, self.nd = len(nm_mapping["Y"]), len(nm_mapping["D"])
        self.now = h_ctrl - 1

    def forward(self, yn: torch.Tensor, d: torch.Tensor,
                r: Optional[torch.Tensor] = None) -> torch.Tensor:
        r_now = {} if r is None else {
            name: r[:, j] for j, name in enumerate(self.r_columns)}
        return self.rule(
            _current(yn, self.now, self.ny),
            r_now,
            _current(d, self.now, self.nd),
            self.nm_mapping,
            self.norm_stats,
        )


class _ResidualPolicy(nn.Module):
    """Command a bounded deviation from a reference rule, not a setpoint.

        δ = clamp(MLP_bounds(window), -below, +above)   net bounded to ±max
        u = clamp(quantize(u_ref + δ), umin, umax)

    The point is not that a network finds it easier to learn zero than
    identity — the `linear` family already expresses `u = sat(Kx + b)`, so
    representing the rule was never the binding constraint. The point is that
    a bounded residual is **a trust region on surrogate error**: the optimizer's
    gradient and the model-selection criterion both come from a surrogate whose
    own held-out score can be anti-correlated with plant fidelity, and a ±Δ box
    caps what that error can cost you without needing to fix it.

    What it buys, in descending order: a hard cap on deployed deviation; an
    exactly-recoverable δ = 0 point that `evaluate_baseline` scores on the same
    reference and the same windows; and a start at the rule itself.

    Note Δ bounds the *action*, not the outcome. The commanded setpoint stays
    within Δ of the rule's; the zone temperature does not, because the plant is
    in between. A setpoint→zone gain well under 1 °C/°C makes Δ = 1 °C of
    setpoint comfortably under 1 °C of zone deviation — state it that way
    rather than as a comfort guarantee.

    The box may be asymmetric — [0, 0.5] lets a heating setpoint anchored at
    the band floor move up and never down. The net is bounded to the symmetric
    ±max(below, above) and its output clamped to the box, rather than bounded
    to the box itself: a sigmoid squash cannot reach an edge, so a box with a
    zero side would have no output for "the rule exactly", and training could
    not start there. A zero side is a hard floor, with no gradient past it —
    the same as the actuator clamp.

    Order is easy to get backwards and matters: the reference is already
    grid-snapped and δ is not, so the **sum** is what gets snapped, and the
    clamp comes **last** — quantizing after clamping can push a value back
    outside the actuator box.
    """

    def __init__(self, base: nn.Module, reference: _ReferenceRule,
                 umin: torch.Tensor, umax: torch.Tensor,
                 below: torch.Tensor, above: torch.Tensor, quantize=None):
        super().__init__()
        self.base = base
        self.reference = reference
        self.quantize = quantize
        self.register_buffer("umin", umin)
        self.register_buffer("umax", umax)
        self.register_buffer("below", below)
        self.register_buffer("above", above)

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        delta = torch.clamp(self.base(*args), -self.below, self.above)
        u = self.reference(*args) + delta
        if self.quantize is not None:
            u = self.quantize(u)
        return torch.clamp(u, self.umin, self.umax)


def _zero_final_layer(net: nn.Module) -> None:
    """Start the residual at δ ≡ 0, exactly.

    Both bound methods send a pre-bound output of 0 to the centre of the
    net's symmetric box: `sigmoid_scale` gives 2Δ·σ(0) − Δ = 0, and
    `relu_clamp` is the identity there. So zeroing the last linear layer makes the policy emit
    the reference rule exactly at initialization — training starts at the rule.
    The gradient at that point is also maximal (σ′(0) = ¼), so it is a
    responsive start, not a dead one.
    """
    last = net.linear[-1]
    nn.init.zeros_(last.weight)
    if last.bias is not None:
        nn.init.zeros_(last.bias)


def _validated_preview_spec(hooks, nm_mapping: dict[str, list[str]]
                            ) -> Optional[dict[str, int]]:
    """Read + validate control_user_defined.preview_config, or None if unset.

    None (the hook is missing) is a different case from {} (the hook is
    defined but names nothing): the former falls back to `cfg.d_preview`
    below, the latter turns disturbance preview off entirely.
    """
    preview = getattr(hooks, "preview_config", None)
    if preview is None:
        return None

    spec = preview(nm_mapping)
    for key, steps in spec.items():
        if key not in nm_mapping["D"]:
            raise ValueError(
                f"preview_config key '{key}' is not a D column. Allowed: "
                f"{nm_mapping['D']}. A reference column's preview is "
                "`reference_preview` in the control config."
            )
        if not isinstance(steps, int) or steps < 0:
            raise ValueError(
                f"preview_config['{key}'] must be a non-negative int, got {steps!r}"
            )
    return spec


def _d_preview_by_column(hooks, cfg, nm_mapping: dict[str, list[str]]
                         ) -> list[int]:
    """Each D column's own preview depth, in nm_mapping["D"] order.

    From the use case's `preview_config(nm_mapping) -> {D column: n_steps}`
    hook; with no hook, `cfg.d_preview` for every column. How far ahead to
    look at the weather is a property of the plant, shared by every run
    against it, so it lives in the hooks file; how far ahead to look at a
    reference column is per run, and is `cfg.reference_preview`.
    """
    spec = _validated_preview_spec(hooks, nm_mapping)
    if spec is None:
        return [cfg.d_preview] * len(nm_mapping["D"])
    return [spec.get(col, 0) for col in nm_mapping["D"]]


def _preview_gather(n: int, history: int, depths: list[int]
                    ) -> Optional[torch.Tensor]:
    """Indices narrowing a shared window to each column's own preview depth.

    The tensor SystemPreview hands the policy for one key is one window,
    time-major, covering `history - 1` past steps, the current step, and
    `max(depths)` future steps (see `SystemPreview.get_mapped_data`) — the
    deepest depth any single column declared, since the window can't vary by
    column.

    When every column declared the same depth (including the all-zero
    default) that window already is what was asked for, so this returns None
    and build_policy builds no gather. Otherwise it keeps, at every past and
    current step, all `n` columns, and at future step k (1-indexed from the
    current step) only the columns whose declared depth is >= k.

    Reuses `windows.key_blocks`/`per_step` for the per-timestep offsets — the
    same position arithmetic `BlockStack` uses — rather than re-deriving
    `t * n + j` here.
    """
    future = max(depths, default=0)
    if all(depth == future for depth in depths):
        return None

    in_steps = history + future
    block = windows.key_blocks((n,), in_steps)[0]
    steps = windows.per_step(block, n, in_steps)

    indices = []
    for t, step in enumerate(steps):
        future_step = t - history + 1        # <= 0 for the past/current steps
        cols = (range(n) if future_step <= 0 else
                [j for j in range(n) if depths[j] >= future_step])
        indices.extend(step.start + j for j in cols)
    return torch.tensor(indices, dtype=torch.long)


def _build_mlp_net(cfg, insize: int, nu: int, umin, umax) -> nn.Module:
    """Bounded MLP with `policy_layers` hidden layers of equal width."""
    return blocks.MLP_bounds(
        insize=insize,
        outsize=nu,
        bias=True,
        linear_map=nn.Linear,
        nonlin=POLICY_NONLIN[cfg.policy_nonlin],
        hsizes=cfg.policy_layers * [2**cfg.log2_policy_hidden],
        min=umin,
        max=umax,
        method=cfg.policy_bound_method,
    )


def _build_linear_net(cfg, insize: int, nu: int, umin, umax) -> nn.Module:
    """Saturated affine feedback law: one linear layer, then the bound squash.

    hsizes=[] makes blocks.MLP build a single linear_map(insize, nu) with an
    Identity nonlinearity, so this is u = sat(K @ x + b) — no hidden layer.
    """
    return blocks.MLP_bounds(
        insize=insize,
        outsize=nu,
        bias=True,
        linear_map=nn.Linear,
        nonlin=nn.Identity,
        hsizes=[],
        min=umin,
        max=umax,
        method=cfg.policy_bound_method,
    )


policy_catalog = {
    "mlp": _build_mlp_net,
    "linear": _build_linear_net,
}

# Bound squashing options come from Neuromancer, so read them off the class
# rather than re-listing them here.
POLICY_BOUND_METHODS = list(blocks.MLP_bounds.bound_methods)


def _u_scaling(nm_mapping, norm_stats, device) -> tuple[torch.Tensor, torch.Tensor]:
    """(stretch, shift) per U column — the physical↔normalized conversion."""
    def col(key):
        return torch.tensor([norm_stats[c][key] for c in nm_mapping["U"]],
                            dtype=torch.float32, device=device)
    return col("stretch"), col("shift")


def _reference_rule(cfg, hooks, nm_mapping, norm_stats, h_ctrl: int,
                    r_columns: list[str]):
    """The residual's anchor rule, or None when `residual_delta` is unset.

    No silent "no reference → plain policy" path: a residual anchored to a bad
    rule is a bad controller with extra steps, and the fix for that is to write
    the rule, not to quietly drop the anchor.
    """
    if cfg.residual_delta is None:
        return None
    rule = getattr(hooks, "reference_control", None)
    if rule is None:
        raise NotImplementedError(
            f"residual_delta is set ({cfg.residual_delta}) but "
            "control_user_defined.reference_control is not defined, so there "
            "is no rule for the residual to deviate from. Implement the hook, "
            "or set residual_delta to null for a plain policy."
        )
    return _ReferenceRule(rule, nm_mapping, norm_stats, h_ctrl, r_columns)


def _input_keys(r_columns: list[str]) -> list[str]:
    return ["yn", "D"] + (["R"] if r_columns else [])


@beartype
def build_reference_node(
    cfg,
    sysid_cfg,
    nm_mapping: dict[str, list[str]],
    norm_stats: NormStats,
) -> Node:
    """The use case's reference rule as a standalone controller Node.

    One rule, two consumers. The same `reference_control` the residual policy
    is anchored to is what `evaluate_baseline` scores, so the δ = 0 floor and
    the baseline column are the same controller by construction. Two copies of
    a cooling inset is two places for it to drift, and the record says that
    inset is what decides whether the rule works at all.

    A history of one step and no preview means every window collapses to the
    current step, so `_ReferenceRule`'s slicing is the identity here and the
    Node reads plain current-step tensors. It has no parameters: it is
    evaluated, never trained.
    """
    hooks = load_control_hooks(cfg)
    rule = getattr(hooks, "reference_control", None)
    if rule is None:
        raise NotImplementedError(
            "control_user_defined.reference_control is not defined."
        )
    r_columns = resolve_objective(cfg, nm_mapping).r_columns
    net = _ReferenceRule(rule, nm_mapping, norm_stats, h_ctrl=1,
                         r_columns=r_columns).to(sysid_cfg.device)
    return Node(net, _input_keys(r_columns), ["U"], name="Reference rule")


@beartype
def reference_over_horizon(
    cfg,
    out: dict,
    hmax: int,
    nm_mapping: dict[str, list[str]],
    norm_stats: NormStats,
) -> torch.Tensor:
    """The reference rule replayed on a recorded rollout, [b, P, nu] normalized.

    The rule is memoryless, so every (window, step) can be evaluated at once by
    folding time into the batch axis. What comes back is exactly the command
    the rule would have issued at each step of this trajectory, which makes
    `U[:, scored_steps("U", hmax)] - u_ref` the *realized* residual — after
    quantization and after the actuator clamp, i.e. the deviation that was
    actually deployed.

    The command at `U[hmax - 1 + k]` was chosen from the signals at that same
    index, so the rule's inputs are read over the command slice, not over
    `[hmax:]` like the scored outputs. Reading them at `hmax + k` would
    silently compare each command against the *next* step's rule, which on a
    reference that moves is a residual that never was.
    """
    rule = getattr(load_control_hooks(cfg), "reference_control", None)
    if rule is None:
        raise NotImplementedError(
            "control_user_defined.reference_control is not defined."
        )
    r_columns = resolve_objective(cfg, nm_mapping).r_columns
    # The other keys run one step past U, so bound the slice by U's length.
    steps = slice(scored_steps("U", hmax).start, out["U"].shape[1])
    P = steps.stop - steps.start
    b = out["U"].shape[0]

    def flat(key):
        x = out[key][:, steps, :]
        return x.reshape(b * P, x.shape[-1])

    r_now = {} if not r_columns else {
        name: flat("R")[:, j] for j, name in enumerate(r_columns)}
    return rule(flat("yn"), r_now, flat("D"), nm_mapping,
                norm_stats).reshape(b, P, -1)


@beartype
def build_policy(
    cfg,
    sysid_cfg,
    nm_mapping: dict[str, list[str]],
    norm_stats: NormStats,
) -> Node:
    if cfg.policy not in policy_catalog:
        raise ValueError(
            f"Unknown policy '{cfg.policy}'. "
            f"Available: {list(policy_catalog.keys())}"
        )

    ny = len(nm_mapping["Y"])
    nu = len(nm_mapping["U"])
    nd = len(nm_mapping["D"])
    hooks = load_control_hooks(cfg)
    _, h_ctrl, _ = horizons(cfg, sysid_cfg)
    r_columns = resolve_objective(cfg, nm_mapping).r_columns
    device = sysid_cfg.device

    # How many future steps of each column the policy sees: per D column from
    # the hooks' preview_config (or cfg.d_preview), per reference column from
    # cfg.reference_preview.
    d_depths = _d_preview_by_column(hooks, cfg, nm_mapping)
    r_depths = [cfg.reference_preview.get(c, 0) for c in r_columns]

    # Flattened window widths (see SystemPreview.get_mapped_data):
    #   yn : past = H_ctrl-1, future = 0             -> ny * H_ctrl
    #   D  : past+current = H_ctrl, plus each column's own future depth
    #                                                 -> h_ctrl*nd + sum(d_depths)
    #   R  : current, plus each column's own future   -> n_ref + sum(r_depths)
    #
    # A key's window is sized to its deepest column, and a column asking for
    # less is gathered down to its own depth before the net sees it
    # (_preview_gather below), so the widths are per-column sums.
    insize = (
        ny * h_ctrl
        + h_ctrl * nd + sum(d_depths)
        + len(r_columns) + sum(r_depths)
    )

    umin, umax = hooks.actuation_bounds(nm_mapping, norm_stats)
    umin, umax = umin.to(device), umax.to(device)
    u_stretch, u_shift = _u_scaling(nm_mapping, norm_stats, device)

    reference = _reference_rule(cfg, hooks, nm_mapping, norm_stats, h_ctrl,
                                r_columns)
    if reference is None:
        net_min, net_max = umin, umax
    else:
        # A residual's box is around the rule, in physical units converted per
        # U column exactly as setpoint_step is. The net takes the symmetric
        # ±max of it; _ResidualPolicy clamps to the box itself.
        below, above = (side / u_stretch for side in cfg.residual_box())
        net_max = torch.maximum(below, above)
        net_min = -net_max

    net = policy_catalog[cfg.policy](cfg, insize, nu, net_min, net_max).to(device)

    quantize = None
    if cfg.setpoint_step > 0:
        # Quantization is in physical °C, but the policy emits normalized U;
        # convert step + shift per channel so the physical grid lands at
        # multiples of cfg.setpoint_step (not multiples of step shifted by
        # the column mean).
        quantize = STEQuantize(cfg.setpoint_step / u_stretch,
                               u_shift / u_stretch)

    if reference is not None:
        _zero_final_layer(net)

    # Per-column preview is a gather in front of whatever net was just built,
    # applied before the residual/quantize wrapping below so the reference
    # (which slices its own current step directly, see _ReferenceRule) and
    # the trainable net see the same windows either way.
    gathers = {}
    d_gather = _preview_gather(nd, h_ctrl, d_depths)
    if d_gather is not None:
        gathers[_D_ARG_INDEX] = d_gather.to(device)
    r_gather = _preview_gather(len(r_columns), 1, r_depths)
    if r_gather is not None:
        gathers[_R_ARG_INDEX] = r_gather.to(device)
    if gathers:
        net = _PreviewGather(net, gathers)

    if reference is not None:
        net = _ResidualPolicy(net, reference, umin, umax, below, above,
                              quantize).to(device)
    elif quantize is not None:
        net = _QuantizedPolicy(net, quantize).to(device)

    input_map = {
        "yn": {"past": h_ctrl - 1, "future": 0, "pad_mode": "nearest"},
        "D": {"past": h_ctrl - 1, "future": max(d_depths, default=0),
              "pad_mode": "nearest"},
    }
    # A reference column is a known future schedule; previewing it lets the
    # policy see what it is steering into (the dynamics have a one-step lag,
    # so a bound at step i depends on the command chosen at i-1). Only add the
    # window when previewing — future=0 is the single current step.
    if max(r_depths, default=0) > 0:
        input_map["R"] = {"past": 0, "future": max(r_depths),
                          "pad_mode": "nearest"}

    return Node(
        net,
        _input_keys(r_columns),
        ["U"],
        name="DPC Policy",
        input_map=input_map,
    )


# ── Policy description (for reportcard) ──────────────────────────


@beartype
def describe_policy(
    policy_node: Node,
    cfg,
    sysid_cfg,
    nm_mapping: dict[str, list[str]],
    norm_stats: NormStats,
) -> dict:
    """
    Describe the constructed policy for the reportcard.

    Mirrors modeling.dynamics.describe_dynamics: reads what is available on
    the built module rather than re-deriving it from cfg, so a mismatch
    between config and reality shows up on the card.

    Preview depths are reported alongside `insize` so a reader can re-check
    the ny*H_ctrl + [h_ctrl*nd + sum(D depths)] + [n_ref + sum(R depths)] sum
    by eye — a wrong window is otherwise an invisible failure.
    """
    # Unwrap: Node -> (_ResidualPolicy | _QuantizedPolicy) -> [_PreviewGather] -> MLP_bounds
    raw = policy_node.callable
    residual = isinstance(raw, _ResidualPolicy)
    wrapped = isinstance(raw, (_ResidualPolicy, _QuantizedPolicy))
    net = raw.base if wrapped else raw
    quantized = raw.quantize is not None if wrapped else False
    if isinstance(net, _PreviewGather):
        net = net.base

    hooks = load_control_hooks(cfg)
    _, h_ctrl, _ = horizons(cfg, sysid_cfg)
    d_depths = _d_preview_by_column(hooks, cfg, nm_mapping)
    r_columns = resolve_objective(cfg, nm_mapping).r_columns

    # Actuator limits are set in normalized U space by the actuation_bounds
    # hook; report them in physical units, which is what a reader can sanity
    # check against the building. For a residual policy the network's own box
    # is ±Δ, so the limits the commands actually live in are the clamp's.
    u_cols = nm_mapping["U"]
    stretch = torch.tensor([norm_stats[c]["stretch"] for c in u_cols])
    shift = torch.tensor([norm_stats[c]["shift"] for c in u_cols])
    lo, hi = (raw.umin, raw.umax) if residual else (net.min, net.max)
    umin_phys = (lo.detach().cpu() * stretch + shift).tolist()
    umax_phys = (hi.detach().cpu() * stretch + shift).tolist()

    return {
        "family": cfg.policy,
        "nonlin": cfg.policy_nonlin if cfg.policy == "mlp" else "n/a (linear)",
        "bound_method": cfg.policy_bound_method,
        "insize": int(net.in_features),
        "outsize": int(net.out_features),
        "h_ctrl": h_ctrl,
        "ny": len(nm_mapping["Y"]),
        "d_preview_by_column": dict(zip(nm_mapping["D"], d_depths)),
        "r_preview_by_column": {c: cfg.reference_preview.get(c, 0)
                                for c in r_columns},
        "hsizes": [int(lin.out_features) for lin in net.linear[:-1]],
        "param_count": sum(p.numel() for p in net.parameters()),
        "u_cols": list(u_cols),
        "umin_phys": umin_phys,
        "umax_phys": umax_phys,
        "quantized": quantized,
        "setpoint_step": cfg.setpoint_step,
        "residual_box": cfg.residual_box() if residual else None,
    }
