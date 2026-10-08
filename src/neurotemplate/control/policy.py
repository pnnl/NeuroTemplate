"""
DPC control policy.

A bounded network wrapped in a Neuromancer Node:

    [yn-window, ymin, ymax, D-window]  ->  U   (per-zone cooling setpoints)

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

# Position of "D" among the policy net's positional inputs. Fixed by the
# input_keys list build_policy constructs — ["yn", "ymin", "ymax", "D"], with
# "price" (if any) always appended after — so this never moves.
_D_ARG_INDEX = 3


def regulated_channels(hooks, nm_mapping: dict[str, list[str]]
                       ) -> tuple[list[str], list[int]]:
    """The OCP regulated output: (names, Y-indices) of the Y channels the
    objective bands and the reportcard scores, from the `regulated` hook.
    """
    fn = getattr(hooks, "regulated", None)
    if fn is None:
        raise AttributeError(
            "control_user_defined must define `regulated(nm_mapping)`."
        )
    return fn(nm_mapping)


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
    """Narrow the D window to the (column, timestep) pairs actually declared.

    SystemPreview's input_map gives one window per key, so the D tensor the
    Node hands the policy is always sized to `d_future`, the deepest preview
    any single D column declared (see `_d_preview_gather`). A column that
    asked for less would otherwise see extra future it never requested. This
    applies the narrowing gather in front of whatever net build_policy
    constructed for the trainable inputs.

    A fixed permutation computed once at construction, applied with
    `index_select` against a registered buffer — same discipline as
    `BlockStack` (modeling/dynamics.py): no data-dependent control flow.
    """

    def __init__(self, base: nn.Module, gather: torch.Tensor, d_index: int):
        super().__init__()
        self.base = base
        self.d_index = d_index
        self.register_buffer("_gather", gather, persistent=False)

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        args = list(args)
        args[self.d_index] = args[self.d_index].index_select(-1, self._gather)
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
    hook plain `[b, n]` tensors.

    Deliberately memoryless: every rule-based controller in the record is a
    function of the band and the current state, and it is what the deploy-side
    callback can supply with no new plumbing. A rule that needs its own
    integrator is out of scope — say so rather than adding a state-carrying
    path for it.

    Extra inputs beyond the four (a price preview, say) are accepted and
    ignored: the rule holds comfort, and anything else is the residual's job.
    """

    def __init__(self, rule, nm_mapping: dict[str, list[str]],
                 norm_stats: NormStats,
                 h_ctrl: int, ny: int, n_ctrl: int, nd: int):
        super().__init__()
        self.rule = rule
        self.nm_mapping = nm_mapping
        self.norm_stats = norm_stats
        # Which block of each flattened window holds the current step. yn and
        # D look `h_ctrl - 1` steps back, so it is the last of their history;
        # the band has no history, so it is the first.
        self.yn_at, self.ny = h_ctrl - 1, ny
        self.d_at, self.nd = h_ctrl - 1, nd
        self.n_ctrl = n_ctrl

    def forward(self, yn: torch.Tensor, ymin: torch.Tensor,
                ymax: torch.Tensor, d: torch.Tensor, *_) -> torch.Tensor:
        return self.rule(
            _current(yn, self.yn_at, self.ny),
            _current(ymin, 0, self.n_ctrl),
            _current(ymax, 0, self.n_ctrl),
            _current(d, self.d_at, self.nd),
            self.nm_mapping,
            self.norm_stats,
        )


class _ResidualPolicy(nn.Module):
    """Command a bounded deviation from a reference rule, not a setpoint.

        δ = MLP_bounds(window)                    bounded to [-Δ, +Δ]
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
    band and the same windows; and a start at the rule itself.

    Note Δ bounds the *action*, not the outcome. The commanded setpoint stays
    within Δ of the rule's; the zone temperature does not, because the plant is
    in between. A setpoint→zone gain well under 1 °C/°C makes Δ = 1 °C of
    setpoint comfortably under 1 °C of zone deviation — state it that way
    rather than as a comfort guarantee.

    Order is easy to get backwards and matters: the reference is already
    grid-snapped and δ is not, so the **sum** is what gets snapped, and the
    clamp comes **last** — quantizing after clamping can push a value back
    outside the actuator box.
    """

    def __init__(self, base: nn.Module, reference: _ReferenceRule,
                 umin: torch.Tensor, umax: torch.Tensor,
                 quantize=None):
        super().__init__()
        self.base = base
        self.reference = reference
        self.quantize = quantize
        self.register_buffer("umin", umin)
        self.register_buffer("umax", umax)

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        u = self.reference(*args) + self.base(*args)
        if self.quantize is not None:
            u = self.quantize(u)
        return torch.clamp(u, self.umin, self.umax)


def _zero_final_layer(net: nn.Module) -> None:
    """Start the residual at δ ≡ 0, exactly.

    Both bound methods send a pre-bound output of 0 to the centre of a
    symmetric box: `sigmoid_scale` gives 2Δ·σ(0) − Δ = 0, and `relu_clamp` is
    the identity there. So zeroing the last linear layer makes the policy emit
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
    below, the latter turns disturbance preview off entirely. Shared by
    `_preview_steps` (the single largest-depth summary the shared D/band
    windows are sized to) and `_d_preview_by_column` (every column's own
    depth, which is what the policy actually gathers down to), so the hook
    is read and validated in exactly one place.
    """
    preview = getattr(hooks, "preview_config", None)
    if preview is None:
        return None

    spec = preview(nm_mapping)
    allowed = set(nm_mapping["D"]) | {"band"}
    for key, steps in spec.items():
        if key not in allowed:
            raise ValueError(
                f"preview_config key '{key}' is not a D column or 'band'. "
                f"Allowed: {sorted(allowed)}"
            )
        if not isinstance(steps, int) or steps < 0:
            raise ValueError(
                f"preview_config['{key}'] must be a non-negative int, got {steps!r}"
            )
    return spec


def _preview_steps(hooks, cfg,
                   nm_mapping: dict[str, list[str]]) -> tuple[int, int, int]:
    """Resolve (disturbance, band, price) future-preview widths for the policy.

    The tariff preview comes from `cfg.price_preview`; the other two come from
    the use case's `preview_config(nm_mapping) -> {signal: n_steps}` hook, whose
    keys may be D column names (per-column disturbance preview) or the literal
    "band" (future comfort-band preview). The disturbance *window* is one
    block, so it is sized to the largest per-column D value requested — a
    column asking for less is gathered back down to its own depth before the
    net sees it (`_d_preview_by_column`, `_d_preview_gather`), so "largest
    value requested" here is a window size, not what any one column reads.
    With no hook, fall back to the global `cfg.d_preview` for disturbances and
    no band preview.

    The split with price is deliberate. How far ahead to look at the weather
    is a property of the plant, shared by every run against it, and belongs in
    the hooks file. How far ahead to look at the tariff is a per-run tuning
    knob — and while it lived in the hook, one config wanting a price preview
    forced price paths onto every other config sharing that hook file, because
    the hook cannot see the config and so cannot ask whether a tariff is
    configured at all.
    """
    price_future = cfg.price_preview
    spec = _validated_preview_spec(hooks, nm_mapping)
    if spec is None:
        return cfg.d_preview, 0, price_future
    d_future = max([spec.get(col, 0) for col in nm_mapping["D"]], default=0)
    return d_future, spec.get("band", 0), price_future


def _d_preview_by_column(hooks, cfg, nm_mapping: dict[str, list[str]]
                         ) -> list[int]:
    """Each D column's own declared preview depth, in nm_mapping["D"] order.

    `_preview_steps` collapses these to one number — the largest, because
    that's all the shared window can be sized to. This keeps every column's
    own value, which `_d_preview_gather` uses to narrow that shared window
    back down to what each column actually asked for.
    """
    spec = _validated_preview_spec(hooks, nm_mapping)
    if spec is None:
        return [cfg.d_preview] * len(nm_mapping["D"])
    return [spec.get(col, 0) for col in nm_mapping["D"]]


def _d_preview_gather(nd: int, h_ctrl: int, d_future: int,
                      d_depths: list[int]) -> Optional[torch.Tensor]:
    """Indices narrowing the shared D window to each column's own depth.

    The D tensor SystemPreview hands the policy is one window, time-major,
    covering `h_ctrl - 1` past steps, the current step, and `d_future` future
    steps (see `SystemPreview.get_mapped_data`) — `d_future` being the deepest
    depth any single column declared, since the window can't vary by column.

    When every column declared the same depth (including the all-zero
    default) that window already is what was asked for, so this returns None
    and build_policy leaves the net exactly as it was before per-column
    preview existed. Otherwise it keeps, at every past/current step, all `nd`
    columns (always fully visible), and at future step k (1-indexed from the
    current step) only the columns whose declared depth is >= k.

    Reuses `windows.key_blocks`/`per_step` for the per-timestep offsets — the
    same position arithmetic `BlockStack` uses — rather than re-deriving
    `t * nd + j` here.
    """
    if all(depth == d_future for depth in d_depths):
        return None

    in_steps = h_ctrl + d_future
    block = windows.key_blocks((nd,), in_steps)[0]
    steps = windows.per_step(block, nd, in_steps)

    indices = []
    for t, step in enumerate(steps):
        future_step = t - h_ctrl + 1        # <= 0 for the past/current steps
        cols = (range(nd) if future_step <= 0 else
               [j for j in range(nd) if d_depths[j] >= future_step])
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


def _reference_rule(cfg, hooks, nm_mapping, norm_stats,
                    h_ctrl: int, ny: int, n_ctrl: int, nd: int):
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
    return _ReferenceRule(rule, nm_mapping, norm_stats, h_ctrl, ny, n_ctrl, nd)


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

    A history of one step and no band preview means every window collapses to
    the current step, so `_ReferenceRule`'s slicing is the identity here and
    the Node reads plain current-step tensors. It has no parameters: it is
    evaluated, never trained.
    """
    hooks = load_control_hooks(cfg)
    rule = getattr(hooks, "reference_control", None)
    if rule is None:
        raise NotImplementedError(
            "control_user_defined.reference_control is not defined."
        )
    net = _ReferenceRule(
        rule, nm_mapping, norm_stats, h_ctrl=1,
        ny=len(nm_mapping["Y"]),
        n_ctrl=len(regulated_channels(hooks, nm_mapping)[0]),
        nd=len(nm_mapping["D"]),
    ).to(sysid_cfg.device)
    return Node(net, ["yn", "ymin", "ymax", "D"], ["U"],
                name="Reference rule")


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
    band that moves is a residual that never was.
    """
    rule = getattr(load_control_hooks(cfg), "reference_control", None)
    if rule is None:
        raise NotImplementedError(
            "control_user_defined.reference_control is not defined."
        )
    # The other keys run one step past U, so bound the slice by U's length.
    steps = slice(scored_steps("U", hmax).start, out["U"].shape[1])
    P = steps.stop - steps.start
    parts = [out[k][:, steps, :] for k in ("yn", "ymin", "ymax", "D")]
    b = parts[0].shape[0]
    flat = [x.reshape(b * P, x.shape[-1]) for x in parts]
    return rule(*flat, nm_mapping, norm_stats).reshape(b, P, -1)


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
    n_ctrl = len(regulated_channels(hooks, nm_mapping)[0])
    device = sysid_cfg.device

    # How many future steps each previewed signal feeds the policy. A use case
    # may define control_user_defined.preview_config for per-column disturbance
    # preview, comfort-band preview and price preview; otherwise fall back to
    # the global cfg.d_preview (disturbances only, nothing else previewed).
    d_future, band_future, price_future = _preview_steps(hooks, cfg, nm_mapping)
    d_depths = _d_preview_by_column(hooks, cfg, nm_mapping)
    # The tariff reaches the policy exactly when a price path is configured.
    reads_price = cfg.train_price_path is not None

    # Flattened window widths (see SystemPreview.get_mapped_data):
    #   yn : past = H_ctrl-1, future = 0                 -> ny * H_ctrl
    #   D  : past+current = H_ctrl, plus each column's own future depth
    #                                                     -> h_ctrl*nd + sum(d_depths)
    #   ymin/ymax: current + band_future preview steps   -> n_ctrl*(1+band_future) each
    #   price: current + price_future preview steps      -> 1 * (1+price_future)
    #
    # D's term is h_ctrl*nd + sum(d_depths) rather than nd*(h_ctrl+d_future):
    # the window SystemPreview builds is sized to d_future, the largest depth
    # any column declared, but a column asking for less is gathered down to
    # its own depth before the net sees it (_d_preview_gather below). The two
    # formulas agree exactly when every column shares one depth.
    insize = (
        ny * h_ctrl
        + 2 * n_ctrl * (1 + band_future)
        + h_ctrl * nd + sum(d_depths)
        + (1 + price_future if reads_price else 0)
    )

    umin, umax = hooks.actuation_bounds(nm_mapping, norm_stats)
    umin, umax = umin.to(device), umax.to(device)
    u_stretch, u_shift = _u_scaling(nm_mapping, norm_stats, device)

    reference = _reference_rule(cfg, hooks, nm_mapping, norm_stats,
                                h_ctrl, ny, n_ctrl, nd)
    if reference is None:
        net_min, net_max = umin, umax
    else:
        # A residual policy is bounded to a symmetric box around the rule, in
        # physical units converted per U column exactly as setpoint_step is.
        delta_norm = cfg.residual_delta / u_stretch
        net_min, net_max = -delta_norm, delta_norm

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

    # Per-column D preview is a gather in front of whatever net was just
    # built, applied before the residual/quantize wrapping below so both the
    # reference (which slices its own current step directly, see
    # _ReferenceRule) and the trainable net see the same D tensor either way.
    d_gather = _d_preview_gather(nd, h_ctrl, d_future, d_depths)
    if d_gather is not None:
        net = _PreviewGather(net, d_gather.to(device), _D_ARG_INDEX)

    if reference is not None:
        net = _ResidualPolicy(net, reference, umin, umax, quantize).to(device)
    elif quantize is not None:
        net = _QuantizedPolicy(net, quantize).to(device)

    input_map = {
        "yn": {"past": h_ctrl - 1, "future": 0, "pad_mode": "nearest"},
        "D": {"past": h_ctrl - 1, "future": d_future, "pad_mode": "nearest"},
    }
    # The comfort band is a known future schedule; previewing it lets the policy
    # see the band it is steering into (the dynamics have a one-step lag, so
    # comfort at step i depends on the setpoint chosen at i-1). Only add the
    # window when previewing — future=0 keeps the legacy single-step behaviour.
    if band_future > 0:
        for key in ("ymin", "ymax"):
            input_map[key] = {"past": 0, "future": band_future,
                              "pad_mode": "nearest"}

    input_keys = ["yn", "ymin", "ymax", "D"]
    if reads_price:
        input_keys.append("price")
        if price_future > 0:
            input_map["price"] = {"past": 0, "future": price_future,
                                  "pad_mode": "nearest"}

    return Node(
        net,
        input_keys,
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

    Window widths are reported alongside `insize` so a reader can re-check the
    ny*H_ctrl + 2*n_ctrl*(1+band) + [h_ctrl*nd + sum(per-column D depths)] +
    (1+price) sum by eye — a wrong window is otherwise an invisible failure.
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
    d_future, band_future, price_future = _preview_steps(hooks, cfg, nm_mapping)
    d_depths = _d_preview_by_column(hooks, cfg, nm_mapping)

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
        "d_preview": d_future,
        "d_preview_by_column": dict(zip(nm_mapping["D"], d_depths)),
        "band_preview": band_future,
        "price_preview": price_future,
        "reads_price": "price" in policy_node.input_keys,
        "hsizes": [int(lin.out_features) for lin in net.linear[:-1]],
        "param_count": sum(p.numel() for p in net.parameters()),
        "u_cols": list(u_cols),
        "umin_phys": umin_phys,
        "umax_phys": umax_phys,
        "quantized": quantized,
        "setpoint_step": cfg.setpoint_step,
        "residual_delta": cfg.residual_delta if residual else None,
    }
