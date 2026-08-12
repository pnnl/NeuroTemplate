"""
Closed-loop assembly + DPC objective.

    SystemPreview([policy_node, frozen_dyn_node])

At rollout step i: the policy writes U (appended to data['U']); the frozen
dynamics consumes a U/D/yn window ending at i and writes the next yn (fed
back). With start_iter = Hmax-1 and explicit nsteps = P, predicted yn occupies
indices [Hmax : Hmax+P], aligned with the comfort-band signal slice
[Hmax : Hmax+P].

A stateful surrogate needs its hidden state cleared between batches, which is
all init_func does here — it warms itself from the same H_sys-wide window a
stateless surrogate reads, so the two enter the rollout on equal footing.
"""

from __future__ import annotations


import torch
from torch.nn.functional import relu

from neuromancer.system import SystemPreview
from neuromancer.problem import Problem
from neuromancer.constraint import variable
from neuromancer.loss import PenaltyLoss

from neurotemplate.control.scenarios import horizons, price_scale
from neurotemplate.modeling.dynamics import make_reset_init
from neurotemplate.hooks import load_control_hooks

from neurotemplate._typing import beartype


@beartype
def build_closed_loop(dyn_node, policy_node, cfg, sysid_cfg) -> SystemPreview:
    """Policy + frozen dynamics, rolled out for cfg.train_pred_steps steps."""
    _, _, hmax = horizons(cfg, sysid_cfg)

    return SystemPreview(
        [policy_node, dyn_node],
        name="closed_loop",
        nstep_key="D",
        nsteps=cfg.train_pred_steps,
        start_iter=hmax - 1,
        # Stateful surrogates only: clear the hidden state so it cannot leak
        # between batches. None for stateless ones.
        init_func=make_reset_init(dyn_node.callable),
    )


@beartype
def resolve_du_weights(du_weight, u_cols: list[str]) -> list[float]:
    """Broadcast `cfg.du_weight` to one weight per control column.

    A scalar applies to every actuator, which is the old behaviour exactly.
    A list must be one entry per U column, in `nm_mapping["U"]` order; a dict
    is keyed by column name and any column left out gets 0 (no smoothing).

    Per-channel weights matter as soon as two actuators have different
    authority: on a plant where one moves the controlled channel 5x further
    per step than the other, a single pooled penalty is necessarily
    mis-weighted for one of them.
    """
    if isinstance(du_weight, (int, float)):
        return [float(du_weight)] * len(u_cols)
    if isinstance(du_weight, dict):
        unknown = [c for c in du_weight if c not in u_cols]
        if unknown:
            raise ValueError(
                f"du_weight names columns that are not controls: {unknown}. "
                f"U = {u_cols}"
            )
        return [float(du_weight.get(c, 0.0)) for c in u_cols]
    if len(du_weight) != len(u_cols):
        raise ValueError(
            f"du_weight has {len(du_weight)} entries but there are "
            f"{len(u_cols)} control columns ({u_cols})"
        )
    return [float(w) for w in du_weight]


@beartype
def power_term(cfg, nm_mapping: dict[str, list[str]],
               norm_stats: dict[str, dict[str, float]]):
    """The per-step power signal as a plain callable, or None when unimplemented.

    Signature is `(yn, U, D) -> [batch, time]` over the *scored horizon*
    slices. Shared with `control.objective` so the shape check sweeps exactly
    the quantity the loss minimizes, not a re-derivation of it.
    """
    hooks = load_control_hooks(cfg)
    if not getattr(hooks, "POWER_METRIC_IMPLEMENTED", False):
        return None

    def term(yn, u, d):
        return hooks.power_metric(yn, u, d, nm_mapping, norm_stats)

    return term


def _penalty(expr, name: str):
    """Label a scalar expression as a named objective term.

    `Objective.minimize(name=...)` does NOT propagate to the Problem's output
    keys — only `Constraint.update_name` does — so every term is wrapped in
    `expr == 0` and named here. That is the whole incantation, in one place.
    """
    term = expr == 0
    term.update_name(name)
    return term


@beartype
def build_control_loss(
    cfg,
    sysid_cfg,
    nm_mapping: dict[str, list[str]],
    norm_stats: dict[str, dict[str, float]],
) -> PenaltyLoss:
    """
    Build all control losses as scalar penalties, summed with flat weights.

    Five terms, in three groups:

      comfort  — squared band violation, both directions
      du       — per-channel setpoint smoothness
      energy / cost / peak — three *reductions of one power signal*

    That last line is the only structural claim here: the per-step power
    signal and the way it is reduced to a scalar are separate concerns.
    `energy` is its mean, `cost` is its price-weighted mean, and `peak` is the
    per-window power mean M_q — same signal, same units, three questions.

    All three need control_user_defined.power_metric, and `cost` additionally
    needs a `price_signal` hook; requesting one without its hook fails fast
    rather than silently substituting a proxy.

    Nothing here checks that a weight suits the magnitude of what it
    multiplies — that is `control.objective`, run once on a real batch before
    training, because it takes a forward pass to know.
    """
    hooks = load_control_hooks(cfg)
    _, _, hmax = horizons(cfg, sysid_cfg)
    _, ctrl_idx = hooks.controlled_channels(nm_mapping)

    yn = variable("yn")
    ymin = variable("ymin")
    ymax = variable("ymax")
    u = variable("U")
    d = variable("D")

    # Predicted controlled channels over the horizon vs. the band.
    y_ctrl = yn[:, hmax:, ctrl_idx]      # [b, P, n_ctrl]
    ymin_h = ymin[:, hmax:, :]           # [b, P, n_ctrl]
    ymax_h = ymax[:, hmax:, :]

    objectives = []

    if cfg.comfort_weight > 0:
        objectives += [
            _penalty(cfg.comfort_weight * torch.mean(relu(ymin_h - y_ctrl)**2),
                     "comfort_lower_loss"),
            _penalty(cfg.comfort_weight * torch.mean(relu(y_ctrl - ymax_h)**2),
                     "comfort_upper_loss"),
        ]

    du_weights = resolve_du_weights(cfg.du_weight, nm_mapping["U"])
    if any(w > 0 for w in du_weights):
        u_h = u[:, hmax:, :]            # policy-produced control [b, P, nu]
        du_abs = torch.abs(u_h[:, 1:, :] - u_h[:, :-1, :])
        # Weighted per channel, then averaged over channels — so a scalar
        # weight reproduces the pooled mean exactly.
        du = sum(w * torch.mean(du_abs[:, :, j])
                 for j, w in enumerate(du_weights)) / len(du_weights)
        objectives.append(_penalty(du, "du_loss"))

    objectives += _power_objectives(cfg, nm_mapping, norm_stats, hmax,
                                    yn, u, d)

    if not objectives:
        raise ValueError(
            "No active control objective. Set at least one of comfort_weight "
            "/ du_weight / energy_weight / cost_weight / peak_weight > 0."
        )

    return PenaltyLoss(objectives, constraints=[])


def _power_objectives(cfg, nm_mapping, norm_stats, hmax, yn, u, d) -> list:
    """The energy / cost / peak terms, all reductions of one power signal.

    Every reduction is a mean over the scored horizon (or a mean over the
    batch of per-window scalars), so a weight tuned at P=48 means the same
    thing at P=288. A sum would make each term's magnitude proportional to
    `train_pred_steps`; the reportcard does the integration instead.
    """
    weights = {"energy": cfg.energy_weight, "cost": cfg.cost_weight,
               "peak": cfg.peak_weight}
    if not any(w > 0 for w in weights.values()):
        return []

    term = power_term(cfg, nm_mapping, norm_stats)
    if term is None:
        requested = [f"{k}_weight" for k, w in weights.items() if w > 0]
        raise NotImplementedError(
            f"{', '.join(requested)} > 0 requires "
            "control_user_defined.power_metric to be implemented (set "
            "POWER_METRIC_IMPLEMENTED=True). Run with "
            + " ".join(f"--set {r}=0" for r in requested)
            + " until a power model exists."
        )

    def _power(yn_v, u_v, d_v):
        return term(yn_v[:, hmax:, :], u_v[:, hmax:, :], d_v[:, hmax:, :])

    # Built once and reduced three ways, so the terms cannot drift apart:
    # a cost that priced a different power signal than the energy term
    # measured would be invisible on every report.
    power = variable([yn, u, d], _power, display_name="power")   # [b, P]

    objectives = []
    if cfg.energy_weight > 0:
        objectives.append(
            _penalty(cfg.energy_weight * torch.mean(power), "energy_loss")
        )

    if cfg.cost_weight > 0:
        scale = price_scale(cfg)
        if scale is None:
            raise NotImplementedError(
                "cost_weight > 0 requires train_price_path/dev_price_path/"
                "test_price_path + price_scale so a tariff reaches the "
                "rollout. Run with --set cost_weight=0 until a tariff exists. "
                "Falling back to a flat price would silently turn the cost "
                "term into a rescaled energy term."
            )
        # Price is data: no gradient, non-negative. So the cost term is a
        # per-timestep *reweighting* of the energy gradient — it cannot flip
        # the sign of anything the energy term was already doing.
        price = variable("price")[:, hmax:, 0]           # [b, P], scaled
        objectives.append(
            _penalty(cfg.cost_weight * torch.mean(power * price * scale),
                     "cost_loss")
        )

    if cfg.peak_weight > 0:
        objectives.append(
            _penalty(cfg.peak_weight * _soft_peak(power, float(cfg.peak_q)),
                     "peak_loss")
        )

    return objectives


def _soft_peak(power, q: float):
    """Mean over the batch of each window's power mean M_q(p).

        M_q(p) = ( mean_t p^q )^(1/q)     q=1 -> mean, q=2 -> RMS, q->inf -> max

    Homogeneous of degree 1, so it has the same units and the same
    weight-scaling behaviour as the energy term — `peak_weight` and
    `energy_weight` are comparable on the balance report. A mean-square term
    would be homogeneous of degree 2 instead, so switching the power metric
    from kW to W would move its correct weight by 1e6 while every neighbour's
    moved by 1e3. Given that mis-scaled weights are this library's most
    frequent recorded failure, a term whose weight scales differently from its
    neighbours is worth refusing.

    The gradient `(1/P)·(p_t/M_q)^(q-1)` is dense over time and concentrated
    on the high-power steps by a factor growing with q. A hard `torch.max`
    would instead give one non-zero entry per window, and the policy would
    learn to shave exactly one timestep.

    Reduced per window and then averaged over the batch, never pooled: the
    batch axis holds independent windows (different days, possibly different
    files), so a pooled peak is the worst day's peak and gives gradient to one
    window and none to the rest. Per-window-then-batch is also exactly
    `EnergyMetrics.mean_window_peak`, so the thing being trained and the thing
    being reported are the same quantity, soft vs. hard.

    `relu` enforces p >= 0, which `power_metric` normally already does. For an
    even q a large negative prediction would otherwise read as a peak.
    """
    per_window = torch.mean(torch.relu(power) ** q, dim=1) ** (1.0 / q)
    return torch.mean(per_window)


@beartype
def build_problem(cl_system: SystemPreview, loss: PenaltyLoss) -> Problem:
    return Problem([cl_system], loss)
