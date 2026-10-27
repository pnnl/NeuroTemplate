# Control

`nt control train` trains a differentiable predictive control (DPC) policy.
The policy is a small network, or a saturated linear law, that reads recent
outputs, disturbances and any reference columns, and commands the `U`
columns. It is trained by rolling the closed loop through a frozen surrogate
from [MODELING.md](MODELING.md) and following the gradient of an objective
you declare. After training it runs alone: one cheap call per control step,
with no optimizer online.

Everything it learns, it learns from the surrogate. A policy is only as
good as the surrogate is about the actuators, and the surrogate's own
scores for a policy are facts about the surrogate. Judge controllers
against each other and against a rule on the same surrogate, and judge the
winner on the plant.

## What you provide

The hooks in `control_user_defined.py`, scaffolded in
[template/](../template/) with a docstring each:

| Hook | Says | |
|---|---|---|
| `actuation_bounds` | the box each control may command | required |
| `power_channels` | which outputs meter power: the plots, and `power_metric`'s default | required |
| `plot_context_channels` | disturbances drawn on the closed-loop plot | required |
| `power_metric` | the power signal that energy, cost and peak terms reduce; by default the sum of `power_channels` | to replace the default with a proxy |
| `reference_control` | a rule to anchor a residual policy to | with `residual_delta` |

The surrogate is a config field, `sysid_run_id`. Column roles and
normalization come from it. Everything else, including the objective and what the policy reads, is in
`control_configs/control.yaml`, which explains each setting and how to size
it.

```bash
nt control train --config control_configs/debug.yaml --set sysid_run_id=<model_run>
```

## The objective is a table over named columns

```yaml
losses:
  low:    {column: "Band Low",  target: "Zone Temperature", map: relu,  norm: 2}
  high:   {column: "Band High", target: "Zone Temperature", map: nrelu, norm: 2}
  moves:  {column: "Cooling Setpoint", map: first_difference, norm: 1}
  cost:   {column: power, norm: 1, weight: "Tariff"}
gains: {low: 1.0, high: 1.0, moves: 1.0, cost: 1.0}    # syntax, not values
```

Each row reads in one order: the operand `column − target`, then the `map`,
then the `norm` over the window, then its gain. Nothing else is in the
objective. `relu` makes a lower bound and `nrelu` an upper one, so the same
column on both sides is a tracked schedule. The reductions do not depend on
the horizon, and terms are in physical units by default, so a gain is an
exchange rate per °C or per W. Gains therefore do not port between plants.

The set of regulated outputs is derived from the `relu` and `nrelu` rows,
and the reportcard scores bands and band changes against exactly those.

**Reference columns are data, in their own files.** A name that is neither
a surrogate column nor `power`, such as a band, a schedule or a tariff, is
read from reference files: one per plant file, row-aligned with it and
carrying exactly its timestamps. The policy sees them; the surrogate never
does. [template/make_references.py](../template/make_references.py) writes
such files, and it is example code to edit. Listing one plant file against
several reference files trains on several draws.

## What the policy reads: `policy_window`

```yaml
policy_window:                       # syntax, not values
  "Zone Temperature": {past: 3}
  "Cooling Setpoint": {past: 1}
  "Outdoor Temperature": {past: 0, future: 12}
  "Tariff":           {past: 0, future: 48}
```

Every column the policy could read is listed: each `Y`, `U` and `D` column
of the surrogate and each reference column the objective names. `past`
counts steps before the current one; `future` counts steps after it, and only
a forecast has one. No `future` may exceed `train_pred_steps`, since nothing
past a training rollout is scored. Training windows carry the rows a
preview reads past the rollout's end, so a file needs those rows too. A `D` or reference column requires it, and a `Y` column
cannot take it. A `U` column's window ends at the previous step, the last
command applied. `past: 0` there means the policy does not read its own
commands; with `past: 1` it sees the move a `first_difference` row is pricing.
The deepest `past`, plus the current step, is the policy's history; it sets
how far apart training windows start and, with the surrogate's
`history_steps`, how many recorded steps seed each rollout. The reportcard
lists the table and the input width it gives.

## Grid response: energy, cost and peak

Energy, cost and peak are three reductions of one signal, `power`, which
the `power_metric` hook computes: a mean, a tariff-weighted mean, and a
power mean that approaches the maximum. Because all three read the same
hook on the same steps, they cannot end up pricing different quantities.
`power_metric` can sum metered channels the surrogate predicts well, or be
a physics proxy that grows with the demand on the equipment. What the
policy learns from is its gradient, and its scale is what has cost
campaigns, so read the objective balance before training: printed in full at
`progress: live` or `log`, and as its warnings alone at `summary`.

Shifting load in time takes more than a cost term. The worked case is
pre-cooling: drive the zone down through cheap hours, then coast through
the expensive ones on the building's own mass. It appears only when all of
these hold:

- **The policy sees the expensive period coming.** The tariff's `future`
  in `policy_window` covers the run-up to it. A policy that sees only the current price can
  react, not shift.
- **There is somewhere to store it.** A narrow band leaves no room, and
  then "no price response" is a measurement of the band.
- **One training rollout holds the action and its payoff**, the dip and the
  coast (`train_pred_steps`).
- **The power side is a cost term, not an energy term.** Energy rewards
  using less at every hour, and its cheapest answer is to stop: load
  shedding, with comfort going backwards.

What it looks like when it works: below the rule through the cheap hours,
then at the band edge, not past it, through the expensive ones.

**Confirm that the policy reads the price and not the clock.** A tariff
that repeats daily can be learned as time of day, which the disturbances
already carry. Train on a real price record sliced at random offsets
(`make_references.py` does this by default). Then evaluate against a test
reference with the peak moved (see below) and check that the load moves
with it, and compare against the same config with the cost gain at 0.

## Train on several draws of the price

A policy trained against one tariff learns that tariff's hours as much as
its prices. Listing the plant file several times against several reference
files (the tariff shifted earlier and later, its contrast changed) makes it
read the price instead, and it then holds the band better on a tariff it has
not seen. `make_references.py` writes draws from a price history; any
transformation of your own reference file works the same way.

## Bounds: where each command may go

By default each control commands anywhere inside `actuation_bounds`, the
data's range. `command_bounds` narrows a control to a range that moves with
a forecast at each step, `{column: [low, high]}`, each side a number in the
column's units or a reference or disturbance column the policy reads:

```yaml
command_bounds:                     # syntax, not values
  "Cooling Setpoint": ["Comfort Low", ["Comfort High", -0.5]]
  "Heating Setpoint": [15.0, "Comfort Low"]
```

A side `[column, offset]` is that column plus a fixed offset in the command's
units. Equipment that starts only once the state is past its setpoint (a
latching thermostat) overshoots a setpoint held at the band edge, so keep the
command a margin inside it.

The policy's whole output range is mapped onto `[low, high]`, so every
command in it is reachable and none outside it is. Bounding cooling inside
the band and heating under its floor also keeps the two from crossing.

## Residual control: a correction to a trusted rule

With `residual_delta`, an anchored control commands
`reference_control(...) + δ`, with `δ` kept inside a per-column box, rather
than commanding a setpoint outright. The rule does the routine work and
keeps the band, the policy adjusts it for whatever else the objective asks,
and the box limits what a surrogate that is wrong somewhere can cost you.
Training starts exactly at the rule. A box of `[0, 0]` locks a control to
the rule, and every box `[0, 0]` scores the rule itself on your surrogate,
which is the run to read every later one against.

Use it for a small correction to a rule you trust. It is not the way to
make a policy behave on a model that is wrong: a box narrow enough to make
that safe is too narrow to reach what pays (a night pre-cool from a setback
rule is several degrees away). Fix the model instead; `nt guide check` is the
test. A column is either anchored or in `command_bounds`, not both.

Nothing is rounded to an actuator's resolution; that is the building
management system's job. Smoothness comes from a `first_difference` row.

## What every run reports

In `control_reportcard.md` and in MLflow. These are instruments, read
against the rule or a previous run.

**Before training:**

- **Objective balance**: each term's share of the objective on one real
  batch, with a warning when one term outweighs the rest combined. It is
  taken at the starting point, the rule for a residual policy, so it says a
  term is present, not that it is strong enough. The same print after
  training says more.
- **Power term shape**: the power signal swept across each actuation box,
  so its size sits next to the gains you are choosing.
- **Power clamped to zero**: the share of steps where power is exactly 0
  and gives no gradient. Those are often the cheap hours that load should
  move into.
- **Objective notes**: legal configurations that cannot do what they were
  switched on for: a weight column the policy does not preview, and power
  terms that are all peak.

**After the closed-loop evaluation on the test split:**

- **Actuation**: mean step change per control, and time spent at each
  bound.
- **Band-jump response**: for every step change in a band that left an
  output out of band, whether a control moved, whether the output
  followed, and whether it got back in. Pooled violation rates can rank
  controllers backwards on exactly this.
- **Residual saturation**: for each anchored control, how much of the time
  `δ` sat at each edge of its box. A box edge that binds most of the time
  means the rule is wrong for this plant, or the optimizer is riding a
  surrogate error into the wall.

[template/README.md](../template/README.md) describes the run-by-run ladder
(the rule, the band terms only, the full objective, its price-blind twin),
and how to read the plots.

## From Python

```python
from neurotemplate.common.config import load_config
from neurotemplate.control.config import ControlConfig
from neurotemplate.control.load import load_trained_policy
from neurotemplate.control.train import train_control

cfg = load_config("control_configs/control.yaml", ControlConfig,
                  {"sysid_run_id": model_run_id})
run = train_control(cfg)              # run.run_id, run.results (the test closed loop)
policy_node, cl_system, cfg, sysid_cfg, nm_mapping, norm_stats = (
    load_trained_policy(run.run_id))
```

As in modeling, `train_control` sets torch's thread count from `threads` and
the default device for the process, and `load_trained_policy` takes the same
`device=` as `load_trained_dynamics`: this machine's, never the run's.

To evaluate a trained policy against another reference without retraining,
for example the same week with the price peak moved:

```python
from neurotemplate.control.pipeline import evaluate_closed_loop
from neurotemplate.control.scenarios import load_control_splits

_, cl_system, cfg, sysid_cfg, nm_mapping, norm_stats = load_trained_policy(run_id)
cfg.test_reference_path = ["data/test_reference_shifted.csv"]
_, _, test_dfs = load_control_splits(cfg, nm_mapping, norm_stats)
results, _ = evaluate_closed_loop(cfg, sysid_cfg, cl_system, test_dfs,
                                  nm_mapping, norm_stats)
results.energy.weighted_energy["Tariff"]   # one entry per weight a power term names
```
