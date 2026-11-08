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
against each other on the same surrogate, and judge the winner on the plant.
A user who wants to compare to a rule can do that themselves.

## What you provide

An objective, in `control_configs/control.yaml`. That and the column roles are all a run needs:
the latest model run is its surrogate, and everything else is a default it prints. `nt docs config`
lists every setting; `nt docs hooks` the hooks, all optional, of which `power_channels` is the one
a `power` term needs. Column roles come from the surrogate, and the record from `data.yaml`.

```bash
nt control train --config control_configs/debug.yaml
nt control train --set sysid_run_id=<another model run>
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

**Reference columns are columns of the record no role claims.** A name that is neither
a surrogate column nor `power`, such as a band, a schedule or a tariff, is read from the
record. The objective just names it. The policy reads it if its window lists it; the surrogate
never does. Dividing it for the policy is by its target's stats if it bounds a Y column, and
otherwise by `reference_scale`, set from its largest magnitude and printed unless you set it.

## What the policy reads: `policy_window`

```yaml
policy_window:                       # syntax, not values
  "Zone Temperature": {past: 15m}
  "Cooling Setpoint": {past: 5m}
  "Outdoor Temperature": {past: 0, future: 1h}
  "Tariff":           {past: 0, future: 4h, stride: 30m}
```

The policy reads what this lists, for any role, and nothing else: each run prints what it leaves
unread, and the exported package takes only what is listed, so nobody has to rebuild a column it
does not use. Unset, it reads the regulated outputs and the `D` columns 3 steps back, the last
command, and each `D` and reference column a horizon ahead. `past` is how far before the current
step; `future` how far after it, and only a forecast has one. No `future` may exceed `horizon`,
since nothing past a training rollout is scored. A `D` or reference column requires one, and a `Y`
column cannot take one. A forecast's `stride` thins its future to every so many steps: the tariff
above is read now and at 8 points ahead, rather than at all 48 rows. Unset, every row is read.
Every step's preview reads real rows of the record, to the rollout's last
step, so a file needs those rows. A `U` column's window ends at the previous step, the last command
applied: `past: 0` there means the policy does not read it; with `past: 5m` it sees the move a
`first_difference` row is pricing. The deepest `past`, with the surrogate's history, sets how many
recorded steps seed each rollout. Training windows start at any row of the records, drawn afresh at
every step, and dev windows `dev_stride` apart, a whole horizon by default. The reportcard lists the
table and the input width it gives.

Beside what it lists, the network reads one more input: the steps its rollout has left. In training
it counts down over the horizon, so the policy learns what the end of a rollout does to the
objective. On dev, on test and when exported it stays a whole horizon, as for a controller that
decides again at every step; the exported package takes no input for it.

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
  coast (`horizon`).
- **The power side is a cost term, not an energy term.** Energy rewards
  using less at every hour, and its cheapest answer is to stop: load
  shedding, with comfort going backwards.

What it looks like when it works: below where the data sat through the cheap hours,
then at the band edge, not past it, through the expensive ones.

**Confirm that the policy reads the price and not the clock.** A tariff
that repeats daily can be learned as time of day, which the disturbances
already carry. Train on draws of a real price record sliced at random offsets. Then
evaluate against a test file with the peak moved (see below) and check that the load moves
with it, and compare against the same config with the cost gain at 0.

## Train on several draws of the price

A policy trained against one tariff learns that tariff's hours as much as its prices. `draws` lists
files against the same record, each with another band or tariff, which makes it read the price instead,
and it then holds the band better on a tariff it has not seen:

```yaml
draws: [data/tariff_later.csv, data/tariff_earlier.csv]
```

Each file replaces the columns it shares with the record, matched by timestamp, and adds a pass over the
training windows. With several records, a draw applies to the one its timestamps match, and matching none
is a loud error. Dev and test take no draws; to score a held-out tariff, point `test` at a file holding it.

## Bounds: where each command may go

`control_bounds` is the one mechanism. A column not listed takes its recorded range. A listed one is
`{column: [low, high]}`, each side a number in the column's units, a reference column the policy reads, or
another command:

```yaml
control_bounds:                     # syntax, not values
  "Cooling Setpoint": ["Comfort Low", ["Comfort High", -0.5]]
  "Heating Setpoint": [15.0, "Comfort Low"]
```

A side `[column, offset]` is that column plus a signed offset in the command's
units: `["Comfort High", -0.5]` is half a degree under the ceiling,
`["Comfort Low", -1.0]` a degree under the floor. Equipment that starts only once the state is past its
setpoint (a latching thermostat) overshoots a setpoint held at the band edge, so keep the
command a margin inside it. A numeric side also sets what "at a bound" means on the reportcard; a side
naming a column is read at each step, so the policy's whole output range is mapped onto `[low, high]`, every
command in it reachable and none outside it. Bounding cooling inside the band and heating under its floor
keeps the two from crossing; where no band separates them, a side naming the other command orders them directly:

```yaml
control_bounds:
  "Heating Setpoint": [15.0, 22.0]
  "Cooling Setpoint": [["Heating Setpoint", 1.0], 30.0]   # at least 1 above heating
```

Bounds that read other commands are mapped after them, in whatever order
they are declared; bounds that read each other in a cycle are refused. A column a bound reads must be
in `policy_window`, so the deployed policy is handed it.

Nothing is rounded to an actuator's resolution; that is the building
management system's job. Smoothness comes from a `first_difference` row.

## MPC: planning through the surrogate at every step

`policy: mpc` trains nothing, a one-line toggle. At every step it plans the next `horizon` of
commands through its own copy of the surrogate, inside `control_bounds`, against the same objective,
with Adam from its previous plan, and commands the first. Its forecast is every D and reference
column over that horizon. DPC is at best MPC; a policy's purpose is to be faster, easier and more
flexible, and MPC stays as a comparator and as a tool for those who want it.

Each command is held as its share of its range, in [0, 1]; Adam steps the share by about the `solver`'s
`lr` of the range and clamps it back into the box. A window's solve stops once
`patience` steps pass without its objective improving by more than `rtol` of it. It keeps the best plan
it saw. `max_iters` caps it, so a solve always ends, and ends the same way on any machine. The reportcard
gives the steps the test windows' solves took and how often one hit the cap; a cap hit means a window was
still improving. A deadline belongs to deployment: the exported `Policy.act(..., time_s=)` stops planning
at it.

A trained policy is one forward pass per step and MPC is a solve, so the choice is compute against what
`nt control gap` says a policy leaves on the table. Both are scored the same way, and both export (MPC with
`--format torch`, see [EXPORT.md](EXPORT.md)).

## What every run reports

`control_reportcard.md` opens with a verdict: comfort, the band-jump response, and a warning
for each check below that fired.
The rest reports exceptions rather than inventories; each instrument's full
output is a JSON file in the run's `results/` folder, and the headline
numbers are MLflow metrics. The surrogate is described on its own run's
reportcard. These are instruments, read against a previous run or the best plan.

**Before training:**

- **Command sensitivity**: per command, the share of recorded windows where
  the surrogate's next outputs do not move with it at all (an exact zero
  gradient: a saturated clamp, a dead unit, a command the model ignores). A
  surrogate with no gradient path to its commands at all (numpy, `.detach()`,
  an integer op inside it) is refused: nothing can be trained or planned
  through it.
- **Objective balance**: each term's share of the objective on one real
  batch, with a warning when one term outweighs the rest combined. It is
  taken at the starting point, so it says a term is present, not that it is
  strong enough. The same print after training says more.
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

**On demand, `nt control gap --run_id <run>`:** the policy beside the best open-loop plan on the same
test windows, term by term, written to the run's `results/control_gap.md`. The plan is a free command per
step through the same surrogate, bounds and objective, optimised with every step ahead known, so nothing
commanding through those bounds does better on this surrogate. The gap between them is the controller's:
its inputs, structure or training. What the plan does not do either (a pre-cool, riding the band edge) the
surrogate or the objective does not reward, and no controller will find it. It cannot see where the surrogate
is wrong about the plant. It is solved as MPC's plans are (the `solver` block) from `--starts` starts, the
controller's own commands and random ones, each window keeping its best; the report says how far the starts
landed apart and whether every window stopped before the cap.

`nt guide control` is the run-by-run ladder (the band terms only, the full objective, its price-blind twin)
and how to read the plots.

Each optional capability is off until you ask for it:

| Want | Implement | Then declare |
|---|---|---|
| an energy objective, and the energy section of the reportcard | `power_channels` (or a proxy in `power_metric`) | a `{column: power, norm: 1}` row |
| price-driven load shifting | `power_channels`, and a tariff column in the record | a `{column: power, norm: 1, weight: <tariff>}` row, a `future` on the tariff in `policy_window` |
| peak shaving | `power_channels` | a `{column: power, norm: 4}` row |

## From Python

```python
from neurotemplate.common.config import load_config
from neurotemplate.control.config import ControlConfig
from neurotemplate.control.load import load_trained_policy
from neurotemplate.control.train import train_control

cfg = load_config("control_configs/control.yaml", ControlConfig)
run = train_control(cfg)              # run.run_id, run.results (the test closed loop)
policy_node, cl_system, cfg, sysid_cfg, nm_mapping, norm_stats = (
    load_trained_policy(run.run_id))
```

As in modeling, `train_control` sets torch's thread count from `threads` and
the default device for the process, and `load_trained_policy` takes the same
`device=` as `load_trained_dynamics`: this machine's, never the run's.

To evaluate a trained policy against another record without retraining,
for example the same week with the price peak moved, point its `data_file` at a
`data.yaml` whose `test` holds it:

```python
from neurotemplate.control.pipeline import evaluate_closed_loop
from neurotemplate.control.windows import load_control_splits

_, cl_system, cfg, sysid_cfg, nm_mapping, norm_stats = load_trained_policy(run_id)
cfg.data_file = "data_shifted.yaml"
_, _, test_dfs = load_control_splits(cfg, nm_mapping)
results, _ = evaluate_closed_loop(cfg, sysid_cfg, cl_system, test_dfs,
                                  nm_mapping, norm_stats)
results.energy.weighted_energy["Tariff"]   # one entry per weight a power term names
```
