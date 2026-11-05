# Hooks

YAML holds settings; Python is where you change the model. All of these are optional, in
`nt_model.py` and `nt_control.py`, and a missing one that matters is named by the run.
`nt docs <hook>` prints one's signature, what it says and an example to paste.

| hook | in | what it lets you do |
|---|---|---|
| `block_mapping` | `nt_model.py` | split the model into parts: an RC zone, a thermostat, a meter |
| `custom_models` | `nt_model.py` | bring your own model |
| `output_heads` | `nt_model.py` | keep an output within limits |
| `column_norm_config` | `nt_model.py` | override a column's normalisation |
| `synth_cfg` | `nt_model.py` | the physics checks: what "at rest" means for your plant |
| `sub_plot` | `nt_model.py` | how a channel is plotted |
| `power_channels` | `nt_control.py` | which outputs meter power, what the objective prices |
| `power_metric` | `nt_control.py` | price something other than a meter: a proxy for power |
| `plot_context_channels` | `nt_control.py` | what the control plots show beside the commands |

Every hook works in physical units, the data's own. A run keeps a copy of its hooks file, and
everything that reloads the run reads the copy. `nt docs grad_path` is the sign file beside them.

## block_mapping

```python
def block_mapping(nm_mapping: dict, cfg) -> dict | None
```

Splits Y into blocks, each its own sub-model with its own parameters, family and hidden state.
Worth doing whenever the plant is several parts that interact mainly through shared inputs: a
joint model has to learn that the cross-couplings are about zero, and short records rarely teach
it. Without this hook the model is one block over every column.

```python
def block_mapping(nm_mapping, cfg):
    return {
        "hvac_a": {"Y": ["Zone A Heat [kW]"],
                   "U": ["Zone A Cool SP [degC]", "Zone A Heat SP [degC]"],
                   "D": ["Outdoor Temp [degC]"], "ambient": "Outdoor Temp [degC]",
                   "reads": ["Zone A Temp [degC]"], "model": "thermostat",
                   "params": {"thermostat": json.load(open("ladder/hvac_a.thermostat.json"))}},
        "zone_a": {"Y": ["Zone A Temp [degC]"], "U": [],
                   "D": ["Outdoor Temp [degC]", "Solar [W/m2]", "Occupied [-]"],
                   "driven_by": ["Zone A Heat [kW]"], "model": "rc_network",
                   "ambient": "Outdoor Temp [degC]", "solar": "Solar [W/m2]",
                   "scale": {"Zone A Heat [kW]": 1.0},
                   "params": {"rc_network": json.load(open("ladder/zone_a.json"))}},
        "meter":  {"Y": ["Total Power [kW]"], "U": [], "D": ["Outdoor Temp [degC]"],
                   "driven_by": ["Zone A Heat [kW]"], "model": "mlp", "predicts": "value"},
    }
```

Every Y channel is in exactly one block. A block's U are its own; a U no block claims is shared by
all. D is all of D unless a block names a subset, and blocks may overlap on it. `reads` lets a block
see another block's Y as an ordinary input, as of the row before the step; `driven_by` as of this
step, the block running after its producer. `predicts` overrides the run's: a meter says the
`value`, since as an `increment` it integrates its own errors. `model` omitted means `cfg.model`, so
a hook naming no family swaps every block from the config, and `block_models` there overrides any
one block.

`rc_network` is a thermal zone: Y the zone temperature alone, its inputs a `driven_by` column (the
heat a `thermostat` block produces), a U (a commanded heat) or a D, `ambient` the outdoor temperature
it conducts to. **`scale` is required and says what sets the units**, since temperature alone places
only gain / C: `{heat column: 1.0}` for a heat in kW, or `{"C_Ti": value}` to fit a power's gain as a
COP. `thermostat` is Y the heat, U its setpoints, `reads` the zone, `ambient` the outdoor temperature
(in its D), which some of its forms read.

`params` maps a family to its declaration, every value a scalar (pinned) or [lo, hi] (fitted inside);
only the family the block takes reads its entry. `nt data ladder` and `nt data thermostat` write them;
there is no default, because a prior about a building is yours to state and those commands are where
the data states it.

Any other block may read a setpoint through a hinge, which changes the physics rather than the
shapes, against the block's own predicted state:

```python
"zone_b": {..., "inputs": {
    "Zone B Cool SP [degC]": {"target": "Zone B Temp [degC]", "map": "nrelu"},   # relu(zone - setpoint)
    "Zone B Heat SP [degC]": {"target": "Zone B Temp [degC]", "map": "relu"},    # relu(setpoint - zone)
}}
```

A setpoint past the zone then commands nothing, as on a real thermostat. Without it a model renders a
setpoint as a smooth, always-active push, and a controller trained on it inherits that.

## custom_models

```python
def custom_models() -> dict[str, Callable[[cfg, nm_mapping, norm_stats], torch.nn.Module]]
```

Any torch module that honours the model contract gets training, the physics checks, reload, control
(a policy or MPC) and export. Set `model: <name>` in the YAML. It is one module over every column, so
`block_mapping` returns None; bound an output inside the model, or give it a `head_stack`.

```python
#   model(y, u, d) -> y_next
#     y, u, d  [batch, history_steps, n]   physical units, oldest row first;
#                                          row -1 of u and d drives this step
#     y_next   [batch, ny]
def custom_models():
    return {"my_zone": lambda cfg, nm_mapping, norm_stats: MyZone(nm_mapping)}
```

State that outlives the window goes through `rollout_state()` / `set_rollout_state(state)`, one
`[batch, width]` tensor, None when cold (seed it from the window then). Keep the model's code, and
any starting values, in this file: the run keeps a copy of it. The library's parts are yours to wire:
`modeling.thermal`, `modeling.families`, `modeling.stepper`. `nt guide own-model` has the decisions;
`nt docs modeling` the full contract.

## output_heads

```python
def output_heads(nm_mapping: dict, norm_stats: dict) -> HeadStack
```

One head per Y column, in Y order. Without this hook every channel is a `RegressionHead`, the right
default. Reach for another only when a channel really is bounded:

```python
from neurotemplate.modeling.heads import BoundedHead, HeadStack, RegressionHead

def output_heads(nm_mapping, norm_stats):
    heads = [BoundedHead(0.0, float("inf")) if "Power" in c else RegressionHead()
             for c in nm_mapping["Y"]]
    return HeadStack(heads, nm_mapping["Y"])
```

`BoundedHead(lo, hi)` stays inside [lo, hi]; pass `float("inf")` / `float("-inf")` on an open side.
Bounds are in the column's physical units. A head that lets a meter dip negative tells a cost term that
running the equipment earns money.

## column_norm_config

```python
def column_norm_config(nm_mapping: dict) -> dict[str, str | None]
```

`{column: "min_max" | "abs_max" | "z_score" | None}`. Columns not listed use the run's `norm_type`.
Override where a column's distribution defeats the default: bounded non-negative signals prefer
`min_max`, already-0/1 flags prefer None.

```python
def column_norm_config(nm_mapping):
    return {c: "min_max" for c in nm_mapping["Y"] if "Power" in c}
```

## synth_cfg

```python
def synth_cfg(nm_mapping: dict, norm_stats: dict) -> SynthSpec | None
```

The physics checks, run after training: equilibrium (states hold when nothing drives them), step
response (each actuator moves each state the declared way, net of the model's drift at rest),
reachability, and free decay (states relax toward ambient with the actuators off). They put the
surrogate in states the recorded data never visits, which is where a controller will take it, and they
are the readout most likely to tell you a surrogate is wrong for control when test error says it is fine.
Without this hook they do not run, and the run says so.

```python
from neurotemplate.modeling.scenarios import SynthSpec

def synth_cfg(nm_mapping, norm_stats):
    cool_sp, zone = "Zone Cooling Setpoint [degC]", "Zone Air Temperature [degC]"
    return SynthSpec(
        states=[zone],
        ambient="Ambient Temperature [degC]",
        off={cool_sp: norm_stats[cool_sp]["max"]},
        # Raising the cooling setpoint asks for less cooling, so the zone warms.
        # The sweep drives it down from `off` and expects a fall.
        responds={zone: {cool_sp: +1}},
    )
```

- `states`: Y channels that hold at rest and relax toward `ambient`, temperatures typically; not
  meters, since an equilibrium value written into a power channel is meaningless.
- `off`: the physical value of each U column when it asks for nothing. Not guessable (a cooling
  setpoint is off at its *maximum*, a heating setpoint at its *minimum*), so required for every U. Pick a
  value that is off across the whole `states` range. Other columns default to their recorded minimum;
  give an `off` for any column where the minimum is not "nothing", such as a signed heat.
- `responds`: `{state: {control: +1 | -1}}`, the direction each state moves as that control
  *increases*, from a state where the control is live. Required for every U. A hinge does not change
  these signs, but it changes where a sweep starts: a step is measured only from zone temperatures
  where the target calls for something, and an end of its range that never does is skipped.

All values are physical; the library validates the spec against the columns before anything runs. The
checks are instruments, not gates. A step that barely moves the state means its forcing is too weak;
an equilibrium that drifts means something is integrated that should not be, or an `off` value is not off
everywhere; a free decay far faster or slower than the plant could manage means the conduction is wrong.
A surrogate that fails step response is not one to plan against. A small drift after an honest attempt or
two is worth noting and moving past.

## sub_plot

```python
def sub_plot(df_true: pd.DataFrame, column_name: str) -> tuple[list[str], list[dict]] | None
```

What goes on a second axis under the plot of `column_name`: the covariates you want to eyeball. Return
None for a single axis, or columns and matching kwargs for `ax2.plot`:

```python
def sub_plot(df_true, column_name):
    return (["Ambient Temperature [degC]", "Heater Command [-]"],
            [{"label": "ambient", "color": "tab:blue"},
             {"label": "heater", "color": "tab:red", "linestyle": "--"}])
```

## power_channels

```python
def power_channels(nm_mapping: dict) -> list[str]
```

Y columns that meter power, in the order you want them plotted. A loss term naming `power` reduces their
sum, and the reportcard integrates it. List them by name rather than by pattern: a filter such as
`"power" in c.lower()` also sweeps in channels nobody is billed for, a second fuel under an electricity
tariff, and those then look like a cost to the objective. An objective naming `power` with neither this
hook nor `power_metric` is a loud error.

```python
def power_channels(nm_mapping):
    return ["Compressor Power [W]", "Fan Power [W]"]
```

What the policy learns from is this signal's gradient, so the meters must respond to the commands: model
them in a block that sees the actuation (`nt guide thermal-zone`, step 6).

## power_metric

```python
def power_metric(yn: Tensor, U: Tensor, D: Tensor, nm_mapping: dict) -> Tensor    # [batch, time]
```

Instantaneous power per timestep, in place of the sum of `power_channels`: what a loss term naming
`power` reduces (a mean for energy, a tariff-weighted mean for cost, a power mean for a peak), and what the
reportcard integrates. Inputs are the trajectories in physical units, `[batch, time, n]`. At each step, `U`
is the command that drove the plant into that step's `yn`. The library converts no units: return Watts and
the integral is in Wh.

Write it when nothing meters the power: a proxy from the physics of when the equipment runs, which does not
decrease as the demand on it grows and is zero when it would not be called. A `relu` there stops the gradient
on the inactive side; `soft_clamp_min` keeps the zero forward and a gradient back toward the active side.

```python
from neurotemplate.modeling.hinge import soft_clamp_min

def power_metric(yn, U, D, nm_mapping):
    zone = yn[:, :, nm_mapping["Y"].index("Zone Temp [degC]")]
    setpoint = U[:, :, nm_mapping["U"].index("Cooling SP [degC]")]
    return 500.0 * soft_clamp_min(zone - setpoint)
```

## plot_context_channels

```python
def plot_context_channels(nm_mapping: dict) -> list[str]
```

D columns to overlay on the controlled-variable axis of the closed-loop plots: the driver that makes the trace
legible, an ambient temperature, a load, irradiance.

```python
def plot_context_channels(nm_mapping):
    return ["Outdoor Temperature [degC]"]
```

## grad_path

Not a hook: the sign file `grad_gain > 0` needs, `{output column: {input column: +1 | -1}}`, the known signs of
the partial derivatives. Only list edges that hold everywhere in state space, because the penalty is evaluated
at random off-trajectory points, so a relationship that holds only while some controller is active fires in the
wrong places. A raw setpoint's partial, for one, is zero wherever the setpoint is satisfied. A hinged setpoint is
differentiated as the hinge instead, so its sign is its push on the zone (-1 for a cooling `nrelu`).

```python
import json

signs = {"Tank Temperature [degC]": {
    "Tank Temperature [degC]": +1,      # thermal mass continuity
    "Ambient Temperature [degC]": +1,   # envelope conduction
    "Heater Command [-]": +1,           # heat in raises temperature
}}
json.dump(signs, open("data/grad_signs.json", "w"), indent=2)
```
