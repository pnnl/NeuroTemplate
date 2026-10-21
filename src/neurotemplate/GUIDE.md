# NeuroTemplate: the guided flow

`nt guide` prints this overview; `nt guide <step>` prints one step. Each step is a
few decisions, each with the command whose output settles it. Read a step when you
reach it. The reference docs (README, MODELING, CONTROL, EXPORT) and the hook
docstrings go deeper; open them when a step points you there, not before.

| step | decides | ends with |
|---|---|---|
| `nt guide setup` | the project, the column roles, the data | a smoke run that loads your data |
| `nt guide model` | the model's structure and priors | a reportcard whose physics checks pass |
| `nt guide check` | whether the model is fit to plan against | the reference rule scored on the model next to the data |
| `nt guide control` | the objective, what the policy sees, its bounds | a policy that beats the rule on the model, read from the plots |
| `nt guide export` | the package that runs without the library | a verified export and a passing check |

Three things decide most outcomes, in this order:

1. **The model must say what the actuator does where the policy will take it**, and
   what that costs. A policy exploits every error in the direction that pays.
2. **The objective must contain the trade you want** (comfort against cost, say),
   with the information to make it (a long enough preview of what is coming).
3. **Read every result against something**: the rule, the previous run, the data.
   A number alone says little.

Every training run ends by printing its run id and its reportcard's path. Read the
reportcard before deciding the next run.

## setup

1. **Project.** `cp -r <template> <project>` and work there; `nt` finds the hooks
   (`modeling_user_defined.py`, `control_user_defined.py`) and the config
   directories in the directory it runs from.
2. **Column roles**, in `modeling_user_defined.py`:
   - `Y`: what the model predicts. The states a controller acts on, plus any meter
     the objective prices.
   - `U`: what the controller writes.
   - `D`: what is known without the plant: weather, schedules, anything a forecast
     can supply at run time. Nothing forecasts a meter, so a meter is never `D`.
3. **Data paths** in `model_configs/*.yaml`, then
   `nt model train --config model_configs/debug.yaml`. It takes seconds and catches
   every wrong name.
4. **Coverage.** The model only knows commands the data issued, and the policy's
   bounds default to that range. If the controller must go somewhere the data never
   went, the model cannot tell it what happens there.

## model

Decide the structure in `block_mapping`; it matters more than any hyperparameter.
Every key below is in MODELING.md ("Blocks", "Declaring what a column is").

1. **Family.** `2r2c` for a temperature you will plan against: a grey-box thermal
   network that is stable under any input, with storage you can read in hours. The
   black boxes (`mlp`, `nssm`, `gru`, `lssm`) fit freely; for planning, give them sign
   priors (step 5).
2. **What each input is.** In the block's `inputs`:
   - `demand` for a setpoint: the model sees the thermostat's error, not the raw
     setpoint.
   - `linear` for a delivered power or irradiance.
   - `ambient` for the outdoor temperature the zone conducts to.
3. **Meters the objective prices** go in their own block (`model: mlp`, `reads` the
   zone). For a meter to respond to the actuator, it must see the actuator. A U
   column that no block claims is shared, and a declared `demand` reaches every
   block that sees it. So list the setpoint the meter draws from in no block's `U`,
   declare it in the zone block's `inputs`, and claim every other actuator in the
   block it drives (a heating setpoint in the zone block's `U`): the meter then sees
   the same demand the zone does, and nothing it does not draw from. A meter that
   sees another actuator learns the data's correlations, such as "heating on, so
   cooling off", and the policy exploits them, here by running the heating to make
   the modelled cooling power fall. A meter cannot
   read below zero, so give it a head that cannot either
   (`BoundedHead(project_physical_to_norm(0.0, stats), float("inf"))` in
   `data_head_config`): a regression head that dips negative tells the cost term
   that running the equipment earns money.
4. **When the actuator's effect depends on conditions** (equipment that delivers
   more on a cool night than on a hot afternoon), name those columns in the demand
   entry's `modulated_by`. **When the equipment latches** (it starts above the
   setpoint and, once running, runs until the zone is some way below), set
   `hysteresis: true`. To check, compare how often it runs at the same
   zone-minus-setpoint when it was running the step before and when it was not.
5. **Priors.** Time constants come from `nt data decay`. Output heads come from
   `nt data heads`. For a black-box block, a sign the physics is sure of, such as
   "more demand never draws less power", goes in `grad_path` with `grad_gain` > 0.
6. **Physics checks**, `synth_cfg`: a few lines that test the model in states the
   data never visits. A failed step response means do not plan against this model.

Reading the reportcard: the physics checks first, then the time constants and each
declared branch in physical units, then test error. Test error is measured where the
data went, not where a controller will go.

## check

Before training a policy, score the rule you would otherwise run, on the model:
`reference_control` in `control_user_defined.py`, every `residual_delta` box `[0, 0]`,
the band terms only. Compare its band violation on the model with the rule's in the
data. If the model says the rule cannot hold a band the data shows it holding, the
model is wrong about the actuators, and every policy trained on it will optimise
against that error. Fix the model, not the controller.

## control

1. **Objective**, in `control_configs/control.yaml`: the band as two terms (`relu`
   below, `nrelu` above), the cost as `power` weighted by the tariff, and a small
   `first_difference` on each command. `power` is `power_metric`, by default the sum
   of `power_channels`, the meters the model predicts. Before training, the run prints
   the share each term has of the objective. A term near 0% is not doing anything;
   size the gains from that print.
2. **What the policy sees**, `policy_window`: every column with a `past`, and for
   forecasts a `future`. To move load in time, the policy must see the expensive
   period coming. The tariff's and the band's `future` must cover the stretch from
   when the action pays off to when it is taken, and `train_pred_steps` must hold
   both the action and its payoff in one rollout (a day, for daily cycles).
3. **Bounds.** The policy writes each command inside `actuation_bounds` (by default
   the data's range). `command_bounds` narrows a command to a range that follows a
   reference at each step: a cooling setpoint inside the comfort band, a heating
   setpoint under its floor, which also keeps the two from crossing. A rule-anchored
   residual (`reference_control` plus a `residual_delta` box) is an option for a
   small, trusted correction. It is not the way to make a policy behave: a box narrow
   enough to keep a wrong model safe is too narrow to reach what pays.
4. **Read the run**, plots first. Does the action come before the expensive period,
   with the relief through it? Is a command pinned at a bound, jittering, or crossing
   another? Then the reportcard's objective balance.
5. **Its blind twin.** Train the same config with the price gain at 0. The difference
   between the two is what the price bought; a cost number alone is mostly a fact
   about the test week.

## export

`nt export policy --run_id <control run> --out <dir>/nt_package` writes the policy as
a graph that runs with numpy and onnxruntime, and checks the package against the
trained policy before writing it. EXPORT.md covers the package layout and the C and
torch targets.
