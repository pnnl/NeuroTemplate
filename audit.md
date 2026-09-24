# Codebase Audit

Open issues against the CLAUDE.md rules (KISS → DRY → Readable → Maintainable),
and known limits of the modeling problem. Last swept 2026-09-22 on the `tidy`
branch, which resolved the checkpoint-format fallbacks in `nt model load`, the
uncached and module-shadowing hook loader, the duplicated `best_devloss`
extraction, the silent curriculum reset, and the unused `get_data(verbose=)`.

---

## Open decisions

### `_mlflow`, a private attribute of Neuromancer's `MLFlowLogger`
`modeling/plotting.py`, `modeling/pipeline.py` and `common/reportcard.py` log
single artifacts through `logger._mlflow`, guarded by
`logger is not None and hasattr(logger, "_mlflow")`. The guard is load-bearing:
`nt model load` passes no logger, and the test doubles in `tests/` implement
only the public `log_metrics` / `log_dict` / `log_artifacts` surface.
`MLFlowLogger` has no public single-artifact method. A small local wrapper (or a
`Protocol`) giving the real logger and its doubles the same public surface
would remove the `hasattr`s — an interface decision, not a one-line fix.
`common/tracking.py` and `modeling/experiment.py` also call `_mlflow` directly,
on loggers that are always real.

### `energy_weight` defaults to `0.1` although it needs a hook
`ControlConfig.energy_weight` defaults on, but a positive weight fails at problem
build unless `control_user_defined.POWER_METRIC_IMPLEMENTED` is true. Both
template YAMLs pin it to `0.0` with a comment. Defaulting it to `0.0` would
match the other hook-gated weights (`cost_weight`, `peak_weight`) — but an
app that implements `power_metric` and relies on the default would then
silently lose its energy term, so it is a decision rather than a fix.

### Tests — near-duplicate `ModelingConfig` builders (low priority)
`tests/test_gru_dynamics.py`, `test_loss.py` and `test_rc_dynamics.py` each
hand-roll a small `make_cfg()`. A shared `modeling_fixtures.py::base_cfg(model,
**overrides)` would be a low-risk extraction the next time one of them is
touched.

---

## Known limits, documented rather than open

Neither of these is a defect to fix; both are properties of the modeling
problem, recorded here so they are not rediscovered as bugs.

### `2r2c` cannot represent a thermostat deadband
The learned heat input reads `u` and `d` and never the state, which is what
makes the rollout unconditionally stable. "The compressor is off whenever the
setpoint is above the zone" is a function of the *difference*, so the family
renders staged cooling as a smooth monotone response to setpoint. Letting the
gain see the state would buy the deadband and cost the guarantee. See
`modeling/rc.py`.

### `2r2c` time constants are not identifiable from a short rollout
The mass temperature is unobserved and has to be seeded from an `in_steps`
window; the seed is wrong, and shortening the slow mode is a cheaper way to
absorb that than fixing the forcing. Measured on synthetic data from a known
31.6 h building: seeded truthfully, training moves `tau_slow` 13.4 → 16.5 h;
seeded as a real rollout must, the same training drives it to 8.7 h. Hence
`rc_train_taus: false` by default and the three `rc_tau_*_h` config fields.
