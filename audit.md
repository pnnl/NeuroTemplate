# Codebase Audit

Issues grouped by severity. All findings are against the CLAUDE.md rules (KISS → DRY → Readable → Maintainable).

---

## KISS / Clarity Issues

### `modeling/experiment.py` — Direct mutation of `Trainer` internals
The curriculum loop resets `trainer.badcount`, `trainer.best_devloss`, and `trainer.current_epoch` directly. These are implementation details of Neuromancer's `LitTrainer`, not a documented API. If Neuromancer changes its internals, this silently breaks. A comment explaining why direct mutation is necessary (and what the correct reset boundary is) would at least surface the fragility.

### `control/config.py:108` — HVAC-era path comment
The comment on `location` still says "Default points at the sys-id store (src/mlflow.db) since control_src runs as a sibling of src/". Both the path and the comment are from the old flat-script layout and should be updated when fixing the bug above.

### ~~`closed_loop.py` — `hasattr(raw.fy, "reset")` for GRU detection~~ — resolved
This was filed as duck-typing against an undocumented internal, on the worry that a future block might grow a `reset` for unrelated reasons. Two have since grown one — `RCBlock` carries a mass temperature, `PerZoneBlock` forwards to whichever of its sub-blocks carry state — and both want exactly the behaviour the check triggers. "A block that can be reset is a block whose state must not leak across a sequence boundary" turned out to be the real contract rather than a proxy for one, and `make_reset_init` now says so. A separate `is_stateful` flag would be a second thing to keep in sync with the first.

### `modeling/plotting.py` — Logger duck-typing across three interfaces
`log_artifact` tries `logger._mlflow`, then a `log_artifact` method, then falls through. Three sequential attribute checks to find one working interface is the "try this, if not then this" smell from CLAUDE.md Rule 5. The logger contract should be narrowed or the callers should pass a concrete type.
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
