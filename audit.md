# Codebase Audit

Issues grouped by severity. All findings are against the CLAUDE.md rules (KISS → DRY → Readable → Maintainable).

---

## KISS / Clarity Issues

### `modeling/experiment.py` — Direct mutation of `Trainer` internals
The curriculum loop resets `trainer.badcount`, `trainer.best_devloss`, and `trainer.current_epoch` directly. These are implementation details of Neuromancer's `LitTrainer`, not a documented API. If Neuromancer changes its internals, this silently breaks. Added a comment surfacing the fragility (2026-09-09); the open decision is whether it's worth upstreaming a public `reset()` on `LitTrainer` versus continuing to accept the risk here.

### `modeling/plotting.py` — Logger duck-typing, partly resolved (2026-09-10)
Audited every `hasattr(logger, ...)` site against the actual call graph instead of assuming they were all disguised `None` checks. Two different things were living under one label:

- **Dead conditions** (logger is provably never `None` or never anything but a full `MLFlowLogger` at that point): `control/cli/train.py:203,216,223`, `modeling/cli/train.py:143`, `modeling/cli/optuna.py:247`, `modeling/pipeline.py:385` (`log_test_results`), `control/pipeline.py:289` (`log_control_results`) — the `hasattr` was checking a method (`log_dict`/`log_artifacts`/`run`) that every real logger and every test double in this repo already implements. Removed; calls are now unconditional (or just `if cfg.verbose:` where that was the real gate).
- **A real, load-bearing check** on `logger._mlflow` / a `None` guard, at `modeling/plotting.py:65-70`, `modeling/pipeline.py:278`, and `common/reportcard.py:42`: `logger` here can genuinely be `None` (`nt model load` never builds one) *and*, separately, tests pass a lightweight duck-typed stand-in (`tests/test_ensemble.py::_StubLogger` and siblings in `test_control_diagnostics_e2e.py`/`test_control_metrics.py`/`test_modeling_diagnostics_e2e.py`) that implements the public `log_metrics`/`log_dict`/`log_artifacts` surface but not the private `_mlflow` handle or `.run` — confirmed by `test_ensemble.py` actually failing (`AttributeError: '_StubLogger' object has no attribute '_mlflow'`) when the `hasattr(logger, "_mlflow")` guard was first dropped. Restored as `logger is not None and hasattr(logger, "_mlflow")` at all three sites; simplified `plotting.py`'s three-way fallback (`_mlflow` attr → `log_artifact` method → no-op) down to just the `_mlflow` branch, since the `log_artifact`-method branch matched no logger type that exists anywhere in this codebase.

Net: the `hasattr` pattern wasn't uniformly one smell — most of it was, one real case wasn't. Still open: `_mlflow` is a private attribute of Neuromancer's `MLFlowLogger`, and the only reason test code needs to be a partial duck type at all is that `MLFlowLogger` has no public single-artifact `log_artifact`/`log_text` method. A small local wrapper (or a `Protocol`) that gives `MLFlowLogger` and its test doubles the same public surface would remove the need for `hasattr` here entirely — that's a real interface-design decision, not a one-line fix.

### `modeling/experiment.py:207` — Unconditional `child_logger._mlflow.log_param(...)`
No guard at all, unlike the sites above — but `child_logger` is only ever constructed when `global_logger is not None` (`experiment.py:249`), and `create_child_logger` always builds a real `MLFlowLogger`, never a test double, so this one is already correct as-is.

### `modeling/experiment.py:288` and `control/pipeline.py:130` — Duplicated `Trainer.best_devloss` extraction
Both sites do the same `best = trainer.best_devloss; if hasattr(best, "item"): best = best.item()` against the same undocumented Neuromancer internal that the curriculum-reset item above already flags. Same underlying fragility, two copies of the workaround — a candidate to fold into one helper if/when the Trainer-internals question above gets resolved.

---

## CLI / config decisions

### `pyproject.toml:51-52` — Local editable override of the pinned `neuromancer` git dependency
`[tool.uv.sources]` points `neuromancer` at `../../neuromancer` (editable), which silently wins over the pinned git commit in `[project.dependencies]` for anyone whose checkout happens to have that sibling directory — i.e. the maintainer's machine today. `uv sync` from a fresh clone elsewhere (or CI) will fail with a missing-path error, not a helpful one. The comment already says "Comment this out to exercise the real git pin locally," so the fix is known; the decision is *when* — leaving it enabled is a live convenience for active neuromancer co-development, at the cost of onboarding breaking for the next person to `uv sync` this repo as-is.

### `modeling/cli/load.py:119-155` (`_load_dyn_sys`) — Four-way checkpoint-format fallback
State-dict path (nested `try: load_state_dict / except RuntimeError:` key-remap retry) → `hasattr(state, "nodes")` → `hasattr(state, "state_dict")` → `raise ValueError`. Plausibly justified by genuinely different historical save formats, but it's the "try this, then this, then that" shape CLAUDE.md Rule 5 calls out — worth checking whether every branch is still reachable, or whether normalizing the format at save time would let load time stop guessing.

### `hooks.py` — Path-based hook loading has no cache
`load_modeling_hooks`/`load_control_hooks` re-run `importlib.util.spec_from_file_location` + `exec_module` on every call when `cfg.user_hooks_path` is an explicit file path (the dotted-module-name branch is fine — `importlib.import_module` caches). A single training run calls this well over ten times (policy build, loss build, objective pass, evaluate_closed_loop, per-loader `comfort_band_signal`, ...), so a path-based hooks file is fully re-parsed and re-executed each time, including any top-level side effects in the user's file. Needs a caching design (by resolved path? by path+mtime, so edits are picked up?), not just a quick patch.

### `common/data_utils.py:19-27` (`get_data`) — `verbose` parameter is documented as unused
The docstring says it straight: "Reserved for caller-level logging control (currently unused here)." `cfg.verbose=True` has no effect on this path even though it's threaded through `_load_split`/`get_data_splits` as if it did. Either wire it to the one real candidate (the frequency-inference warning) or drop the parameter — carrying a "reserved for later" argument is the speculative-extensibility YAGNI Rule 4 warns about.

### `template/control_configs/{control,debug}.yaml` — `energy_weight: 0.0` pinned against the stated "commented out by default" convention
The template's own convention is "every field is shown at its library default, commented out." `control/config.py:136` defaults `energy_weight` to `0.1`, but both template YAMLs explicitly uncomment and zero it — because the template's stub `power_metric` sets `POWER_METRIC_IMPLEMENTED = False`, and a nonzero `energy_weight` with no power metric implemented is a fail-fast error at `control/closed_loop.py:94`. It's explained in-file, but is the one field a user could delete "to match the pattern" and immediately break. Decide whether the library default should be `0.0` (a hook-gated feature probably shouldn't default on) rather than leaning on template comments to keep it off.

### `tests/test_run_summary.py:100-106` — Golden-file snapshot self-heals instead of failing when missing
`if os.environ.get("REGEN_GOLDEN") or not golden_path.exists(): golden_path.write_text(text); return` means a golden file that's missing (deleted, bad checkout, etc.) silently regenerates and passes rather than failing loud — a mild Rule 5 violation (fail-fast) for what is otherwise a deliberate, well-labeled convention. Decide whether a missing golden file should instead be `pytest.fail("golden file missing — rerun with REGEN_GOLDEN=1 if intentional")`.

### Tests — five near-duplicate `ModelingConfig` builders (low priority)
`tests/test_dummy_dynamics.py:26`, `test_gru_dynamics.py:45`, `test_loss.py:20`, `test_model_whitelist.py:29`, `test_rc_dynamics.py:65` each hand-roll a small `make_cfg()` that differs only in 2-3 fields — the same shape `tests/control_fixtures.py` already centralizes as `make_sysid_cfg`/`make_control_cfg` for the control side. CLAUDE.md explicitly tolerates repeating "a tiny bit of code" when abstraction would obscure intent, so this isn't a violation as-is; noting it only because a shared `modeling_fixtures.py::base_cfg(model, **overrides)` would be a natural, low-risk extraction the next time one of these five is touched.

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
