"""
`nt data heads`, `nt data ladder` and `nt data thermostat`: offline
diagnostics that read a dataset the way `nt model train` does and print a
recommendation. This file is argument parsing and printing; the statistics
are pure functions in `diagnostics`, `ladder`, `network` and `thermostat`.

None is wired into `nt model train`. `heads` prints a snippet the user pastes
into `data_head_config`; `ladder` and `thermostat` save the declarations a
thermal block's `params` takes. They need judgment the pipeline does not
have, so nothing applies them automatically.

Usage:
    nt data heads --config model_configs/modeling.yaml [--column "Total DX Power [W]"]
    nt data ladder --config model_configs/modeling.yaml [--zone COL] [--dt 30min] \
        [--candidate mine.json]
    nt data thermostat --config model_configs/modeling.yaml \
        --cooling "Cooling Setpoint [°C]" --heating "Heating Setpoint [°C]"
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import torch

from neurotemplate._typing import beartype

from neurotemplate.common.config import load_config_from_cli, MODEL_CONFIG_DIR
from neurotemplate.common.data_utils import get_data_splits, pool_dfs
from neurotemplate.hooks import load_modeling_hooks
from neurotemplate.modeling import ladder
from neurotemplate.modeling import network as nw
from neurotemplate.modeling import thermostat as th
from neurotemplate.modeling.blocks import build_blocks
from neurotemplate.modeling.config import ModelingConfig
from neurotemplate.modeling.ladder import Rung
from neurotemplate.modeling.network import Fit
from neurotemplate.modeling.dynamics import family_by_channel
from neurotemplate.modeling.pipeline import resolve_block_spec, resolve_nm_mapping
from neurotemplate.modeling.diagnostics import K_MAX_DEFAULT, recommend_head


def _print_head_recommendation(rec) -> None:
    header, *rest = rec.evidence.splitlines()
    print(f"\n{rec.column}  —  {header}")
    for line in rest:
        print(f"  {line}")
    print()
    print(f"  {rec.snippet}")


@beartype
def heads_main() -> None:
    flags = argparse.ArgumentParser(add_help=False)
    flags.add_argument("--column", type=str, default=None,
                       help="Only this Y column (default: every Y column).")
    flags.add_argument("--k-max", type=int, default=K_MAX_DEFAULT,
                       help=f"Largest K tried in the k-means sweep (default {K_MAX_DEFAULT}).")
    cfg, args = load_config_from_cli(
        sys.argv[1:], cls=ModelingConfig, config_dir=MODEL_CONFIG_DIR,
        default_config_path=MODEL_CONFIG_DIR / "modeling.yaml",
        command_flags=flags,
    )

    hooks = load_modeling_hooks(cfg)
    train_raws, _dev_raws, _test_raws, _freq_s = get_data_splits(cfg)
    nm_mapping = resolve_nm_mapping(hooks, train_raws[0], cfg)
    train_pool = pool_dfs(train_raws)

    # Which family builds which channel, resolved exactly the way the trainer
    # resolves it, so a head/family conflict is reported per channel rather
    # than from the run-wide cfg.model — which is the wrong answer for every
    # block that declares its own family.
    families = family_by_channel(
        cfg, nm_mapping, resolve_block_spec(hooks, nm_mapping, cfg),
    )

    columns = [args.column] if args.column is not None else nm_mapping["Y"]
    unknown = [c for c in columns if c not in nm_mapping["Y"]]
    if unknown:
        raise SystemExit(f"--column {unknown} not in Y = {nm_mapping['Y']}")

    for col in columns:
        rec = recommend_head(
            train_pool[col].to_numpy(), col,
            family=families.get(col), k_max=args.k_max,
        )
        _print_head_recommendation(rec)
    print()


# ── nt data ladder ──────────────────────────────────────────────

#: Rollouts scored on dev: each 24 h long, one every 6 h, after a day of filtering.
LADDER_HORIZON_H, LADDER_EVERY_H, LADDER_BURN_IN_H = 24.0, 6.0, 24.0


@dataclass
class ThermalZone:
    block: str
    zone: str
    ambient: str
    solar: Optional[str]
    drives: tuple[str, ...]     # the scaled inputs and `driven_by` heats, first
    gains: tuple[str, ...]      # the other inputs, entering through a fitted gain
    scale: dict[str, float]
    y_inputs: tuple[str, ...]   # the inputs that are Y of the model

    @classmethod
    def from_block(cls, block) -> "ThermalZone":
        problems = []
        if len(block.y_names) != 1:
            problems.append(f"Y must be the zone temperature alone; got {list(block.y_names)}")
        if block.ambient is None:
            problems.append("no `ambient`: the outdoor temperature every network conducts to")
        if problems:
            raise SystemExit(f"thermal block {block.name!r}: " + "; ".join(problems))
        inputs = [c for c in block.driven_by + block.u_names + block.d_names
                  if c not in (block.ambient, block.solar)]
        drives = tuple(c for c in inputs if c in block.driven_by or c in block.scale)
        return cls(block.name, block.y_names[0], block.ambient, block.solar, drives,
                   tuple(c for c in inputs if c not in drives), dict(block.scale),
                   tuple(block.driven_by))

    def network(self, rung: Rung):
        return rung.network(self.drives, self.solar, self.gains, self.scale)

    def series(self, raw, network, every: int, dt_h: float) -> nw.Series:
        """One file's data for `network`, on a grid of `every` rows."""
        return ladder.zone_series(raw, self.zone, self.ambient, ladder.network_columns(network),
                                  self.y_inputs, every, dt_h)


def _warn_signed_gains(zones: list[ThermalZone], train_raws) -> None:
    """A gain input enters as gain · value with gain ≥ 0, so a column that
    goes negative cannot push both ways and fits at zero. Solar is left out:
    irradiance reads a little below zero at night, which is sensor noise."""
    columns = sorted({c for z in zones for c in z.gains + z.drives if c not in z.y_inputs})
    signed = [c for c in columns if min(float(raw[c].min()) for raw in train_raws) < 0]
    if signed:
        print(f"WARNING: input(s) {signed} take negative values. Every gain is >= 0, so a "
              "signed feature (sin/cos of the hour, a centred flag) enters at zero. Use "
              "non-negative indicators or schedules, e.g. a 0/1 flag per part of the day.")


def _parse_dt_steps(text: str, freq_s: float) -> int:
    seconds = pd.Timedelta(text).total_seconds()
    every = seconds / freq_s
    if seconds <= 0 or abs(every - round(every)) > 1e-9:
        raise SystemExit(f"--dt {text} is not a whole number of the data's "
                         f"{freq_s:g} s steps.")
    return int(round(every))


@beartype
def ladder_main() -> None:
    flags = argparse.ArgumentParser(add_help=False)
    flags.add_argument("--zone", type=str, default=None,
                       help="Only the thermal block whose zone is this column.")
    flags.add_argument("--dt", type=str, default="30min",
                       help="Step of the likelihood fits, e.g. 30min (the default) or 15min.")
    flags.add_argument("--candidate", type=str, action="append", default=[],
                       help="A JSON file holding a network declaration of your own, fitted "
                            "and compared beside the ladder's rungs (repeatable).")
    cfg, args = load_config_from_cli(
        sys.argv[1:], cls=ModelingConfig, config_dir=MODEL_CONFIG_DIR,
        default_config_path=MODEL_CONFIG_DIR / "modeling.yaml",
        command_flags=flags,
    )
    hooks = load_modeling_hooks(cfg)
    train_raws, dev_raws, _test_raws, freq_s = get_data_splits(cfg)
    nm_mapping = resolve_nm_mapping(hooks, train_raws[0], cfg)
    blocks = build_blocks(nm_mapping, resolve_block_spec(hooks, nm_mapping, cfg))
    zones = [ThermalZone.from_block(b) for b in blocks if b.scale is not None]
    if args.zone is not None:
        zones = [z for z in zones if z.zone == args.zone]
    if not zones:
        raise SystemExit(
            "No thermal block to fit. A thermal block declares its zone temperature as Y, "
            "`ambient` (and `solar`, if any) among its D, and `scale`, what sets its units; "
            "see `nt guide thermal-zone`.")
    _warn_signed_gains(zones, train_raws)

    every = _parse_dt_steps(args.dt, freq_s)
    coarse_h, fine_h = every * freq_s / 3600.0, freq_s / 3600.0
    candidates = {Path(path).stem: json.loads(Path(path).read_text())
                  for path in args.candidate}

    cache: dict[tuple[str, str], Fit] = {}

    def fit_zone(zone: ThermalZone, network, start) -> Fit:
        key = (zone.block, network.name)
        if key not in cache:
            data = [zone.series(raw, network, every, coarse_h) for raw in train_raws]
            cache[key] = nw.fit(network, data, start)
        return cache[key]

    def fit_rung_for(group: list[ThermalZone]):
        def fit_rung(rung: Rung, parents: Optional[list[Fit]]):
            return [fit_zone(z, z.network(rung), p.values if p else None)
                    for z, p in zip(group, parents or [None] * len(group))]
        return fit_rung

    out_dir = Path("ladder")
    out_dir.mkdir(exist_ok=True)
    picks = {}
    for zone in zones:
        selection = ladder.forward_select(fit_rung_for([zone]), solar=zone.solar is not None)
        picks[zone.block] = selection.chosen
        candidate_fits = {}
        for name, decl in candidates.items():
            network, boxes, noise = nw.parse_declaration(decl, name)
            candidate_fits[name] = fit_zone(
                zone, network, {**nw.box_centres(boxes), **noise})
        _print_ladder_card(zone, selection, candidate_fits, dev_raws, every, coarse_h,
                           fine_h, len(train_raws), out_dir)

    if len(zones) > 1:
        shared = ladder.forward_select(fit_rung_for(zones),
                                       solar=all(z.solar is not None for z in zones),
                                       bic=True)
        print(f"\nShared across {len(zones)} zones: {shared.chosen}  "
              f"(likelihoods summed, and the summed BIC must fall too; per zone: "
              + ", ".join(f"{b} {r}" for b, r in picks.items()) + ")")
        for note in _selection_notes(shared):
            print(f"  {note}")
        for zone, fit in zip(zones, shared.fits[shared.chosen]):
            if picks[zone.block] != shared.chosen:
                path = out_dir / f"{zone.block}.shared.json"
                path.write_text(json.dumps(nw.declaration(fit), indent=2, ensure_ascii=False))
                print(f"  {zone.block} at {shared.chosen}: {path}")


def _print_ladder_card(zone: ThermalZone, selection, candidate_fits: dict, dev_raws,
                       every: int, coarse_h: float, fine_h: float, n_files: int,
                       out_dir: Path) -> None:
    fits = {name: fs[0] for name, fs in selection.fits.items()}
    fits.update(candidate_fits)
    n_obs = next(iter(fits.values())).n_obs
    print(f"\n{zone.zone}   coarse dt {coarse_h * 60:g} min, {n_obs:,} train steps"
          + (f" in {n_files} files" if n_files > 1 else ""))
    print(f"  {'rung':<14}{'states':>6}{'params':>7}{'log L':>10}{'p vs parent':>13}"
          f"{'BIC':>10}{'whiteness':>11}   1 h / 24 h RMSE on dev")
    lags = ladder.whiteness_lags(coarse_h)
    for name, fit in fits.items():
        parent = selection.parents.get(name)
        p = ("—" if parent is None else
             _p_text(ladder.lr_p_value(fits[parent].log_likelihood, fits[parent].n_params,
                                       fit.log_likelihood, fit.n_params)))
        white = _p_text(ladder.ljung_box(fit.innovations, lags))
        scores = [ladder.rollout_rmse(
            fit.network, fit.values, zone.series(raw, fit.network, 1, fine_h),
            horizon=round(LADDER_HORIZON_H / fine_h), burn_in=round(LADDER_BURN_IN_H / fine_h),
            every=round(LADDER_EVERY_H / fine_h)) for raw in dev_raws]
        scores = [s for s in scores if s is not None]
        rmse = (f"{sum(s[0] for s in scores) / len(scores):.3f} / "
                f"{sum(s[1] for s in scores) / len(scores):.3f}" if scores else "dev too short")
        label = name + ("  (yours)" if name in candidate_fits else "")
        print(f"  {label:<14}{len(fit.network.nodes):>6}{fit.n_params:>7}"
              f"{fit.log_likelihood:>10.1f}{p:>13}{fit.bic:>10.1f}{white:>11}   {rmse}")
    chosen = fits[selection.chosen]
    taus = ", ".join(f"{t:.3g} h ± {s:.2g}" for t, s in zip(chosen.taus_h, chosen.taus_se_h))
    print(f"  → {selection.chosen}   time constants {taus}")
    by_bic = min(fits, key=lambda n: fits[n].bic)
    if by_bic != selection.chosen:
        print(f"  BIC prefers {by_bic}; the likelihood-ratio tests stop at {selection.chosen}.")
    for note in _selection_notes(selection) + _fit_notes(chosen):
        print(f"  {note}")
    if ladder.ljung_box(chosen.innovations, lags) < 0.05:
        print("  Innovations not white: time structure the network leaves out. On "
              "closed-loop data that is often the controller answering the zone's own "
              "error, or a daily driver no input carries; add the latter as a "
              "non-negative D column (an occupancy or schedule indicator) and rerun.")
    path = out_dir / f"{zone.block}.json"
    path.write_text(json.dumps(nw.declaration(chosen), indent=2, ensure_ascii=False))
    print(f"  snippet: {path}  — the block's params `network`; edit it as you like.")
    print("  p vs parent: the likelihood-ratio test against the rung it extends. "
          "Whiteness: Ljung–Box on the innovations over "
          f"{ladder.WHITENESS_HOURS:g} h; small means time structure the network "
          "leaves out.")


def _p_text(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def _selection_notes(selection) -> list[str]:
    """The extensions left out because they add a mode the data cannot place."""
    return [f"Not selected from {step.parent}: {', '.join(step.unidentified)} — each adds "
            "a mode this record does not place (an infinite error bar, or slower than "
            "the record is long)." for step in selection.steps if step.unidentified]


def _fit_notes(fit: Fit) -> list[str]:
    """The signatures of a fit that is placing noise: a mode the record does
    not place, a parameter with an error bar wider than a factor of e, and
    two that move in lockstep."""
    notes = []
    if fit.unplaced_modes:
        notes.append(
            "Not placed by this record: " + ", ".join(f"{t:.3g} h" for t in fit.unplaced_modes)
            + f" (fitted on {fit.record_h:.0f} h). A node that slow is decoupled from "
            "ambient over the record; on closed-loop data, where the HVAC holds the zone, "
            "the envelope often cannot be identified. Keep the snippet's wide box, or set "
            "the value from design data.")
    limit = nw.at_limit(fit)
    if limit:
        notes.append(f"At a limit (zero or infinity): {limit}. A gain there is left out "
                     "of the snippet; a noise there is a node the network needs no push for.")
    wide = [k for k, se in fit.log_se.items() if se > 1.0 and k not in limit]
    if wide:
        notes.append(f"Barely placed (error bar wider than ×e): {wide}.")
    names = fit.network.free_names()
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            r = float(fit.correlation[i, j])
            if abs(r) > 0.95:
                notes.append(f"{names[i]} and {names[j]} move together (r = {r:+.2f}).")
    if not fit.converged:
        notes.append("The fit stopped at its iteration cap before converging.")
    return notes


# ── nt data thermostat ──────────────────────────────────────────

def _declared_network(block) -> tuple:
    """The block's declared network, at the centre of its boxes, and the lowest
    zone capacitance its box allows."""
    decl = block.params.get("network")
    if decl is None:
        raise SystemExit(
            f"thermal block {block.name!r} declares no network. Run `nt data ladder`, and "
            "put its snippet in the block's params as `network`.")
    network, boxes, noise = nw.parse_declaration(decl, block.name)
    return network, {**nw.box_centres(boxes), **noise}, boxes[f"C_{network.zone}"][0]


@beartype
def thermostat_main() -> None:
    flags = argparse.ArgumentParser(add_help=False)
    flags.add_argument("--zone", type=str, default=None,
                       help="Only the thermal block whose zone is this column.")
    flags.add_argument("--cooling", type=str, default=None,
                       help="The cooling setpoint column (a U column of the thermostat's block).")
    flags.add_argument("--heating", type=str, default=None,
                       help="The heating setpoint column (a U column of the thermostat's block).")
    cfg, args = load_config_from_cli(
        sys.argv[1:], cls=ModelingConfig, config_dir=MODEL_CONFIG_DIR,
        default_config_path=MODEL_CONFIG_DIR / "modeling.yaml",
        command_flags=flags,
    )
    setpoints = {side: col for side, col in (("cooling", args.cooling),
                                             ("heating", args.heating)) if col}
    if not setpoints:
        raise SystemExit("Name the setpoint(s) the thermostat follows: --cooling COL "
                         "and/or --heating COL.")
    hooks = load_modeling_hooks(cfg)
    train_raws, dev_raws, _test_raws, freq_s = get_data_splits(cfg)
    nm_mapping = resolve_nm_mapping(hooks, train_raws[0], cfg)
    every_block = build_blocks(nm_mapping, resolve_block_spec(hooks, nm_mapping, cfg))
    blocks = [b for b in every_block if b.scale is not None]
    if args.zone is not None:
        blocks = [b for b in blocks if ThermalZone.from_block(b).zone == args.zone]
    if len(blocks) != 1:
        raise SystemExit(f"--zone must pick one thermal block; found "
                         f"{[b.name for b in blocks]}.")
    block = blocks[0]
    zone = ThermalZone.from_block(block)
    if len(block.driven_by) != 1:
        raise SystemExit(f"block {block.name!r} must be `driven_by` the one heat its thermostat "
                         f"produces; it has {list(block.driven_by)}. A zone whose heat is a U "
                         "has no thermostat to fit.")
    heat = block.driven_by[0]
    producer = next(b for b in every_block if heat in b.y_names)
    missing = [c for c in setpoints.values() if c not in producer.u_names]
    if missing:
        raise SystemExit(f"setpoint(s) {missing} are not U columns of block {producer.name!r}, "
                         f"which produces {heat!r}: {list(producer.u_names)}.")
    network, values, lowest_c = _declared_network(block)
    fine_h = freq_s / 3600.0

    def record(raw) -> th.Record:
        col = lambda c: torch.tensor(raw[c].to_numpy(dtype=float), dtype=nw.DTYPE)
        return th.Record(col(zone.zone), {s: col(c) for s, c in setpoints.items()},
                         col(zone.ambient), col(heat))

    sides = tuple(setpoints)
    selection = th.forward_select(sides, [record(raw) for raw in train_raws])
    scores = {name: _thermostat_scores(fit, network, values, zone, heat, setpoints, dev_raws,
                                       fine_h)
              for name, fit in selection.fits.items()}

    print(f"\n{zone.zone} ← {', '.join(f'{s} {c!r}' for s, c in setpoints.items())}"
          f"   one-step fit on {selection.fits[selection.chosen].n_obs:,} train steps; "
          f"24 h rollouts on dev through the declared network")
    print(f"  {'form':<30}{'params':>7}{'log L':>11}{'p vs parent':>13}{'BIC':>11}"
          f"   zone / heat RMSE: all, " + ", ".join(th.REGIMES))
    for name, fit in selection.fits.items():
        parent = selection.parents.get(name)
        p = ("—" if parent is None else _p_text(ladder.lr_p_value(
            selection.fits[parent].log_likelihood, selection.fits[parent].n_params,
            fit.log_likelihood, fit.n_params)))
        print(f"  {name:<30}{fit.n_params:>7}{fit.log_likelihood:>11.1f}{p:>13}"
              f"{fit.bic:>11.1f}   {_score_text(scores[name])}")
    chosen = selection.fits[selection.chosen]
    print(f"  → {selection.chosen}")
    twin = selection.fits[selection.latched]
    if twin.bic < chosen.bic:
        print(f"  BIC prefers its latched form, {selection.latched}.")
    ranked = [n for n in scores if scores[n] is not None]
    if ranked:
        best = min(ranked, key=lambda n: scores[n]["all"][0])
        # A tie is not news: only a form that rolls out clearly better.
        if scores[best]["all"][0] < 0.98 * scores[selection.chosen]["all"][0]:
            print(f"  {best} rolls out better on dev (zone RMSE "
                  f"{scores[best]['all'][0]:.3f} against {scores[selection.chosen]['all'][0]:.3f}).")
    unplaced = [k for k, se in chosen.se.items() if math.isinf(se)]
    if unplaced:
        print(f"  Not placed by this record: {unplaced}. The snippet leaves an offset or "
              "slope null and a scale's box a factor of 1000 wide: declare those from what "
              "you know of the equipment.")
    out_dir = Path("ladder")
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"{producer.name}.thermostat.json"
    path.write_text(json.dumps(th.declaration(chosen, setpoints, lowest_c / fine_h), indent=2,
                               ensure_ascii=False))
    print(f"  snippet: {path}  — block {producer.name!r}'s params `thermostat`.")
    print("  Regimes by the measured heat: satisfied under a tenth of the active side's "
          "capacity, saturated over nine tenths. The one-step fit predicts the heat over "
          "each step from the row that starts it.")


def _thermostat_scores(fit, network, values, zone: ThermalZone, heat: str,
                       setpoints: dict[str, str], dev_raws, fine_h: float) -> Optional[dict]:
    """Zone and heat RMSE of 24 h rollouts on dev, overall and by regime,
    each started from the filtered state after a day of dev; None when dev
    is too short."""
    horizon = round(LADDER_HORIZON_H / fine_h)
    burn_in, every = round(LADDER_BURN_IN_H / fine_h), round(LADDER_EVERY_H / fine_h)
    heat_column = 1 + [c for c, _ in network.inputs].index(heat)
    errors_zone, errors_heat, regime = [], [], []
    for raw in dev_raws:
        series = zone.series(raw, network, 1, fine_h)
        starts = torch.arange(burn_in, series.y.shape[0] - horizon - 1, every)
        if not len(starts):
            continue
        states = nw.filtered_states(network, values, series)
        rows = starts[None, :] + 1 + torch.arange(horizon)[:, None]
        col = lambda c: torch.tensor(raw[c].to_numpy(dtype=float), dtype=nw.DTYPE)
        outdoor = col(zone.ambient)[rows]
        measured = col(heat)[rows + 1]
        zone_path, heat_path = th.composed_rollout(
            fit, network, values, states[starts], {s: col(c)[rows] for s, c in setpoints.items()},
            outdoor, nw.drive_matrix(series)[rows], heat_column, fine_h)
        errors_zone.append((zone_path - series.y[rows]).flatten())
        errors_heat.append((heat_path - measured).flatten())
        regime.append(th.regimes(fit, measured, outdoor).flatten())
    if not errors_zone:
        return None
    ez, eh, rg = torch.cat(errors_zone), torch.cat(errors_heat), torch.cat(regime)
    rmse = lambda e: float(e.pow(2).mean().sqrt()) if len(e) else float("nan")
    out = {"all": (rmse(ez), rmse(eh))}
    for i, name in enumerate(th.REGIMES):
        out[name] = (rmse(ez[rg == i]), rmse(eh[rg == i]))
    return out


def _score_text(score: Optional[dict]) -> str:
    if score is None:
        return "dev too short"
    return ", ".join(f"{z:.2f} / {h:.1f}" for z, h in score.values())
