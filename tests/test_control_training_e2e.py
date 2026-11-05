"""One real optimisation run per feature, through the trainer the CLI uses.

The unit tests check each piece; this checks that the pieces compose under
`train_policy` — a real optimizer, a real dev loader, real checkpointing —
because a term that builds and a term that trains are different claims.
"""

import pytest
import torch

from neurotemplate.common.config import _parse_value, dict_to_dataclass
from neurotemplate.control.closed_loop import (
    build_closed_loop, build_control_loss, build_problem,
)
from neurotemplate.control.config import ControlConfig
from neurotemplate.control.losses import extend_norm_stats, resolve_objective
from neurotemplate.control.pipeline import (
    evaluate_closed_loop, train_policy,
)
from neurotemplate.control.policy import build_policy
from neurotemplate.control.windows import (
    build_loaders, horizons, load_control_splits, scored_steps,
)
from control_fixtures import (
    BAND_GAINS, BAND_LOSSES, HIGH, LOW, POWER_LOSSES, SETPT, TARIFF, TARIFF_SCALE,
    full_setup, make_norm_stats, make_reference_df,
    write_splits,
)

# The file-backed tests reuse one plant file for every split.
pytestmark = pytest.mark.filterwarnings("ignore:.*timestamps from training")


def _train(s, dfs=None):
    cfg, sysid_cfg, nm, ns = s["cfg"], s["sysid_cfg"], s["nm"], s["norm_stats"]
    dfs = dfs or [s["frame"]]
    policy = build_policy(cfg, sysid_cfg, nm, ns)
    cl = build_closed_loop(s["dyn_node"], policy, cfg, sysid_cfg)
    problem = build_problem(cl, build_control_loss(cfg, sysid_cfg, nm))
    train_loader, dev_loader = build_loaders(cfg, sysid_cfg, nm, dfs, dfs)
    trained, best_dev, _ = train_policy(cfg, problem, train_loader, dev_loader)
    return problem, trained, best_dev


def test_a_grid_response_objective_trains():
    s = full_setup(epochs=3, losses={**BAND_LOSSES, **POWER_LOSSES},
                   gains={**BAND_GAINS, "energy": 0.1, "cost": 0.1, "peak": 0.1},
                   reference_scale={TARIFF: TARIFF_SCALE})
    _, trained, best_dev = _train(s)

    assert torch.isfinite(torch.tensor(best_dev))
    results, out = evaluate_closed_loop(
        s["cfg"], s["sysid_cfg"], trained, [s["frame"]], s["nm"],
        s["norm_stats"],
    )
    assert TARIFF in results.energy.weighted_energy
    assert torch.isfinite(out["U"]).all()


def test_a_bounded_policy_over_a_reference_file_trains_and_reloads(tmp_path):
    """The band arrives as a reference file, the bounds read its columns by
    name, and the run rebuilds from its logged config to the same policy."""
    s = full_setup(epochs=3, control_bounds={SETPT: [LOW, HIGH]})
    cfg, sysid_cfg, nm, ns = s["cfg"], s["sysid_cfg"], s["nm"], s["norm_stats"]
    write_splits(cfg, tmp_path, s["plant"], [make_reference_df(s["plant"])])
    train_dfs, _, test_dfs = load_control_splits(cfg, nm)
    _, _, hmax = horizons(cfg, sysid_cfg)
    problem, trained, best_dev = _train(s, train_dfs)

    assert torch.isfinite(torch.tensor(best_dev))
    results, out = evaluate_closed_loop(cfg, sysid_cfg, trained, test_dfs, nm, ns)
    # The commanded setpoint never leaves the band after training.
    r = resolve_objective(cfg, nm).r_columns
    u = out["U"][:, scored_steps("U", hmax), 0]
    low = out["R"][:, hmax - 1:hmax - 1 + u.shape[1], r.index(LOW)]
    high = out["R"][:, hmax - 1:hmax - 1 + u.shape[1], r.index(HIGH)]
    assert (u >= low - 1e-4).all() and (u <= high + 1e-4).all()

    # Reload as `load_trained_policy` does: the config from its logged params,
    # the reference stats rebuilt from it, the checkpoint loaded strictly.
    logged = {k: _parse_value(str(v)) for k, v in cfg.asdict().items()
              if k != "device"}
    again = dict_to_dataclass(logged, ControlConfig)
    again.device = cfg.device
    again.validate()
    ns_again = extend_norm_stats(resolve_objective(again, nm), again,
                                 make_norm_stats(s["plant"]))
    assert ns_again == ns
    policy = build_policy(again, sysid_cfg, nm, ns_again)
    cl = build_closed_loop(s["dyn_node"], policy, again, sysid_cfg)
    rebuilt = build_problem(cl, build_control_loss(again, sysid_cfg, nm))
    rebuilt.load_state_dict(problem.state_dict())
    _, out_again = evaluate_closed_loop(again, sysid_cfg, cl, test_dfs, nm, ns_again)
    assert torch.equal(out_again["U"], out["U"])
