"""One-thread CPU cost of a closed-loop training step on a project. Temporary bench.
    bench_step.py <project dir> <control config> [--profile] [key=value ...]
Builds the run as `train_control` does up to its first batch, then times `fit`'s step on
that batch: forward, backward, clip, Adam. Prints ms per step and us per window-step."""
import cProfile, os, pstats, sys, time
os.environ["OMP_NUM_THREADS"] = "1"
project, config = sys.argv[1], sys.argv[2]
profile = "--profile" in sys.argv
overrides = dict(a.split("=", 1) for a in sys.argv[3:] if "=" in a)
os.chdir(project); sys.path.insert(0, project)
import mlflow, torch
torch.set_num_threads(1)
from neurotemplate.common.config import load_config, _parse_value
from neurotemplate.control import train as T
from neurotemplate.control.config import ControlConfig
from neurotemplate.control.windows import fill_start_box

cfg = load_config(config, ControlConfig, {"progress": "silent", "plot": False,
                                          **{k: _parse_value(v) for k, v in overrides.items()}})
cfg.validate(); cfg.device = "cpu"; cfg.seed = T.set_seeds(1)
mlflow.set_tracking_uri(cfg.location)
cfg.sysid_run_id = cfg.sysid_run_id or T.latest_model_run()
dyn_sys, dyn_node, scfg, mapping, stats = T.load_frozen_dynamics(cfg.sysid_run_id, cfg.run_dir,
                                                                  cfg.weights, device="cpu")
cfg.steps_per_hour = scfg.steps_per_hour
objective = T.resolve_objective(cfg, mapping)
cfg.fill_window_table(objective.regulated, mapping, objective.r_columns)
objective = T.resolve_objective(cfg, mapping)
train, dev, test = T.load_control_splits(cfg, mapping)
cfg.fill_common_windows(T.shortest_test_rollout(cfg, scfg, test)); cfg.check_windows()
T.fill_reference_scale(objective, cfg, train)
fill_start_box(cfg, dyn_sys, mapping, train)
norm = T.extend_norm_stats(objective, cfg, stats)
batches, _ = T.build_loaders(cfg, scfg, mapping, train, dev)
policy = T.build_controller(cfg, scfg, mapping, norm, dyn_node)
problem = T.build_problem(T.build_closed_loop(dyn_node, policy, cfg, scfg),
                          T.build_control_loss(cfg, scfg, mapping))
optimizer = torch.optim.Adam([p for p in problem.parameters() if p.requires_grad], lr=1e-3)
batch = next(batches)
B, N = batch["yn"].shape[0], cfg.horizon_steps


def step():
    problem.train()
    loss = problem(batch)["train_loss"]
    optimizer.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(problem.parameters(), 1.0); optimizer.step()


step()
n, c0 = 0, time.process_time()
while time.process_time() - c0 < 10 or n < 2:
    step(); n += 1
cpu = (time.process_time() - c0) / n
params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
print(f"B={B} N={N} policy={cfg.policy} params={params} cpu_ms/step={cpu * 1e3:.0f} "
      f"us/window-step={cpu / (B * N) * 1e6:.2f} steps_timed={n}", flush=True)
if profile:
    pr = cProfile.Profile(); pr.enable()
    for _ in range(2):
        step()
    pr.disable()
    stats = pstats.Stats(pr); total = stats.total_tt
    rows = sorted(((v[2], k) for k, v in stats.stats.items()), reverse=True)[:18]
    print(f"profile {total:.2f}s; top self-time:")
    for tt, k in rows:
        print(f"  {tt / total * 100:5.1f}%  {os.path.basename(k[0])}:{k[1]} {k[2][:50]}")
