"""One-thread CPU cost of develop's closed-loop training step on a ported project."""
import cProfile, os, pstats, sys, tempfile, time
os.environ["OMP_NUM_THREADS"] = "1"
project, db = sys.argv[1], sys.argv[2]
os.chdir(project); sys.path.insert(0, project)
import mlflow, torch
from neurotemplate.common.config import load_config
from neurotemplate.control import train as T
from neurotemplate.control.config import ControlConfig
torch.set_num_threads(1)

import inspect
from neurotemplate.modeling.config import ModelingConfig
from neurotemplate.modeling.train import train_model
from neurotemplate.modeling import experiment as E
from neurotemplate.modeling.pipeline import prepare_data
from neurotemplate.hooks import load_modeling_hooks
mover = {"location": f"sqlite:///{db}", "progress": "silent", "plot": False}
def model_side():
    # the model side's own training step
    m = load_config("model_configs/port.yaml", ModelingConfig, mover)
    m.validate(); m.device = "cpu"; m.seed = E.set_seeds(1)
    data = prepare_data(m, load_modeling_hooks(m))
    want = inspect.signature(E.build_trainer).parameters
    tr = E.build_trainer(m, **{k: v for k, v in vars(data).items() if k in want})
    tb = next(iter(tr.train_data)); mB = tb["yn"].shape[0] if "yn" in tb else len(next(v for v in tb.values() if hasattr(v, "shape")))
    def mstep():
        o = tr.model(dict(tb)); tr.optimizer.zero_grad(); o[f"{tb['name']}_loss"].backward()
        torch.nn.utils.clip_grad_norm_(tr.model.parameters(), 100.0); tr.optimizer.step()
    mstep(); n = 0; c0 = time.process_time()
    while time.process_time() - c0 < 10 or n < 2: mstep(); n += 1
    c = (time.process_time() - c0) / n
    print(f"MODEL {m.model}: windows={len(tr.train_data.dataset)} batches/epoch={len(tr.train_data)} B={mB} N={m.horizon_steps} hist={m.history_steps} params={sum(p.numel() for p in tr.model.parameters() if p.requires_grad)} cpu_ms/step={c*1e3:.0f} us/window-step={c/(mB*m.horizon_steps)*1e6:.2f} steps_timed={n}", flush=True)
    
try:
    pass
except Exception as e:
    print("MODEL side skipped:", str(e)[:120], flush=True)

def build(over):
    cfg = load_config("control_configs/port.yaml", ControlConfig,
                      {"location": f"sqlite:///{db}", "progress": "silent", "plot": False, **over})
    cfg.validate(); cfg.device = "cpu"; cfg.seed = T.set_seeds(1)
    mlflow.set_tracking_uri(cfg.location)
    cfg.sysid_run_id = T.latest_model_run()
    dyn_sys, dyn_node, scfg, mapping, stats = T.load_frozen_dynamics(cfg.sysid_run_id, cfg.run_dir, cfg.weights, device="cpu")
    cfg.steps_per_hour = scfg.steps_per_hour
    obj = T.resolve_objective(cfg, mapping)
    cfg.fill_window_table(obj.regulated, mapping, obj.r_columns)
    obj = T.resolve_objective(cfg, mapping)
    tr, de, te = T.load_control_splits(cfg, mapping)
    cfg.fill_common_windows(T.shortest_test_rollout(cfg, scfg, te)); cfg.check_windows()
    T.fill_reference_scale(obj, cfg, tr)
    norm = T.extend_norm_stats(obj, cfg, stats)
    tl, dl = T.build_loaders(cfg, scfg, mapping, tr, de)
    pol = T.build_controller(cfg, scfg, mapping, norm, dyn_node)
    cl = T.build_closed_loop(dyn_node, pol, cfg, scfg)
    loss = T.build_control_loss(cfg, scfg, mapping)
    return cfg, scfg, tl, T.build_problem(cl, loss).to("cpu"), pol

def step(problem, opt, batch):
    out = problem(batch)
    opt.zero_grad(); out[f"{batch['name']}_loss"].backward()
    torch.nn.utils.clip_grad_norm_(problem.parameters(), 100.0); opt.step()

for label, over in (("mlp 3x64", {"policy": "mlp", "log2_policy_hidden": 6, "policy_layers": 3}),):
    cfg, scfg, tl, problem, pol = build(over)
    params = sum(p.numel() for p in problem.parameters() if p.requires_grad)
    opt = torch.optim.Adam([p for p in problem.parameters() if p.requires_grad], lr=1e-3)
    batch = next(iter(tl)); B = batch["yn"].shape[0]; N = cfg.horizon_steps
    step(problem, opt, batch)
    n = 0; c0 = time.process_time(); w0 = time.time()
    while time.process_time() - c0 < 12 or n < 2:
        step(problem, opt, batch); n += 1
    cpu = (time.process_time() - c0) / n; wall = (time.time() - w0) / n
    print(f"{label}: windows={len(tl.dataset)} batches/epoch={len(tl)} B={B} N={N} hmax={batch['yn'].shape[1]} "
          f"policy_params={params} cpu_ms/step={cpu*1e3:.0f} wall_ms/step={wall*1e3:.0f} us/window-step={cpu/(B*N)*1e6:.2f} steps_timed={n}", flush=True)
    if label == "as configured":
        pr = cProfile.Profile(); pr.enable()
        for _ in range(2): step(problem, opt, batch)
        pr.disable()
        st = pstats.Stats(pr); total = st.total_tt
        rows = sorted(((v[2], v[3], k) for k, v in st.stats.items()), reverse=True)[:14]
        print(f"  profile total {total:.2f}s; top self-time:")
        for tt, ct, k in rows:
            print(f"    {tt/total*100:5.1f}%  {os.path.basename(k[0])}:{k[1]} {k[2][:60]}")
