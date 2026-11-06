"""One-thread CPU cost of the paper's building iteration, same machine."""
import os, sys, time
os.environ["OMP_NUM_THREADS"] = "1"
os.chdir(os.path.expanduser("~/papers/amortizedWeightsDPC"))
import torch
from amortdpc.train import load_config, make, objective, compiled_step
torch.set_num_threads(1)
cfg = load_config("configs/buildings/cond2d.yaml")
torch.manual_seed(0); g = torch.Generator().manual_seed(0)
case, ws, pol = make(cfg); t = cfg["train"]; N, B = t["N"], t["batch"]
ws.running.fill_(1.0)
params = sum(p.numel() for p in pol.parameters())
opt = torch.optim.Adam(pol.parameters(), lr=1e-3)
for label, step in (("eager", None), ("compiled", compiled_step(case, pol))):
    def it():
        sc = case.sample(B, N, "train", g); kv = ws.sample(B, g, None)
        loss, *_ = objective(case, ws, pol, sc, kv, N, update=True, step=step)
        opt.zero_grad(); loss.mean().backward(); torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0); opt.step()
    it(); it()
    n = 0; c0 = time.process_time(); w0 = time.time()
    while time.process_time() - c0 < 15: it(); n += 1
    cpu = (time.process_time() - c0) / n
    print(f"paper {label}: B={B} N={N} params={params} cpu_ms/it={cpu*1e3:.0f} wall_ms/it={(time.time()-w0)/n*1e3:.0f} us/window-step={cpu/(B*N)*1e6:.2f} its={n}", flush=True)
