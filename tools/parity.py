"""Same-seed fingerprint of every kind of run, in one process per run.
    parity.py run <nt-binary> <out.json>     parity.py diff a.json b.json
Each arm: every metric at every step, every checkpoint tensor, the results json."""
import hashlib, json, os, shutil, sqlite3, subprocess, sys, tempfile
from pathlib import Path

def sh(nt, *args, cwd):
    r = subprocess.run([nt, *args], cwd=cwd, capture_output=True, text=True, env={**os.environ, "OMP_NUM_THREADS": "1"})
    if r.returncode:
        raise RuntimeError(f"{args}: {r.stdout[-1500:]}\n{r.stderr[-1500:]}")

def fingerprint(project):
    import torch
    db = sqlite3.connect(Path(project) / "mlflow.db")
    out = {}
    runs = db.execute("select run_uuid, artifact_uri, name from runs order by start_time").fetchall()
    for i, (uuid, uri, name) in enumerate(runs):
        rows = db.execute("select key, step, value from metrics where run_uuid=? and key not like '%time%' order by key, step", (uuid,)).fetchall()
        ck = {}
        art = Path(uri.replace("file://", ""))
        for pth in sorted(art.glob("*state_dict*.pth")):
            sd = torch.load(pth, map_location="cpu", weights_only=False)
            ck[pth.name] = {k: hashlib.sha1(v.numpy().tobytes()).hexdigest()[:12] for k, v in sd.items()}
        out[f"{i}:{name}"] = {"metrics": [[k, s, v] for k, s, v in rows], "checkpoint": ck}
    return out

def write_roles_project(nt, d, model, extra_model=(), control=True, ctl=()):
    from neurotemplate.example import SPLITS, thermal_plant, ZONE, HEAT, POWER, SETPOINT, OUTDOOR, SOLAR, BAND_LOW, BAND_HIGH
    import yaml
    sh(nt, "init", cwd=d)
    for split, days, seed, start in SPLITS:
        thermal_plant(days, seed, start).to_csv(Path(d) / "data" / f"{split}.csv")
    data = yaml.safe_load((Path(d) / "data.yaml").read_text())
    data["columns"] = {"Y": [ZONE, HEAT, POWER], "U": [SETPOINT], "D": [OUTDOOR, SOLAR]}
    (Path(d) / "data.yaml").write_text(yaml.safe_dump(data, allow_unicode=True))
    return yaml, ZONE, BAND_LOW, BAND_HIGH

QUICK = ["--set", "plot=false", "--set", "seed=3", "--set", "progress=silent"]

def arm_example(nt, d, policy):
    sh(nt, "example", "thermal", "project", cwd=d); p = Path(d) / "project"
    sh(nt, "model", "train", "--set", "epochs=3", *QUICK, cwd=p)
    if policy == "mpc":
        import yaml
        f = p / "control_configs" / "control.yaml"; c = yaml.safe_load(f.read_text())
        for k, v in c["policy_window"].items():
            if "future" in v: v["future"] = "12h"
        c["solver"] = {"max_iters": 5}; c["test_stride"] = "24h"
        f.write_text(yaml.safe_dump(c, allow_unicode=True))
    sh(nt, "control", "train", "--set", "epochs=3", "--set", f"policy={policy}", *QUICK, cwd=p)
    return p

def arm_roles(nt, d, model, *more):
    yaml, ZONE, LOW, HIGH = write_roles_project(nt, d, model)
    sh(nt, "model", "train", "--set", "epochs=4", "--set", f"model={model}", *more, *QUICK, cwd=d)
    return d

def arm_roles_control(nt, d, model, policy):
    yaml, ZONE, LOW, HIGH = write_roles_project(nt, d, model)
    sh(nt, "model", "train", "--set", "epochs=3", "--set", f"model={model}", *QUICK, cwd=d)
    cfgp = Path(d) / "control_configs"
    objective = {"losses": {"low": {"column": LOW, "target": ZONE, "map": "relu", "norm": 1},
                            "high": {"column": HIGH, "target": ZONE, "map": "nrelu", "norm": 1}},
                 "gains": {"low": 1.0, "high": 1.0}}
    f = cfgp / "control.yaml"; c = yaml.safe_load(f.read_text()) or {}; c.update(objective); f.write_text(yaml.safe_dump(c, allow_unicode=True))
    sh(nt, "control", "train", "--set", "epochs=3", "--set", f"policy={policy}", *QUICK, cwd=d)
    return d

ARMS = {
    "example_linear": lambda nt, d: arm_example(nt, d, "linear"),
    "example_mlp": lambda nt, d: arm_example(nt, d, "mlp"),
    "example_mpc": lambda nt, d: arm_example(nt, d, "mpc"),
    "mlp": lambda nt, d: arm_roles(nt, d, "mlp"),
    "nssm": lambda nt, d: arm_roles(nt, d, "nssm"),
    "lssm": lambda nt, d: arm_roles(nt, d, "lssm"),
    "gru": lambda nt, d: arm_roles(nt, d, "gru"),
    "nssm_ensemble": lambda nt, d: arm_roles(nt, d, "nssm", "--set", "ensemble_seeds=2"),
    "nssm_control_mlp": lambda nt, d: arm_roles_control(nt, d, "nssm", "mlp"),
    "gru_control_linear": lambda nt, d: arm_roles_control(nt, d, "gru", "linear"),
}

if sys.argv[1] == "run":
    nt, outp = sys.argv[2], sys.argv[3]; only = sys.argv[4:] or list(ARMS); res = {}
    for name in only:
        d = tempfile.mkdtemp()
        try:
            p = ARMS[name](nt, d); res[name] = fingerprint(p); print("ok", name, flush=True)
        except Exception as e:
            res[name] = {"error": str(e)[-800:]}; print("FAIL", name, str(e)[-800:], flush=True)
        shutil.rmtree(d, ignore_errors=True)
    Path(outp).write_text(json.dumps(res))
else:
    a, b = (json.loads(Path(x).read_text()) for x in sys.argv[2:4]); bad = 0
    for arm in sorted(set(a) | set(b)):
        x, y = a.get(arm), b.get(arm)
        if x == y and x is not None and "error" not in x:
            print("same", arm, sum(len(r["metrics"]) for r in x.values()), "metric rows"); continue
        bad += 1; print("DIFF", arm)
        if not x or not y or "error" in x or "error" in y: print("  ", (x or {}).get("error"), (y or {}).get("error")); continue
        for (rx, vx), (ry, vy) in zip(x.items(), y.items()):
            if vx["checkpoint"] != vy["checkpoint"]: print("   checkpoint differs", rx)
            mx = {(k, s): v for k, s, v in vx["metrics"]}; my = {(k, s): v for k, s, v in vy["metrics"]}
            keys = set(mx) ^ set(my); d = [(k, mx[k], my[k]) for k in set(mx) & set(my) if mx[k] != my[k]]
            print("   ", rx, "keys only one side", sorted(keys)[:6], "differ", len(d), sorted(d, key=str)[:3])
    print("bad arms:", bad); sys.exit(bool(bad))
