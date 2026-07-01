"""
Synthetic Scenario Penalties
============================
Generates physically meaningful synthetic scenarios, runs them through the
Neuromancer System (same autoregressive rollout used for training), and
penalizes predictions that violate known physics.

Scenarios:
  A) Equilibrium Stability: all zones at the same temperature, HVAC off,
     outdoor temp matches indoor.
     Expected: zones remain constant.
  B) Step Response: seed with equilibrium (in_steps), then activate HVAC.
     Expected: zones move in the correct direction.
  C) Free Decay: seed with equilibrium (outdoor = indoor), then outdoor
     temp jumps to a different value at rollout start.
     Expected: zones drift toward outdoor temperature.

All scenarios share the same seeding logic: in_steps of equilibrium state
built by _build_equilibrium_inputs, then the rollout diverges by scenario.

Each class uses _build_data() to construct the full input dict and
_score_prediction() to grade the System output. compute() and evaluate()
both run through the same machinery — evaluate() additionally returns the
physical-unit predictions and the actual U/D tensors fed to the model.

Dataset-specific physical values (what "off" and "active" mean per U column,
which D column is outdoor temperature) are provided via synth_cfg,
returned by the user_defined.py hook. This makes the scenarios reusable
across datasets without modifying this file.

Each scenario class provides a compute() method returning a differentiable
scalar penalty. They are wrapped as Neuromancer Loss objects in loss.py.
"""

import torch
from typing import Optional


def build_scenarios(
    system,
    nm_mapping,
    norm_stats,
    synth_cfg,
    in_steps,
    n_scenarios,
    n_steps,
    device,
):
    """
    Build the list of synthetic-scenario penalty objects, or return None if
    the subsystem is disabled.

    Returns None when synth_cfg is None (user opted out via user_defined.synth_cfg).
    Otherwise returns a list of scenario instances ready to .compute() / .evaluate().

    This is the single construction site for the synth subsystem. All callers
    (training loss, eval, plotting) should go through it and check for None.
    """
    if synth_cfg is None:
        return None

    common = dict(
        system=system,
        nm_mapping=nm_mapping,
        norm_stats=norm_stats,
        synth_cfg=synth_cfg,
        in_steps=in_steps,
        n_scenarios=n_scenarios,
        n_steps=n_steps,
        device=device,
    )
    return [
        SyntheticEquilibriumPenalty(**common),
        SyntheticStepResponsePenalty(**common),
        SyntheticFreeDecayPenalty(**common),
    ]


def _find_outdoor_temp_col(cfg, d_names):
    """Return the D column marked 'outdoor_temp': True in cfg, or None."""
    for col in d_names:
        if col in cfg and cfg[col].get("outdoor_temp"):
            return col
    return None


def _normalize_value(physical_value, col_name, norm_stats):
    """Convert a physical value to normalized space using stored statistics."""
    if col_name not in norm_stats:
        raise KeyError(
            f"Column '{col_name}' not found in norm_stats. "
            f"Available columns: {list(norm_stats.keys())}"
        )
    stats = norm_stats[col_name]
    if stats["stretch"] == 0:
        raise ValueError(
            f"Column '{col_name}' has zero stretch in norm_stats — "
            "cannot normalize (constant column?)."
        )
    return (physical_value - stats["shift"]) / stats["stretch"]


def _build_equilibrium_inputs(T_phys, nm_mapping, norm_stats, synth_cfg, device):
    """
    Build normalized initial conditions for an equilibrium starting point.

    All zone temps = T_phys. U columns use "off" from synth_cfg (required).
    D columns use "off" from synth_cfg, except the outdoor temp column which
    is set to T_phys (matching indoor — true equilibrium).

    Returns:
        y_init: [n_scenarios, ny]
        u_init: [n_scenarios, nu]
        d_init: [n_scenarios, nd]
    """
    n = T_phys.shape[0]
    Y_names = nm_mapping["Y"]
    U_names = nm_mapping["U"]
    D_names = nm_mapping["D"]
    if _find_outdoor_temp_col(synth_cfg, D_names) is None:
        raise KeyError(
            "No D column in synth_cfg has 'outdoor_temp': True. "
            "Mark the outdoor temperature column in synth_cfg."
        )

    ny, nu, nd = len(Y_names), len(U_names), len(D_names)

    # All zone temps = T_phys
    y_init = torch.zeros(n, ny, device=device)
    for i, col in enumerate(Y_names):
        y_init[:, i] = _normalize_value(T_phys, col, norm_stats)

    # U: use "off" value from synth_cfg (required)
    u_init = torch.zeros(n, nu, device=device)
    for i, col in enumerate(U_names):
        if col not in synth_cfg:
            raise KeyError(f"U column '{col}' not found in synth_cfg.")
        if "off" not in synth_cfg[col]:
            raise KeyError(f"synth_cfg['{col}'] is missing required key 'off'.")
        u_init[:, i] = _normalize_value(
            torch.full((n,), synth_cfg[col]["off"], device=device), col, norm_stats
        )

    # D: use "off" for non-outdoor columns; outdoor temp = T_phys
    d_init = torch.zeros(n, nd, device=device)
    for i, col in enumerate(D_names):
        if col not in synth_cfg:
            raise KeyError(f"D column '{col}' not found in synth_cfg.")
        if synth_cfg[col].get("outdoor_temp"):
            d_init[:, i] = _normalize_value(T_phys, col, norm_stats)
        else:
            if "off" not in synth_cfg[col]:
                raise KeyError(f"synth_cfg['{col}'] is missing required key 'off'.")
            d_init[:, i] = _normalize_value(
                torch.full((n,), synth_cfg[col]["off"], device=device), col, norm_stats
            )

    return y_init, u_init, d_init


def _build_system_data(y_init, u_init, d_init, in_steps, n_steps, ny, device):
    """
    Construct a data dict in the format the SystemPreview expects.

    To get n_steps predictions we need Y.shape[1] = (in_steps - 1) + n_steps.
    u_init and d_init are tiled uniformly across all time steps.
    """
    n_scenarios = y_init.shape[0]
    total_time = (in_steps - 1) + n_steps

    yn = y_init.unsqueeze(1).expand(-1, in_steps, -1).contiguous()
    Y = torch.zeros(n_scenarios, total_time, ny, device=device)
    U = u_init.unsqueeze(1).expand(-1, total_time, -1).contiguous()
    D = d_init.unsqueeze(1).expand(-1, total_time, -1).contiguous()

    return {"yn": yn, "Y": Y, "U": U, "D": D}

def _get_temp_range(nm_mapping, norm_stats):
    """Get (min, max) zone temperature range across all Y columns."""
    y_cols = nm_mapping["Y"]
    t_min = min(norm_stats[col]["min"] for col in y_cols)
    t_max = max(norm_stats[col]["max"] for col in y_cols)
    return t_min, t_max


def _get_outdoor_temp_range(nm_mapping, norm_stats, synth_cfg):
    """Get (min, max) outdoor temperature range from the configured D column."""
    outdoor_col = _find_outdoor_temp_col(synth_cfg, nm_mapping["D"])
    if outdoor_col is None:
        raise KeyError(
            "No D column in synth_cfg has 'outdoor_temp': True. "
            "Mark the outdoor temperature column in synth_cfg."
        )
    return norm_stats[outdoor_col]["min"], norm_stats[outdoor_col]["max"]


def _unnormalize_y(predictions, nm_mapping, norm_stats):
    """Convert normalized zone predictions [n, steps, ny] back to physical units."""
    phys = torch.zeros_like(predictions)
    for i, col in enumerate(nm_mapping["Y"]):
        stats = norm_stats[col]
        phys[:, :, i] = predictions[:, :, i] * stats["stretch"] + stats["shift"]
    return phys


def _unnormalize_u(u_norm, nm_mapping, norm_stats):
    """Convert normalized U [n, steps, nu] back to physical units."""
    phys = torch.zeros_like(u_norm)
    for i, col in enumerate(nm_mapping["U"]):
        stats = norm_stats[col]
        phys[:, :, i] = u_norm[:, :, i] * stats["stretch"] + stats["shift"]
    return phys


def _unnormalize_d(d_norm, nm_mapping, norm_stats):
    """Convert normalized D [n, steps, nd] back to physical units."""
    phys = torch.zeros_like(d_norm)
    for i, col in enumerate(nm_mapping["D"]):
        stats = norm_stats[col]
        phys[:, :, i] = d_norm[:, :, i] * stats["stretch"] + stats["shift"]
    return phys


def _outdoor_temp_traj(data, nm_mapping, norm_stats, synth_cfg):
    """Extract the outdoor temp column from data["D"] and unnormalize it."""
    outdoor_col = _find_outdoor_temp_col(synth_cfg, nm_mapping["D"])
    outdoor_idx = nm_mapping["D"].index(outdoor_col)
    d_phys = _unnormalize_d(data["D"], nm_mapping, norm_stats)
    return d_phys[:, :, outdoor_idx]


class SyntheticEquilibriumPenalty:
    """
    Scenario A: Equilibrium Stability.

    All zones at the same temperature, HVAC off, outdoor temp matches indoor.
    Seeded with in_steps of that state, then rolled out for n_steps.
    Expected: zones remain constant.
    """

    def __init__(self, system, nm_mapping, norm_stats, synth_cfg,
                 in_steps, n_scenarios, n_steps, device):
        self.system = system
        self.nm_mapping = nm_mapping
        self.norm_stats = norm_stats
        self.synth_cfg = synth_cfg
        self.in_steps = in_steps
        self.n_scenarios = n_scenarios
        self.n_steps = n_steps
        self.device = device
        self.ny = len(nm_mapping["Y"])
        self.T_min, self.T_max = _get_temp_range(nm_mapping, norm_stats)

    def _build_data(self, n):
        T_phys = (
            torch.rand(n, device=self.device)
            * (self.T_max - self.T_min) + self.T_min
        )
        y_init, u_init, d_init = _build_equilibrium_inputs(
            T_phys, self.nm_mapping, self.norm_stats,
            self.synth_cfg, self.device
        )
        return _build_system_data(
            y_init, u_init, d_init,
            self.in_steps, self.n_steps, self.ny, self.device,
        )

    def _score_prediction(self, y_pred):
        """Per-instance MSE vs equilibrium reference. Returns [n]."""
        preds = y_pred[:, self.in_steps - 1:, :]
        return (preds[:, 1:, :] - preds[:, 0:1, :]).pow(2).mean(dim=(1, 2))

    def compute(self):
        data = self._build_data(self.n_scenarios)
        y_pred = self.system(data)["yn"]
        return self._score_prediction(y_pred).mean()

    def evaluate(self, n_scenarios: Optional[int] = None):
        """Run scenario and return physical predictions + per-instance RMSE."""
        n = n_scenarios or self.n_scenarios
        data = self._build_data(n)
        y_pred = self.system(data)["yn"]
        losses = self._score_prediction(y_pred).sqrt().tolist()

        return {
            "name": "Equilibrium",
            "losses": losses,
            "predictions": _unnormalize_y(y_pred, self.nm_mapping, self.norm_stats).detach().cpu(),
            "outdoor_temp_traj": _outdoor_temp_traj(data, self.nm_mapping, self.norm_stats, self.synth_cfg).detach().cpu(),
            "zone_names": self.nm_mapping["Y"],
            "seed_steps": self.in_steps,
        }


class SyntheticStepResponsePenalty:
    """
    Scenario B: Step Response.

    Seeded with in_steps of equilibrium, then HVAC steps to active_cooling or
    active_heating for the full n_steps rollout.
    Expected: zones move in the correct direction from step 0.

    The cooling test runs if any U column in synth_cfg has "active_cooling".
    The heating test runs if any U column has "active_heating".
    Both tests run independently; their penalties are summed.
    """

    def __init__(self, system, nm_mapping, norm_stats, synth_cfg,
                 in_steps, n_scenarios, n_steps, device):
        self.system = system
        self.nm_mapping = nm_mapping
        self.norm_stats = norm_stats
        self.synth_cfg = synth_cfg
        self.in_steps = in_steps
        self.n_scenarios = n_scenarios
        self.n_steps = n_steps
        self.device = device
        self.ny = len(nm_mapping["Y"])
        self.T_min, self.T_max = _get_temp_range(nm_mapping, norm_stats)

        # Determine which directions are active for this dataset (U columns only)
        self.has_cooling = any(
            "active_cooling" in synth_cfg.get(col, {}) for col in nm_mapping["U"]
        )
        self.has_heating = any(
            "active_heating" in synth_cfg.get(col, {}) for col in nm_mapping["U"]
        )

    def _build_u_for_mode(self, y_init, mode: str):
        """
        Build u_init for a given mode ("active_cooling" or "active_heating").
        Falls back to "off" if the column doesn't have the active mode key.
        Both "off" and the active key must be present when used; missing keys error out.
        """
        n = y_init.shape[0]
        nu = len(self.nm_mapping["U"])
        u_init = torch.zeros(n, nu, device=self.device)
        for i, col in enumerate(self.nm_mapping["U"]):
            if col not in self.synth_cfg:
                raise KeyError(f"U column '{col}' not found in synth_cfg.")
            col_cfg = self.synth_cfg[col]
            if mode in col_cfg:
                phys_val = col_cfg[mode]
            elif "off" in col_cfg:
                phys_val = col_cfg["off"]
            else:
                raise KeyError(
                    f"synth_cfg['{col}'] has neither '{mode}' nor 'off'."
                )
            u_init[:, i] = _normalize_value(
                torch.full((n,), phys_val, device=self.device), col, self.norm_stats
            )
        return u_init

    def _build_data(self, mode: str, n: int):

        # Initial temperature for zones
        T_phys = (
            torch.rand(n, device=self.device)
            * (self.T_max - self.T_min) + self.T_min
        )
        # Populate vectors with equilibrium inputs, and normalize,
        # these are for one step, no sizing for steps has happened yet
        y_init, u_off, d_init = _build_equilibrium_inputs(
            T_phys, self.nm_mapping, self.norm_stats,
            self.synth_cfg, self.device
        )
        # Create active u inputs
        u_active = self._build_u_for_mode(y_init, mode)

        """Build data dict with U split: off during history, active during rollout."""
        total_time = (self.in_steps - 1) + self.n_steps
        yn = y_init.unsqueeze(1).expand(-1, self.in_steps, -1).contiguous()
        Y = torch.zeros(n, total_time, self.ny, device=self.device)
        U = torch.cat([
            u_off.unsqueeze(1).expand(-1, self.in_steps - 1, -1),
            u_active.unsqueeze(1).expand(-1, self.n_steps, -1),
        ], dim=1).contiguous()
        D = d_init.unsqueeze(1).expand(-1, total_time, -1).contiguous()

        return {"yn": yn, "Y": Y, "U": U, "D": D}

    def _score_prediction(self, y_pred: torch.Tensor, mode: str):
        """Per-instance directional penalty. Returns [n]."""
        just_pred = y_pred[:, self.in_steps-1:, :]
        deltas = just_pred[:, 1:, :] - just_pred[:, :-1, :]
        if mode == "active_cooling":
            return torch.relu(deltas).mean(dim=(1, 2))
        else:
            return torch.relu(-deltas).mean(dim=(1, 2))

    def compute(self):
        total = torch.tensor(0.0, device=self.device)
        if self.has_cooling:
            mode = "active_cooling"
            data = self._build_data(mode, self.n_scenarios)
            y_pred = self.system(data)['yn']
            total = total + self._score_prediction(y_pred, mode).mean()
        if self.has_heating:
            mode = "active_heating"
            data = self._build_data(mode, self.n_scenarios)
            y_pred = self.system(data)['yn']
            total = total + self._score_prediction(y_pred, mode).mean()

        return total

    def evaluate(self, n_scenarios: Optional[int] = None):
        """Run scenario and return per-instance losses per mode."""
        n = n_scenarios or self.n_scenarios
        modes = {}

        for mode in ["active_cooling", "active_heating"]:
            if mode == "active_cooling" and not self.has_cooling:
                continue
            if mode == "active_heating" and not self.has_heating:
                continue

            data = self._build_data(mode, n)
            y_pred = self.system(data)["yn"]

            modes[mode] = {
                "losses": self._score_prediction(y_pred, mode).tolist(),
                "predictions": _unnormalize_y(y_pred, self.nm_mapping, self.norm_stats).detach().cpu(),
                "u_traj": _unnormalize_u(data["U"], self.nm_mapping, self.norm_stats).detach().cpu(),
                "outdoor_temp_traj": _outdoor_temp_traj(data, self.nm_mapping, self.norm_stats, self.synth_cfg).detach().cpu(),
            }

        return {
            "name": "Step Response",
            "modes": modes,
            "zone_names": self.nm_mapping["Y"],
            "u_names": self.nm_mapping["U"],
            "seed_steps": self.in_steps,
        }


class SyntheticFreeDecayPenalty:
    """
    Scenario C: Free Decay.

    Seeded with in_steps of equilibrium (outdoor temp = indoor temp).
    At rollout start, outdoor temp jumps to a different value — as if the
    building were teleported to a new climate. HVAC stays off throughout.
    Expected: zones drift toward outdoor temperature.
    """

    def __init__(self, system, nm_mapping, norm_stats, synth_cfg,
                 in_steps, n_scenarios, n_steps, device):
        self.system = system
        self.nm_mapping = nm_mapping
        self.norm_stats = norm_stats
        self.synth_cfg = synth_cfg
        self.in_steps = in_steps
        self.n_scenarios = n_scenarios
        self.n_steps = n_steps
        self.device = device
        self.ny = len(nm_mapping["Y"])

        self.outdoor_col = _find_outdoor_temp_col(synth_cfg, nm_mapping["D"])
        if self.outdoor_col is None:
            raise KeyError(
                "No D column in synth_cfg has 'outdoor_temp': True. "
                "Mark the outdoor temperature column in synth_cfg."
            )

        self.T_in_min, self.T_in_max = _get_temp_range(nm_mapping, norm_stats)
        self.T_out_min, self.T_out_max = _get_outdoor_temp_range(
            nm_mapping, norm_stats, synth_cfg
        )

    def _build_data(self, n):
        T_indoor = (
            torch.rand(n, device=self.device)
            * (self.T_in_max - self.T_in_min) + self.T_in_min
        )
        T_outdoor = (
            torch.rand(n, device=self.device)
            * (self.T_out_max - self.T_out_min) + self.T_out_min
        )

        # Ensure meaningful temperature delta (at least 2°C)
        delta = T_outdoor - T_indoor
        too_close = delta.abs() < 2.0
        T_outdoor[too_close] = T_indoor[too_close] + 2.0 * delta[too_close].sign()
        exact_zero = delta == 0
        T_outdoor[exact_zero] = T_indoor[exact_zero] + 2.0

        # Seed: equilibrium with outdoor = indoor
        y_init, u_off, d_history = _build_equilibrium_inputs(
            T_indoor, self.nm_mapping, self.norm_stats,
            self.synth_cfg, self.device
        )

        # Rollout disturbances: same as history but outdoor jumps to T_outdoor
        d_rollout = d_history.clone()
        d_idx = self.nm_mapping["D"].index(self.outdoor_col)
        d_rollout[:, d_idx] = _normalize_value(
            T_outdoor, self.outdoor_col, self.norm_stats
        )

        total_time = (self.in_steps - 1) + self.n_steps
        yn = y_init.unsqueeze(1).expand(-1, self.in_steps, -1).contiguous()
        Y = torch.zeros(n, total_time, self.ny, device=self.device)
        U = u_off.unsqueeze(1).expand(-1, total_time, -1).contiguous()
        D = torch.cat([
            d_history.unsqueeze(1).expand(-1, self.in_steps - 1, -1),
            d_rollout.unsqueeze(1).expand(-1, self.n_steps, -1),
        ], dim=1).contiguous()

        data = {"yn": yn, "Y": Y, "U": U, "D": D}
        return data, T_indoor, T_outdoor, y_init

    def _bound_loss(self, predictions, y_init, T_outdoor):
        """
        Flat-well bound penalty: \\_/ shaped.

        Zero when zone temps are between initial and outdoor temps,
        quadratic penalty outside that range (prevents overshoot).
        """
        n = T_outdoor.shape[0]
        outdoor_norm = torch.zeros(n, self.ny, device=self.device)
        for i, col in enumerate(self.nm_mapping["Y"]):
            outdoor_norm[:, i] = _normalize_value(T_outdoor, col, self.norm_stats)

        y0 = y_init.unsqueeze(1)
        y_out = outdoor_norm.unsqueeze(1)

        lower = torch.min(y0, y_out)
        upper = torch.max(y0, y_out)

        overshoot = (
            torch.relu(predictions - upper).pow(2)
            + torch.relu(lower - predictions).pow(2)
        )
        return overshoot.mean(dim=(1, 2))

    def _score_prediction(self, y_pred, T_indoor, T_outdoor, y_init):
        """Per-instance monotonicity + bound penalty. Returns [n]."""
        preds = y_pred[:, self.in_steps - 1:, :]
        deltas = preds[:, 1:, :] - preds[:, :-1, :]
        sign_direction = (T_outdoor - T_indoor).sign().unsqueeze(1).unsqueeze(2)
        monotonicity = torch.relu(-deltas * sign_direction).mean(dim=(1, 2))
        bound = self._bound_loss(preds[:, 1:, :], y_init, T_outdoor)
        return monotonicity + bound

    def compute(self):
        data, T_indoor, T_outdoor, y_init = self._build_data(self.n_scenarios)
        y_pred = self.system(data)["yn"]
        return self._score_prediction(y_pred, T_indoor, T_outdoor, y_init).mean()

    def evaluate(self, n_scenarios: Optional[int] = None):
        """Run scenario and return physical predictions + per-instance losses."""
        n = n_scenarios or self.n_scenarios
        data, T_indoor, T_outdoor, y_init = self._build_data(n)
        y_pred = self.system(data)["yn"]
        losses = self._score_prediction(y_pred, T_indoor, T_outdoor, y_init).sqrt().tolist()

        return {
            "name": "Free Decay",
            "losses": losses,
            "predictions": _unnormalize_y(y_pred, self.nm_mapping, self.norm_stats).detach().cpu(),
            "outdoor_temp_traj": _outdoor_temp_traj(data, self.nm_mapping, self.norm_stats, self.synth_cfg).detach().cpu(),
            "zone_names": self.nm_mapping["Y"],
            "seed_steps": self.in_steps,
        }
