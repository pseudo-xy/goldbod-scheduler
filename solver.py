"""
GoldBod Stochastic DP Solver — Vectorized NumPy Implementation

Key vectorization: for each period t, evaluate all combinations of
(inventory_state, purchase_x, release_y) simultaneously using
broadcasting, instead of three nested Python loops.

Speedup vs. loop version: ~30-80x on typical parameters.
"""

import numpy as np
import pandas as pd
import time
from dataclasses import dataclass, field
from typing import Dict, Any


# ─────────────────────────────────────────────────────────────
# Parameter container
# ─────────────────────────────────────────────────────────────

@dataclass
class DPParams:
    # Horizon
    T: int = 52
    S0: float = 0.0

    # Conversion constants
    C_TROY: float = 32.1507
    K_CONV: float = 4.186

    # Gold price forecast (quarterly, USD/oz)
    jp_quarterly: tuple = (4440.0, 4655.0, 4860.0, 5055.0)

    # FX distribution
    fx_low: float = 10.50
    fx_high: float = 12.50
    fx_threshold: float = 11.00

    # ASM availability
    v_min: float = 2000.0
    v_max: float = 3000.0

    # Supply
    rho: float = 0.99242
    r_min: float = 0.0
    s_bar: float = 3000.0

    # Annual target
    q_annual: float = 127_000.0

    # Objective weights
    alpha: float = 3.861
    beta: float = 1.650
    gamma: float = 0.164
    phi: float = 5.0

    # Stochastic setup
    num_scenarios: int = 20
    seed: int = 42

    # DP grid
    inv_grid_points: int = 300
    x_grid_points: int = 30
    y_grid_points: int = 30


# ─────────────────────────────────────────────────────────────
# Weekly gold price interpolation
# ─────────────────────────────────────────────────────────────

def build_weekly_gold(jp_quarterly, T=52):
    """Linearly interpolate quarterly USD/oz forecasts to weekly series."""
    g = []
    for q in range(4):
        start = jp_quarterly[q]
        end = jp_quarterly[q + 1] if q < 3 else jp_quarterly[q]
        for w in range(13):
            g.append(start + (end - start) * w / 13)
    return np.array(g[:T])


# ─────────────────────────────────────────────────────────────
# Scenario generation
# ─────────────────────────────────────────────────────────────

def generate_scenarios(params: DPParams, gold_weekly: np.ndarray):
    rng = np.random.default_rng(params.seed)
    n = params.num_scenarios
    T = params.T

    fx = rng.uniform(params.fx_low, params.fx_high, (n, T))
    vol = rng.uniform(params.v_min, params.v_max, (n, T))

    # Buy cost per kg: (gold * fx * C_TROY) / K_CONV
    cost = gold_weekly[None, :] * fx * params.C_TROY / params.K_CONV
    inflow = cost * vol
    target = params.rho * inflow
    probs = np.ones(n) / n

    return cost, vol, fx, inflow, target, probs


# ─────────────────────────────────────────────────────────────
# Vectorized DP solver (single scenario)
# ─────────────────────────────────────────────────────────────

def run_single_dp_vectorized(
    cost_xi, vol_xi, target_xi,
    lambda_xi, lambda_mean_xi, p_sell_xi,
    params: DPParams
):
    """
    Vectorized backward DP.

    For each period t, we build a 3-D tensor of costs indexed by
    (s_prev_idx, x_idx, y_idx). We broadcast-evaluate the Bellman
    update across all (x, y) combinations for all inventory states
    simultaneously, then argmin over the (x, y) axes.
    """
    T = params.T
    S_BAR = params.s_bar
    ALPHA = params.alpha
    BETA = params.beta
    GAMMA = params.gamma
    R_MIN = params.r_min
    n_inv = params.inv_grid_points

    inv_grid = np.linspace(0, S_BAR, n_inv)
    V_dp = np.full((T + 1, n_inv), np.inf)
    policy_x = np.zeros((T, n_inv))
    policy_y = np.zeros((T, n_inv))

    # Terminal condition: V_T(s) = -lambda_mean * s for s >= R_MIN
    mask = inv_grid >= R_MIN
    V_dp[T][mask] = -lambda_mean_xi * inv_grid[mask]

    for t in reversed(range(T)):
        Et_xi = target_xi[t]
        vol_t = vol_xi[t]
        p_sell_t = p_sell_xi[t]

        # Grids
        x_grid = np.linspace(0, vol_t, params.x_grid_points)   # (Nx,)
        # y-fraction in [0, 1] — scales per-state to match loop version,
        # which used y_grid = linspace(0, min(avail, S_BAR), Ny) per state.
        y_frac = np.linspace(0, 1, params.y_grid_points)  # (Ny,)

        # Shapes:
        #   s_prev: (Ni, 1, 1)
        #   x:     (1,  Nx, 1)
        #   y_frac:(1,  1,  Ny)
        s_prev = inv_grid[:, None, None]            # (Ni, 1, 1)
        x = x_grid[None, :, None]                   # (1,  Nx, 1)

        avail = s_prev + x                          # (Ni, Nx, 1)
        # Per-state y upper bound = min(avail, S_BAR), then scale by y_frac
        y_upper = np.minimum(avail, S_BAR)          # (Ni, Nx, 1)
        y = y_upper * y_frac[None, None, :]         # (Ni, Nx, Ny)

        # By construction y <= avail and y <= S_BAR, so always feasible.
        feasible = np.ones_like(y, dtype=bool)

        s_next = np.clip(avail - y, 0.0, S_BAR)     # (Ni, Nx, Ny)

        # Index into V_{t+1}
        j = np.minimum(
            np.round(s_next / S_BAR * (n_inv - 1)).astype(np.int64),
            n_inv - 1
        )
        V_next = V_dp[t + 1][j]                     # (Ni, Nx, Ny)

        # Stage cost
        disc = Et_xi - p_sell_t * y                 # (1, 1, Ny) -> broadcasts
        d_plus = np.maximum(disc, 0.0)
        d_minus = np.maximum(-disc, 0.0)

        stage = (ALPHA * d_plus + BETA * d_minus
                 - GAMMA * lambda_mean_xi * s_next)  # (Ni, Nx, Ny)

        total = stage + V_next                      # (Ni, Nx, Ny)

        # Infeasible -> +inf
        total = np.where(feasible, total, np.inf)
        # Also block transitions where V_next is inf
        total = np.where(np.isinf(V_next), np.inf, total)

        # Reshape to (Ni, Nx*Ny) and argmin
        Ni = n_inv
        Nx = params.x_grid_points
        Ny = params.y_grid_points
        flat = total.reshape(Ni, Nx * Ny)

        best_idx = np.argmin(flat, axis=1)          # (Ni,)
        best_val = np.take_along_axis(flat, best_idx[:, None], axis=1).ravel()

        # Decode (x, y) indices
        x_idx = best_idx // Ny
        y_idx = best_idx % Ny

        # Recover the actual y values (y is now (Ni, Nx, Ny) - state-dependent)
        y_flat = y.reshape(-1, Ny) if y.shape[0] == 1 else y  # ensure shape
        # y has shape (Ni, Nx, Ny); index by (i, x_idx[i], y_idx[i])
        i_range = np.arange(Ni)
        best_y_vals = y[i_range, x_idx, y_idx]      # (Ni,)

        V_dp[t] = np.where(np.isinf(best_val), np.inf, best_val)
        policy_x[t] = x_grid[x_idx]
        policy_y[t] = best_y_vals

    obj = V_dp[0][0]
    status = 'Optimal' if obj < np.inf else 'Infeasible'
    return V_dp, policy_x, policy_y, status, obj


# ─────────────────────────────────────────────────────────────
# Forward simulation
# ─────────────────────────────────────────────────────────────

def simulate_forward(
    policy_x, policy_y, vol_xi, fx_xi,
    target_xi, p_sell_xi, lambda_mean_xi,
    params: DPParams
):
    T = params.T
    S_BAR = params.s_bar
    n_inv = params.inv_grid_points
    FX_THRESHOLD = params.fx_threshold
    PHI = params.phi
    S0 = params.S0
    Q_ANNUAL = params.q_annual

    s = S0
    cum_purch = 0.0
    traj = {k: [] for k in [
        'inventory', 'purchase', 'release',
        'release_bog', 'release_market',
        'discrepancy', 'over_supply', 'under_supply',
        'cum_purchase']}

    for t in range(T):
        Et = target_xi[t]
        vol_t = vol_xi[t]
        fx_t = fx_xi[t]

        i = min(int(round(s / S_BAR * (n_inv - 1))), n_inv - 1)
        x_pol = float(np.clip(policy_x[t][i], 0.0, vol_t))
        avail = s + x_pol
        y_pol = float(np.clip(policy_y[t][i], 0.0, avail))
        s_next = float(np.clip(avail - y_pol, 0.0, S_BAR))
        cum_purch += x_pol

        if fx_t > FX_THRESHOLD:
            y_market, y_bog = y_pol, 0.0
        else:
            y_bog, y_market = y_pol, 0.0

        disc = Et - p_sell_xi[t] * y_pol
        traj['inventory'].append(s_next)
        traj['purchase'].append(x_pol)
        traj['release'].append(y_pol)
        traj['release_bog'].append(y_bog)
        traj['release_market'].append(y_market)
        traj['discrepancy'].append(disc)
        traj['over_supply'].append(max(disc, 0.0))
        traj['under_supply'].append(max(-disc, 0.0))
        traj['cum_purchase'].append(cum_purch)
        s = s_next

    shortfall_kg = max(Q_ANNUAL - cum_purch, 0.0)
    traj['shortfall_kg'] = shortfall_kg
    traj['shortfall_ghs'] = PHI * lambda_mean_xi * shortfall_kg
    traj['final_inv'] = s
    return traj


# ─────────────────────────────────────────────────────────────
# Full stochastic solve
# ─────────────────────────────────────────────────────────────

def solve_stochastic_dp(params: DPParams, progress_callback=None) -> Dict[str, Any]:
    """
    Solve the full stochastic DP across all scenarios.
    progress_callback(xi, n_scen, elapsed) is invoked after each scenario
    so Streamlit can update a progress bar.
    """
    gold_weekly = build_weekly_gold(params.jp_quarterly, params.T)
    cost_scen, vol_scen, fx_scen, inflow_scen, target_scen, probs = \
        generate_scenarios(params, gold_weekly)

    fx_mid = (params.fx_low + params.fx_high) / 2
    lambda_arr = gold_weekly * fx_mid * params.C_TROY / params.K_CONV

    n_scen = len(probs)
    t0 = time.time()
    all_V, all_px, all_py, all_traj, results = [], [], [], [], []

    for xi in range(n_scen):
        lambda_xi = gold_weekly * fx_scen[xi] * params.C_TROY / params.K_CONV
        lambda_mean_xi = float(lambda_xi.mean())
        p_sell_xi = lambda_xi.copy()

        V_dp, px, py, status_xi, obj_xi = run_single_dp_vectorized(
            cost_scen[xi], vol_scen[xi], target_scen[xi],
            lambda_xi, lambda_mean_xi, p_sell_xi, params
        )
        traj_xi = simulate_forward(
            px, py, vol_scen[xi], fx_scen[xi],
            target_scen[xi], p_sell_xi, lambda_mean_xi, params
        )

        all_V.append(V_dp)
        all_px.append(px)
        all_py.append(py)
        all_traj.append(traj_xi)

        disc_arr = (np.array(traj_xi['over_supply'])
                    + np.array(traj_xi['under_supply']))
        results.append({
            'Scenario': xi,
            'Status': status_xi,
            'Yearend_Inventory_kg': traj_xi['final_inv'],
            'Total_Purchase_kg': sum(traj_xi['purchase']),
            'Total_Release_kg': sum(traj_xi['release']),
            'Total_Release_BOG_kg': sum(traj_xi['release_bog']),
            'Total_Release_Market_kg': sum(traj_xi['release_market']),
            'Shortfall_kg': traj_xi['shortfall_kg'],
            'Shortfall_GHS': traj_xi['shortfall_ghs'],
            'Total_Abs_Disc_GHS': float(disc_arr.sum()),
            'Objective_GHS': obj_xi,
            'Over_Supply_Periods': sum(1 for v in traj_xi['over_supply'] if v > 0),
            'Under_Supply_Periods': sum(1 for v in traj_xi['under_supply'] if v > 0),
        })

        if progress_callback is not None:
            progress_callback(xi + 1, n_scen, time.time() - t0)

    solve_time = time.time() - t0
    df_scen = pd.DataFrame(results)
    n_opt = int((df_scen['Status'] == 'Optimal').sum())
    n_inf = int(n_scen - n_opt)

    w = probs
    inv_all = np.array([t['inventory'] for t in all_traj])
    purch_all = np.array([t['purchase'] for t in all_traj])
    release_all = np.array([t['release'] for t in all_traj])
    rel_bog_all = np.array([t['release_bog'] for t in all_traj])
    rel_mkt_all = np.array([t['release_market'] for t in all_traj])
    disc_all = np.array([t['discrepancy'] for t in all_traj])
    cum_all = np.array([t['cum_purchase'] for t in all_traj])

    inv_mean = np.average(inv_all, axis=0, weights=w)
    purch_mean = np.average(purch_all, axis=0, weights=w)
    release_mean = np.average(release_all, axis=0, weights=w)
    rel_bog_mean = np.average(rel_bog_all, axis=0, weights=w)
    rel_mkt_mean = np.average(rel_mkt_all, axis=0, weights=w)
    disc_mean = np.average(disc_all, axis=0, weights=w)
    cum_mean = np.average(cum_all, axis=0, weights=w)
    fx_mean = np.average(fx_scen, axis=0, weights=w)

    exp_obj = sum(
        probs[xi] * all_V[xi][0][0]
        for xi in range(n_scen)
        if all_V[xi][0][0] < np.inf
    )

    return {
        'exp_obj': exp_obj,
        'solve_time': solve_time,
        'inv_mean': inv_mean,
        'purch_mean': purch_mean,
        'release_mean': release_mean,
        'rel_bog_mean': rel_bog_mean,
        'rel_mkt_mean': rel_mkt_mean,
        'disc_mean': disc_mean,
        'cum_mean': cum_mean,
        'fx_mean': fx_mean,
        'inv_all': inv_all,
        'purch_all': purch_all,
        'release_all': release_all,
        'rel_bog_all': rel_bog_all,
        'rel_mkt_all': rel_mkt_all,
        'df_scen': df_scen,
        'all_V': all_V,
        'all_px': all_px,
        'all_py': all_py,
        'lambda_mean': float(lambda_arr.mean()),
        'lambda_arr': lambda_arr,
        'gold_weekly': gold_weekly,
        'n_opt': n_opt,
        'n_inf': n_inf,
        'params': params,
    }
