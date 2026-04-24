"""
Verify the vectorized solver produces the same results as the
original loop-based implementation, and measure speedup.
"""
import sys
import time
import numpy as np

sys.path.insert(0, '/home/claude/goldbod_app')
from solver import DPParams, solve_stochastic_dp, build_weekly_gold


# ─────────────────────────────────────────────────────────────
# Original loop-based solver for comparison (from user's code)
# ─────────────────────────────────────────────────────────────

def run_single_dp_loops(
    cost_xi, vol_xi, target_xi,
    lambda_xi, lambda_mean_xi, p_sell_xi,
    params
):
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

    for i, sv in enumerate(inv_grid):
        if sv >= R_MIN:
            V_dp[T][i] = -lambda_mean_xi * sv

    for t in reversed(range(T)):
        Et_xi = target_xi[t]
        vol_t = vol_xi[t]
        p_sell_t = p_sell_xi[t]
        x_grid = np.linspace(0, vol_t, params.x_grid_points)

        for i, s_prev in enumerate(inv_grid):
            best_val = np.inf
            best_x = 0.0
            best_y = 0.0

            for xv in x_grid:
                avail = s_prev + xv
                y_grid = np.linspace(0, min(avail, S_BAR), params.y_grid_points)
                for yv in y_grid:
                    s_next = float(np.clip(avail - yv, 0.0, S_BAR))
                    j = min(int(round(s_next / S_BAR * (n_inv - 1))), n_inv - 1)
                    if V_dp[t + 1][j] == np.inf:
                        continue
                    disc = Et_xi - p_sell_t * yv
                    d_plus = max(disc, 0.0)
                    d_minus = max(-disc, 0.0)
                    val = (ALPHA * d_plus + BETA * d_minus
                           - GAMMA * lambda_mean_xi * s_next
                           + V_dp[t + 1][j])
                    if val < best_val:
                        best_val = val
                        best_x = xv
                        best_y = yv

            V_dp[t][i] = best_val if best_val < np.inf else np.inf
            policy_x[t][i] = best_x
            policy_y[t][i] = best_y

    return V_dp, policy_x, policy_y


# ─────────────────────────────────────────────────────────────
# Setup — use a small problem for the loop version
# ─────────────────────────────────────────────────────────────

# Smaller grid for loop-based timing (full grid would take too long)
params_small = DPParams(
    T=52,
    inv_grid_points=100,
    x_grid_points=15,
    y_grid_points=15,
    num_scenarios=3,
)

gold_weekly = build_weekly_gold(params_small.jp_quarterly, params_small.T)
rng = np.random.default_rng(params_small.seed)
n = params_small.num_scenarios
T = params_small.T

fx = rng.uniform(params_small.fx_low, params_small.fx_high, (n, T))
vol = rng.uniform(params_small.v_min, params_small.v_max, (n, T))
cost = gold_weekly[None, :] * fx * params_small.C_TROY / params_small.K_CONV
inflow = cost * vol
target = params_small.rho * inflow

# Take scenario 0 for comparison
xi = 0
lambda_xi = gold_weekly * fx[xi] * params_small.C_TROY / params_small.K_CONV
lambda_mean_xi = float(lambda_xi.mean())
p_sell_xi = lambda_xi.copy()

# ─────────────────────────────────────────────────────────────
# Time the loop version
# ─────────────────────────────────────────────────────────────
print("Testing loop-based solver (scenario 0, smaller grid)...")
t0 = time.time()
V_loop, px_loop, py_loop = run_single_dp_loops(
    cost[xi], vol[xi], target[xi],
    lambda_xi, lambda_mean_xi, p_sell_xi,
    params_small
)
t_loop = time.time() - t0
print(f"  Loop version:       {t_loop:.2f}s")

# ─────────────────────────────────────────────────────────────
# Time the vectorized version on the same params
# ─────────────────────────────────────────────────────────────
from solver import run_single_dp_vectorized

t0 = time.time()
V_vec, px_vec, py_vec, status, obj = run_single_dp_vectorized(
    cost[xi], vol[xi], target[xi],
    lambda_xi, lambda_mean_xi, p_sell_xi,
    params_small
)
t_vec = time.time() - t0
print(f"  Vectorized version: {t_vec:.2f}s")
print(f"  Speedup: {t_loop / t_vec:.1f}x")

# ─────────────────────────────────────────────────────────────
# Compare results
# ─────────────────────────────────────────────────────────────
print("\nValue function comparison (finite entries only):")
V_diff = np.where(np.isinf(V_loop) | np.isinf(V_vec), 0, V_loop - V_vec)
print(f"  Max abs diff: {np.abs(V_diff).max():.6f}")
print(f"  Mean abs diff: {np.abs(V_diff).mean():.6f}")
print(f"  V_loop[0][0] = {V_loop[0][0]:,.2f}")
print(f"  V_vec[0][0]  = {V_vec[0][0]:,.2f}")
print(f"  Rel diff at (0,0): {abs(V_loop[0][0] - V_vec[0][0]) / abs(V_loop[0][0]) * 100:.4f}%")

print("\nPolicy comparison:")
px_diff = np.abs(px_loop - px_vec)
py_diff = np.abs(py_loop - py_vec)
print(f"  Purchase policy max diff: {px_diff.max():.4f} kg")
print(f"  Release policy max diff: {py_diff.max():.4f} kg")
print(f"  Purchase mean diff: {px_diff.mean():.4f} kg")
print(f"  Release mean diff: {py_diff.mean():.4f} kg")
