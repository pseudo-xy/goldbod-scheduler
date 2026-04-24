# GoldBod Stochastic DP Scheduler

Interactive web application for GoldBod's stochastic finite-horizon dynamic
programming model. Generates optimal weekly gold procurement and release
schedules under FX and ASM volume uncertainty.

## Features

- **Full parameter control** — every model input is user-editable:
  quarterly gold price forecasts, FX range, ASM volume, objective weights
  (α, β, γ, φ), grid resolution, scenario count.
- **Vectorized NumPy DP solver** — ~170× faster than the original loop-based
  implementation. A full 20-scenario solve at 300×30×30 grid resolution
  runs in ~7 seconds.
- **Interactive outputs** — KPI tiles, inventory fan, cumulative purchases vs
  GANRAP target, policy maps, value function, outcome distributions.
- **Downloadable results** — CSV exports for weekly schedule, per-scenario
  summary, inventory paths, and aggregate statistics. PNG exports for all plots.
- **Live input preview** — weekly gold price and buy-cost curves update as
  you edit the quarterly forecasts.

## Project Structure

```
goldbod_app/
├── app.py              # Streamlit UI
├── solver.py           # Vectorized DP solver
├── requirements.txt    # Python dependencies
├── test_correctness.py # Correctness test vs loop-based solver
└── README.md           # This file
```

## Local Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Deploy to Streamlit Community Cloud (free)

1. Push this folder to a new public GitHub repository.
2. Go to <https://share.streamlit.io> and sign in with GitHub.
3. Click **New app** → select the repo and branch → set main file to `app.py`.
4. Click **Deploy**. Takes about 2–3 minutes the first time.
5. You'll get a URL like `https://<your-app>.streamlit.app`.

To restrict access, use Streamlit's built-in authentication or keep the repo
private (paid plan) — for a truly private internal tool, consider Render or
Railway instead.

## Model Summary

- **Horizon**: T = 52 weekly periods (1 year)
- **State**: inventory buffer sₜ ∈ [0, S̄]
- **Decisions**: purchase xₜ ∈ [0, Vₜ], release yₜ ∈ [0, sₜ + xₜ]
- **Uncertainty**:
  - FX: rₜ ~ U[fx_low, fx_high] GHS/USD
  - ASM volume: Vₜ ~ U[v_min, v_max] kg/week
  - Gold price: fixed from quarterly forecasts (linear weekly interpolation)
- **Buy cost per kg**: λₜ = (goldₜ × rₜ × C_TROY) / K_CONV
- **Transfer price proxy**: p_sellₜ = λₜ (Option A, since no 2026 actuals)
- **Release routing**: if rₜ > FX_THRESHOLD → market; else → BoG reserve
- **Objective** (per scenario, minimized):
  ```
  Σₜ [α·(Eₜ - p_sell·yₜ)₊ + β·(p_sell·yₜ - Eₜ)₊ - γ·λ̄·sₜ₊₁]
    − λ̄·s_T  +  φ·λ̄·max(Q − Σxₜ, 0)
  ```
  where `Eₜ = ρ·λₜ·Vₜ`.

## Performance Notes

Solve time scales roughly as:

```
O(num_scenarios × T × inv_grid × x_grid × y_grid)
```

Typical timings (single CPU, Streamlit Cloud free tier):

| Grid (inv × x × y) | Scenarios | Time   |
|--------------------|-----------|--------|
| 100 × 15 × 15      | 20        | ~1 s   |
| 300 × 30 × 30      | 20        | ~7 s   |
| 500 × 60 × 60      | 20        | ~40 s  |

For quick exploration use lower resolution. For final production schedules,
use the full 300×30×30 grid or higher.

## Assumptions & Caveats

1. **Transfer price proxy**: without 2026 BoG transfer prices, we set
   p_sell = λ. The discrepancy metric therefore reflects cost-based supply
   performance rather than revenue-based performance. For 2025 data the
   actual transfer price was ~4.76× the buy cost.
2. **Gold price deterministic**: treated as known from JP Morgan forecasts.
   Only FX and volume are stochastic.
3. **Weekly interpolation**: linear between quarter-end forecast anchors.
4. **Annual shortfall penalty** (φ): applied only in the forward simulation,
   not in the backward DP. This is consistent with GANRAP being an annual
   mandate, not a per-week constraint.
