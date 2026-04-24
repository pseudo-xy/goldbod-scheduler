"""
GoldBod Stochastic DP — Interactive Web Application

Streamlit front-end for the vectorized stochastic DP solver.
Every parameter is user-editable. All outputs (tables, CSVs, plots)
are downloadable.

Run locally:
    streamlit run app.py

Deploy on Streamlit Community Cloud:
    1. Push this repo to GitHub
    2. Connect at https://share.streamlit.io
    3. Select app.py as the entry point
"""

import io
import time
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import streamlit as st

from solver import DPParams, solve_stochastic_dp, build_weekly_gold


# ─────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="GoldBod DP Scheduler",
    page_icon="🟡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Plot colors
C1 = '#1f77b4'
C2 = '#d62728'
C3 = '#2ca02c'
C4 = '#ff7f0e'


# ─────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────

st.title("🟡 GoldBod Stochastic DP Scheduler")
st.caption(
    "Optimal weekly gold procurement & release schedule — "
    "stochastic finite-horizon dynamic programming with FX and ASM volume uncertainty."
)

# ─────────────────────────────────────────────────────────────
# Sidebar — all inputs
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Model Inputs")

    # ─── Gold price forecast ───
    st.subheader("Gold Price Forecast")
    st.caption("Quarterly forecasts (USD/oz). Weekly series interpolated linearly.")
    col_q1, col_q2 = st.columns(2)
    with col_q1:
        q1 = st.number_input("Q1", value=4440.0, min_value=100.0, max_value=20000.0, step=10.0, format="%.0f")
        q3 = st.number_input("Q3", value=4860.0, min_value=100.0, max_value=20000.0, step=10.0, format="%.0f")
    with col_q2:
        q2 = st.number_input("Q2", value=4655.0, min_value=100.0, max_value=20000.0, step=10.0, format="%.0f")
        q4 = st.number_input("Q4", value=5055.0, min_value=100.0, max_value=20000.0, step=10.0, format="%.0f")

    # ─── FX distribution ───
    st.subheader("Exchange Rate (GHS/USD)")
    col_fx1, col_fx2 = st.columns(2)
    with col_fx1:
        fx_low = st.number_input("FX low", value=10.50, min_value=0.1, max_value=100.0, step=0.1, format="%.2f")
    with col_fx2:
        fx_high = st.number_input("FX high", value=12.50, min_value=0.1, max_value=100.0, step=0.1, format="%.2f")
    fx_threshold = st.number_input(
        "FX release threshold",
        value=11.00, min_value=0.1, max_value=100.0, step=0.1, format="%.2f",
        help="Above threshold → sell on international market. At/below → transfer to BoG reserve."
    )

    # ─── ASM volume ───
    st.subheader("ASM Weekly Volume (kg)")
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        v_min = st.number_input("V_min", value=2000.0, min_value=0.0, max_value=50000.0, step=100.0, format="%.0f")
    with col_v2:
        v_max = st.number_input("V_max", value=3000.0, min_value=0.0, max_value=50000.0, step=100.0, format="%.0f")

    # ─── Procurement target ───
    st.subheader("GANRAP Target")
    q_annual = st.number_input(
        "Q_ANNUAL (kg)", value=127000.0, min_value=0.0, max_value=1_000_000.0, step=1000.0, format="%.0f",
        help="Annual procurement mandate."
    )

    # ─── Supply ratio & buffer ───
    st.subheader("Supply Parameters")
    rho = st.number_input("Supply ratio ρ", value=0.99242, min_value=0.0, max_value=1.0, step=0.001, format="%.5f")
    s_bar = st.number_input("Working buffer cap S̄ (kg)", value=3000.0, min_value=0.0, max_value=100000.0, step=100.0, format="%.0f")

    # ─── Conversion constants ───
    with st.expander("Conversion constants (advanced)"):
        c_troy = st.number_input("C_TROY (troy oz/kg)", value=32.1507, step=0.0001, format="%.4f")
        k_conv = st.number_input("K_CONV (GHP/troy oz)", value=4.186, step=0.001, format="%.3f")

    # ─── Objective weights ───
    st.subheader("Objective Weights")
    alpha = st.number_input("α (under-release penalty)", value=3.861, min_value=0.0, step=0.01, format="%.3f")
    beta = st.number_input("β (over-release penalty)", value=1.650, min_value=0.0, step=0.01, format="%.3f")
    gamma = st.number_input("γ (intermediate inventory reward)", value=0.164, min_value=0.0, step=0.001, format="%.3f")
    phi = st.number_input("φ (annual shortfall penalty)", value=5.0, min_value=0.0, step=0.1, format="%.2f")

    # ─── Stochastic setup ───
    st.subheader("Stochastic Setup")
    num_scenarios = st.slider("Number of scenarios", min_value=5, max_value=100, value=20, step=5)
    seed = st.number_input("Random seed", value=42, min_value=0, max_value=100000, step=1)

    # ─── DP grid ───
    st.subheader("DP Grid Resolution")
    st.caption("Higher resolution = more accurate but slower.")
    inv_grid = st.slider("Inventory grid points", min_value=50, max_value=500, value=300, step=50)
    x_grid = st.slider("Purchase grid points", min_value=10, max_value=60, value=30, step=5)
    y_grid = st.slider("Release grid points", min_value=10, max_value=60, value=30, step=5)

    # ─── Run button ───
    st.divider()
    run_button = st.button("🚀 Run DP Solver", type="primary", use_container_width=True)

# ─────────────────────────────────────────────────────────────
# Build params object
# ─────────────────────────────────────────────────────────────

params = DPParams(
    T=52,
    jp_quarterly=(q1, q2, q3, q4),
    fx_low=fx_low,
    fx_high=fx_high,
    fx_threshold=fx_threshold,
    v_min=v_min,
    v_max=v_max,
    q_annual=q_annual,
    rho=rho,
    s_bar=s_bar,
    C_TROY=c_troy,
    K_CONV=k_conv,
    alpha=alpha,
    beta=beta,
    gamma=gamma,
    phi=phi,
    num_scenarios=num_scenarios,
    seed=int(seed),
    inv_grid_points=inv_grid,
    x_grid_points=x_grid,
    y_grid_points=y_grid,
)

# ─────────────────────────────────────────────────────────────
# Validation warnings
# ─────────────────────────────────────────────────────────────

warnings = []
if fx_low >= fx_high:
    warnings.append("FX low must be less than FX high.")
if v_min >= v_max:
    warnings.append("V_min must be less than V_max.")
if not (fx_low <= fx_threshold <= fx_high):
    warnings.append(f"FX threshold ({fx_threshold}) is outside FX range [{fx_low}, {fx_high}]. "
                    "All scenarios will route to one side.")
if q1 <= 0 or q2 <= 0 or q3 <= 0 or q4 <= 0:
    warnings.append("All quarterly gold forecasts must be positive.")

for w in warnings:
    st.warning(w)

# ─────────────────────────────────────────────────────────────
# Input preview pane (always visible)
# ─────────────────────────────────────────────────────────────

st.subheader("Input Preview")

preview_cols = st.columns([2, 1])

with preview_cols[0]:
    # Gold price weekly preview
    gold_preview = build_weekly_gold(params.jp_quarterly, params.T)
    fx_mid = (fx_low + fx_high) / 2
    lambda_preview = gold_preview * fx_mid * c_troy / k_conv

    fig_preview, axp = plt.subplots(figsize=(10, 3.2), facecolor='white')
    axp.plot(range(params.T), gold_preview, color='goldenrod', lw=2, marker='o', ms=3)
    axp.set_xlabel('Week')
    axp.set_ylabel('Gold Price (USD/oz)', color='goldenrod')
    axp.tick_params(axis='y', labelcolor='goldenrod')
    axp.set_title('Weekly Gold Price (interpolated) & Buy Cost (mid-FX)', fontsize=11, fontweight='bold')
    axp.grid(True, alpha=0.3)

    axp2 = axp.twinx()
    axp2.plot(range(params.T), lambda_preview, color=C1, lw=1.5, ls='--')
    axp2.set_ylabel('λ buy cost (GHS/kg)', color=C1)
    axp2.tick_params(axis='y', labelcolor=C1)

    plt.tight_layout()
    st.pyplot(fig_preview, use_container_width=True)
    plt.close(fig_preview)

with preview_cols[1]:
    st.markdown("**Quarterly forecast**")
    st.dataframe(
        pd.DataFrame({
            "Quarter": ["Q1", "Q2", "Q3", "Q4"],
            "USD/oz": [f"${q1:,.0f}", f"${q2:,.0f}", f"${q3:,.0f}", f"${q4:,.0f}"],
        }),
        hide_index=True,
        use_container_width=True,
    )
    st.metric("Mid-FX buy cost (mean)", f"GHS {lambda_preview.mean():,.0f}/kg")
    st.metric("Gold range", f"${gold_preview.min():,.0f} – ${gold_preview.max():,.0f}")

st.divider()

# ─────────────────────────────────────────────────────────────
# Run the solver
# ─────────────────────────────────────────────────────────────

if run_button and not warnings:
    progress_bar = st.progress(0.0, text="Solving scenarios...")
    status_area = st.empty()

    def _cb(done, total, elapsed):
        progress_bar.progress(done / total, text=f"Solving scenarios... {done}/{total} (elapsed {elapsed:.1f}s)")

    with st.spinner("Running stochastic DP..."):
        t0 = time.time()
        res = solve_stochastic_dp(params, progress_callback=_cb)
        elapsed = time.time() - t0

    progress_bar.empty()
    st.success(f"✅ Solve complete in {elapsed:.1f}s — "
               f"{res['n_opt']}/{num_scenarios} scenarios optimal")

    # Cache in session state so tabs persist across reruns
    st.session_state['res'] = res
    st.session_state['params'] = params

# ─────────────────────────────────────────────────────────────
# Results display
# ─────────────────────────────────────────────────────────────

if 'res' in st.session_state:
    res = st.session_state['res']
    params = st.session_state['params']
    df_scen = res['df_scen']
    gold_weekly = res['gold_weekly']
    T = params.T
    t_ax = list(range(T))

    # ─── KPI tiles ───
    st.subheader("Key Results")
    kpi_cols = st.columns(5)
    with kpi_cols[0]:
        st.metric("Expected Objective (GHS)", f"{res['exp_obj']:,.0f}")
    with kpi_cols[1]:
        st.metric(
            "Mean Year-end Inventory",
            f"{df_scen['Yearend_Inventory_kg'].mean():,.0f} kg",
            f"±{df_scen['Yearend_Inventory_kg'].std():,.0f}"
        )
    with kpi_cols[2]:
        mean_purch = df_scen['Total_Purchase_kg'].mean()
        st.metric(
            "Mean Total Purchase",
            f"{mean_purch:,.0f} kg",
            f"{(mean_purch - params.q_annual):+,.0f} vs Q"
        )
    with kpi_cols[3]:
        n_meeting = int((df_scen['Shortfall_kg'] == 0).sum())
        st.metric(
            "Scenarios Meeting Q",
            f"{n_meeting} / {len(df_scen)}",
            f"{n_meeting / len(df_scen) * 100:.0f}%"
        )
    with kpi_cols[4]:
        st.metric(
            "Mean Shortfall",
            f"{df_scen['Shortfall_kg'].mean():,.0f} kg",
            f"±{df_scen['Shortfall_kg'].std():,.0f}"
        )

    # ─── Second row: release routing summary ───
    _mean_bog = df_scen['Total_Release_BOG_kg'].mean()
    _mean_mkt = df_scen['Total_Release_Market_kg'].mean()
    _mean_rel = df_scen['Total_Release_kg'].mean()
    _bog_pct = (_mean_bog / _mean_rel * 100) if _mean_rel > 0 else 0.0
    _mkt_pct = (_mean_mkt / _mean_rel * 100) if _mean_rel > 0 else 0.0

    route_cols = st.columns(3)
    with route_cols[0]:
        st.metric(
            "Mean Release to BoG Reserve",
            f"{_mean_bog:,.0f} kg",
            f"{_bog_pct:.1f}% of releases",
        )
    with route_cols[1]:
        st.metric(
            "Mean Release to Market",
            f"{_mean_mkt:,.0f} kg",
            f"{_mkt_pct:.1f}% of releases",
        )
    with route_cols[2]:
        st.metric(
            "Mean Total Release",
            f"{_mean_rel:,.0f} kg",
            f"across {params.num_scenarios} scenarios",
        )

    st.divider()

    # ─── Tabs for different views ───
    tabs = st.tabs([
        "📈 Trajectory",
        "🗺️ Policy Maps",
        "🔥 Value Function",
        "📊 Distributions",
        "🏦 BoG vs Market Split",
        "📋 Results Table",
        "💾 Downloads",
    ])

    # ═══ Trajectory tab ═══
    with tabs[0]:
        fig, axes = plt.subplots(4, 1, figsize=(14, 17), facecolor='white')
        inv_all = res['inv_all']

        # Inventory fan
        ax = axes[0]
        ax.fill_between(t_ax, inv_all.min(0), inv_all.max(0),
                        alpha=0.12, color=C2, label='Scenario range')
        ax.fill_between(t_ax,
                        np.percentile(inv_all, 25, 0),
                        np.percentile(inv_all, 75, 0),
                        alpha=0.25, color=C2, label='IQR')
        ax.plot(t_ax, res['inv_mean'], color=C2, lw=2.2,
                label='Mean inventory', marker='o', ms=2.5)
        ax.axhline(params.s_bar, color='grey', lw=1.0, ls=':',
                   label=f'S̄ ({params.s_bar:,.0f} kg)')
        ax.set_title('GoldBod Inventory Buffer', fontsize=12, fontweight='bold')
        ax.set_ylabel('Inventory (kg)')
        ax.legend(ncol=3, fontsize=8)
        ax.grid(True, alpha=0.3)

        # Cumulative purchase vs Q
        ax2 = axes[1]
        cum_scen = np.cumsum(res['purch_all'], axis=1)
        ax2.plot(t_ax, res['cum_mean'], color=C1, lw=2.2,
                 label='Mean cumulative purchase')
        ax2.fill_between(t_ax, cum_scen.min(0), cum_scen.max(0),
                         alpha=0.10, color=C1)
        ax2.axhline(params.q_annual, color='red', lw=1.8, ls='--',
                    label=f'Q_ANNUAL = {params.q_annual:,.0f} kg')
        ax2.set_title('Cumulative Purchases vs GANRAP Target', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Cumulative Purchase (kg)')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        # Weekly purchase and release
        ax3 = axes[2]
        ax3.bar(t_ax, res['purch_mean'], color=C1, alpha=0.75,
                width=0.7, label='Mean purchase xₜ')
        ax3.bar(t_ax, res['release_mean'], color=C4, alpha=0.6,
                width=0.7, label='Mean release yₜ')
        ax3.axhline(params.v_min, color='orange', lw=1.3, ls='--',
                    label=f'V_min = {params.v_min:,.0f} kg/week')
        ax3.set_title('Weekly Purchase and Release Decisions', fontsize=12, fontweight='bold')
        ax3.set_ylabel('kg/week')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3, axis='y')

        # Discrepancy + gold price
        ax4 = axes[3]
        ax4b = ax4.twinx()
        ax4.plot(t_ax, res['disc_mean'], color=C3, lw=2.0,
                 marker='s', ms=3, label='Mean discrepancy')
        ax4b.plot(t_ax, gold_weekly, color='goldenrod', lw=1.5,
                  ls='--', label='Gold price')
        ax4.axhline(0, color='black', lw=0.8, ls=':')
        ax4.set_title('Mean Supply Discrepancy & Gold Price', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Week')
        ax4.set_ylabel('Discrepancy (GHS)', color=C3)
        ax4b.set_ylabel('Gold Price (USD/oz)', color='goldenrod')
        lines = ax4.get_legend_handles_labels()
        lines2 = ax4b.get_legend_handles_labels()
        ax4.legend(lines[0] + lines2[0], lines[1] + lines2[1], fontsize=8)
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

        # Download button for figure
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=200, bbox_inches='tight', facecolor='white')
        buf.seek(0)
        st.download_button(
            "⬇️ Download trajectory plot (PNG)",
            buf,
            file_name=f"dp_trajectory_{datetime.now():%Y%m%d_%H%M%S}.png",
            mime="image/png",
        )
        plt.close(fig)

    # ═══ Policy Maps tab ═══
    with tabs[1]:
        avg_px = np.mean(res['all_px'], axis=0)
        avg_py = np.mean(res['all_py'], axis=0)

        fig, axes = plt.subplots(1, 2, figsize=(16, 5), facecolor='white')
        for ax, data, title, label in [
            (axes[0], avg_px, 'Optimal Purchase Policy x*(t,s)', 'Purchase x* (kg)'),
            (axes[1], avg_py, 'Optimal Release Policy y*(t,s)', 'Release y* (kg)'),
        ]:
            im = ax.imshow(data.T, aspect='auto', origin='lower', cmap='Blues',
                           extent=[-0.5, T - 0.5, 0, params.s_bar])
            plt.colorbar(im, ax=ax, label=label)
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.set_xlabel('Week t')
            ax.set_ylabel('Inventory State s (kg)')
        plt.suptitle('Average Optimal Policy Maps', fontsize=12, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=180, bbox_inches='tight', facecolor='white')
        buf.seek(0)
        st.download_button(
            "⬇️ Download policy map (PNG)",
            buf,
            file_name=f"dp_policy_map_{datetime.now():%Y%m%d_%H%M%S}.png",
            mime="image/png",
        )
        plt.close(fig)

    # ═══ Value Function tab ═══
    with tabs[2]:
        inv_grid_pts = np.linspace(0, params.s_bar, params.inv_grid_points)
        V_avg = np.mean(res['all_V'], axis=0)
        V_plot = np.where(np.isinf(V_avg), np.nan, V_avg)

        fig, ax = plt.subplots(figsize=(14, 5), facecolor='white')
        im = ax.contourf(range(T + 1), inv_grid_pts, V_plot.T, levels=30, cmap='RdYlGn')
        plt.colorbar(im, ax=ax, label='Vₜ(s) [GHS]')
        ax.set_title('Average Value Function', fontsize=12, fontweight='bold')
        ax.set_xlabel('Week t')
        ax.set_ylabel('Inventory s (kg)')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=180, bbox_inches='tight', facecolor='white')
        buf.seek(0)
        st.download_button(
            "⬇️ Download value function (PNG)",
            buf,
            file_name=f"dp_value_function_{datetime.now():%Y%m%d_%H%M%S}.png",
            mime="image/png",
        )
        plt.close(fig)

    # ═══ Distributions tab ═══
    with tabs[3]:
        fig, axes = plt.subplots(2, 2, figsize=(14, 9), facecolor='white')
        panels = [
            (df_scen['Yearend_Inventory_kg'], 'Year-end Inventory (kg)', axes[0, 0], C1),
            (df_scen['Shortfall_kg'],         'Annual Shortfall vs Q (kg)', axes[0, 1], C2),
            (df_scen['Total_Release_kg'],     'Total Release (kg)', axes[1, 0], C4),
            (df_scen['Total_Purchase_kg'],    'Total Purchase (kg)', axes[1, 1], C1),
        ]
        for vals, xlabel, ax, color in panels:
            ax.hist(vals, bins=8, color=color, alpha=0.78, edgecolor='white')
            ax.axvline(vals.mean(), color='black', lw=2, ls='--',
                       label=f'Mean: {vals.mean():,.0f}')
            if vals.std() > 0:
                ax.axvline(vals.mean() + vals.std(), color='grey', lw=1.2, ls=':',
                           label=f'±1SD: {vals.std():,.0f}')
                ax.axvline(vals.mean() - vals.std(), color='grey', lw=1.2, ls=':')
            ax.set_xlabel(xlabel, fontsize=9)
            ax.set_ylabel('Frequency')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        plt.suptitle(
            f'Outcome Distributions — {params.num_scenarios} Scenarios\n'
            f'FX~U[{params.fx_low},{params.fx_high}], '
            f'Vol~U[{params.v_min:.0f},{params.v_max:.0f}] kg/week',
            fontsize=11, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=180, bbox_inches='tight', facecolor='white')
        buf.seek(0)
        st.download_button(
            "⬇️ Download distributions (PNG)",
            buf,
            file_name=f"dp_distributions_{datetime.now():%Y%m%d_%H%M%S}.png",
            mime="image/png",
        )
        plt.close(fig)

    # ═══ BoG vs Market Split tab ═══
    with tabs[4]:
        st.markdown("### Release Routing — BoG Reserve vs International Market")
        st.caption(
            f"Routing rule: when FX > {params.fx_threshold:.2f} GHS/USD (cedi weak), "
            f"gold is sold on the international market. "
            f"When FX ≤ {params.fx_threshold:.2f}, it's transferred to BoG's national reserve."
        )

        # ─── Summary metrics ───
        mean_bog_total = df_scen['Total_Release_BOG_kg'].mean()
        mean_mkt_total = df_scen['Total_Release_Market_kg'].mean()
        mean_total_rel = df_scen['Total_Release_kg'].mean()

        if mean_total_rel > 0:
            bog_pct = mean_bog_total / mean_total_rel * 100
            mkt_pct = mean_mkt_total / mean_total_rel * 100
        else:
            bog_pct = mkt_pct = 0.0

        split_kpi_cols = st.columns(4)
        with split_kpi_cols[0]:
            st.metric(
                "Mean Release to BoG Reserve",
                f"{mean_bog_total:,.0f} kg",
                f"{bog_pct:.1f}% of total",
            )
        with split_kpi_cols[1]:
            st.metric(
                "Mean Release to Market",
                f"{mean_mkt_total:,.0f} kg",
                f"{mkt_pct:.1f}% of total",
            )
        with split_kpi_cols[2]:
            # Count weeks (mean across scenarios) where each route was used
            bog_weeks = int(np.mean([
                np.sum(np.array(res['rel_bog_all'][xi]) > 0)
                for xi in range(params.num_scenarios)
            ]))
            st.metric(
                "Avg Weeks Routed to BoG",
                f"{bog_weeks} / {T}",
                f"{bog_weeks / T * 100:.0f}% of year",
            )
        with split_kpi_cols[3]:
            mkt_weeks = int(np.mean([
                np.sum(np.array(res['rel_mkt_all'][xi]) > 0)
                for xi in range(params.num_scenarios)
            ]))
            st.metric(
                "Avg Weeks Routed to Market",
                f"{mkt_weeks} / {T}",
                f"{mkt_weeks / T * 100:.0f}% of year",
            )

        st.divider()

        # ─── Three-panel chart ───
        fig_split, axes_s = plt.subplots(3, 1, figsize=(14, 13), facecolor='white')

        # Panel 1: Weekly stacked bars (BoG bottom, Market top)
        ax_s1 = axes_s[0]
        ax_s1.bar(t_ax, res['rel_bog_mean'], color=C3, alpha=0.85,
                  width=0.8, label='Mean release to BoG reserve')
        ax_s1.bar(t_ax, res['rel_mkt_mean'], bottom=res['rel_bog_mean'],
                  color=C4, alpha=0.85, width=0.8,
                  label='Mean release to international market')
        ax_s1.set_title('Weekly Release Split — BoG Reserve vs Market',
                        fontsize=12, fontweight='bold')
        ax_s1.set_ylabel('Release (kg/week)')
        ax_s1.set_xlabel('Week')
        ax_s1.legend(fontsize=9)
        ax_s1.grid(True, alpha=0.3, axis='y')

        # Panel 2: Cumulative split over the year
        ax_s2 = axes_s[1]
        cum_bog = np.cumsum(res['rel_bog_mean'])
        cum_mkt = np.cumsum(res['rel_mkt_mean'])
        ax_s2.fill_between(t_ax, 0, cum_bog, color=C3, alpha=0.5,
                           label=f'Cumulative BoG reserve ({cum_bog[-1]:,.0f} kg)')
        ax_s2.fill_between(t_ax, cum_bog, cum_bog + cum_mkt, color=C4, alpha=0.5,
                           label=f'Cumulative market ({cum_mkt[-1]:,.0f} kg)')
        ax_s2.plot(t_ax, cum_bog, color=C3, lw=2.0)
        ax_s2.plot(t_ax, cum_bog + cum_mkt, color=C4, lw=2.0)
        ax_s2.set_title('Cumulative Release by Destination',
                        fontsize=12, fontweight='bold')
        ax_s2.set_ylabel('Cumulative Release (kg)')
        ax_s2.set_xlabel('Week')
        ax_s2.legend(fontsize=9)
        ax_s2.grid(True, alpha=0.3)

        # Panel 3: FX trajectory with threshold line, colored by routing
        ax_s3 = axes_s[2]
        fx_mean_arr = res['fx_mean']
        # Shade regions based on mean FX vs threshold
        above_mask = fx_mean_arr > params.fx_threshold
        ax_s3.plot(t_ax, fx_mean_arr, color='navy', lw=2.0, marker='o', ms=3,
                   label='Mean FX across scenarios')
        ax_s3.fill_between(t_ax, params.fx_low, fx_mean_arr,
                           where=above_mask, color=C4, alpha=0.2,
                           label=f'FX > {params.fx_threshold} → Market')
        ax_s3.fill_between(t_ax, params.fx_low, fx_mean_arr,
                           where=~above_mask, color=C3, alpha=0.2,
                           label=f'FX ≤ {params.fx_threshold} → BoG')
        ax_s3.axhline(params.fx_threshold, color='red', lw=1.5, ls='--',
                      label=f'FX threshold = {params.fx_threshold:.2f}')
        ax_s3.set_title('FX Trajectory & Routing Decision',
                        fontsize=12, fontweight='bold')
        ax_s3.set_ylabel('FX (GHS/USD)')
        ax_s3.set_xlabel('Week')
        ax_s3.legend(fontsize=8, loc='best')
        ax_s3.grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig_split, use_container_width=True)

        # Download button
        buf_split = io.BytesIO()
        fig_split.savefig(buf_split, format='png', dpi=200,
                          bbox_inches='tight', facecolor='white')
        buf_split.seek(0)
        st.download_button(
            "⬇️ Download split analysis (PNG)",
            buf_split,
            file_name=f"dp_bog_market_split_{datetime.now():%Y%m%d_%H%M%S}.png",
            mime="image/png",
        )
        plt.close(fig_split)

        st.divider()

        # ─── Per-scenario split breakdown ───
        st.markdown("#### Per-Scenario Split Breakdown")

        # Build a tidy scenario-level split table
        split_df = pd.DataFrame({
            'Scenario': df_scen['Scenario'],
            'Total_Release_kg': df_scen['Total_Release_kg'],
            'BoG_Reserve_kg': df_scen['Total_Release_BOG_kg'],
            'Market_kg': df_scen['Total_Release_Market_kg'],
        })
        # Avoid division by zero
        safe_total = split_df['Total_Release_kg'].replace(0, np.nan)
        split_df['BoG_Pct'] = (split_df['BoG_Reserve_kg'] / safe_total * 100).fillna(0)
        split_df['Market_Pct'] = (split_df['Market_kg'] / safe_total * 100).fillna(0)

        # Horizontal stacked bar chart — one bar per scenario
        fig_scen, ax_scen = plt.subplots(figsize=(13, max(4, 0.3 * params.num_scenarios)),
                                         facecolor='white')
        y_pos = np.arange(params.num_scenarios)
        ax_scen.barh(y_pos, split_df['BoG_Pct'], color=C3, alpha=0.85,
                     label='BoG reserve %')
        ax_scen.barh(y_pos, split_df['Market_Pct'], left=split_df['BoG_Pct'],
                     color=C4, alpha=0.85, label='Market %')
        ax_scen.set_yticks(y_pos)
        ax_scen.set_yticklabels([f'Scenario {i}' for i in range(params.num_scenarios)])
        ax_scen.set_xlabel('Percentage of Total Release')
        ax_scen.set_xlim(0, 100)
        ax_scen.set_title('Release Split by Scenario (% of total release)',
                          fontsize=11, fontweight='bold')
        ax_scen.axvline(50, color='black', lw=0.8, ls=':', alpha=0.5)
        ax_scen.legend(fontsize=9, loc='lower right')
        ax_scen.grid(True, alpha=0.3, axis='x')
        ax_scen.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig_scen, use_container_width=True)

        buf_scen = io.BytesIO()
        fig_scen.savefig(buf_scen, format='png', dpi=180,
                         bbox_inches='tight', facecolor='white')
        buf_scen.seek(0)
        st.download_button(
            "⬇️ Download per-scenario split (PNG)",
            buf_scen,
            file_name=f"dp_scenario_split_{datetime.now():%Y%m%d_%H%M%S}.png",
            mime="image/png",
        )
        plt.close(fig_scen)

        # ─── Data table ───
        st.markdown("#### Scenario-Level Split Table")
        st.dataframe(
            split_df.style.format({
                'Total_Release_kg': '{:,.1f}',
                'BoG_Reserve_kg': '{:,.1f}',
                'Market_kg': '{:,.1f}',
                'BoG_Pct': '{:.1f}%',
                'Market_Pct': '{:.1f}%',
            }),
            hide_index=True,
            use_container_width=True,
        )

        # Download CSV
        st.download_button(
            "⬇️ Download split table (CSV)",
            split_df.to_csv(index=False),
            file_name=f"dp_bog_market_split_{datetime.now():%Y%m%d_%H%M%S}.csv",
            mime="text/csv",
        )

    # ═══ Results Table tab ═══
    with tabs[5]:
        st.markdown("### Weekly Schedule (mean across scenarios)")
        schedule_df = pd.DataFrame({
            'Week': t_ax,
            'Gold_Price_USD': gold_weekly,
            'Avg_FX': res['fx_mean'],
            'Avg_Purchase_kg': res['purch_mean'],
            'Avg_Release_kg': res['release_mean'],
            'Avg_Release_BOG_kg': res['rel_bog_mean'],
            'Avg_Release_Market_kg': res['rel_mkt_mean'],
            'Avg_Inventory_kg': res['inv_mean'],
            'Avg_CumPurchase_kg': res['cum_mean'],
            'Avg_Discrepancy_GHS': res['disc_mean'],
            'Lambda_GHS_per_kg': res['lambda_arr'],
        })
        st.dataframe(schedule_df.style.format({
            'Gold_Price_USD': '{:,.0f}',
            'Avg_FX': '{:.3f}',
            'Avg_Purchase_kg': '{:,.1f}',
            'Avg_Release_kg': '{:,.1f}',
            'Avg_Release_BOG_kg': '{:,.1f}',
            'Avg_Release_Market_kg': '{:,.1f}',
            'Avg_Inventory_kg': '{:,.1f}',
            'Avg_CumPurchase_kg': '{:,.1f}',
            'Avg_Discrepancy_GHS': '{:,.0f}',
            'Lambda_GHS_per_kg': '{:,.2f}',
        }), use_container_width=True, height=400)

        st.markdown("### Per-Scenario Summary")
        st.dataframe(df_scen.style.format({
            'Yearend_Inventory_kg': '{:,.1f}',
            'Total_Purchase_kg': '{:,.1f}',
            'Total_Release_kg': '{:,.1f}',
            'Total_Release_BOG_kg': '{:,.1f}',
            'Total_Release_Market_kg': '{:,.1f}',
            'Shortfall_kg': '{:,.1f}',
            'Shortfall_GHS': '{:,.0f}',
            'Total_Abs_Disc_GHS': '{:,.0f}',
            'Objective_GHS': '{:,.0f}',
        }), use_container_width=True, height=400)

    # ═══ Downloads tab ═══
    with tabs[6]:
        st.markdown("### CSV Downloads")

        # Schedule CSV
        schedule_df = pd.DataFrame({
            'Week': t_ax,
            'Gold_Price_USD': gold_weekly,
            'Avg_FX': res['fx_mean'],
            'Avg_Purchase_kg': res['purch_mean'],
            'Avg_Release_kg': res['release_mean'],
            'Avg_Release_BOG_kg': res['rel_bog_mean'],
            'Avg_Release_Market_kg': res['rel_mkt_mean'],
            'Avg_Inventory_kg': res['inv_mean'],
            'Avg_CumPurchase_kg': res['cum_mean'],
            'Avg_Discrepancy_GHS': res['disc_mean'],
            'Lambda_GHS_per_kg': res['lambda_arr'],
        })
        timestamp = f"{datetime.now():%Y%m%d_%H%M%S}"

        dl_cols = st.columns(2)
        with dl_cols[0]:
            st.download_button(
                "⬇️ Weekly schedule (mean)",
                schedule_df.to_csv(index=False),
                file_name=f"dp_schedule_{timestamp}.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.download_button(
                "⬇️ Per-scenario summary",
                df_scen.to_csv(index=False),
                file_name=f"dp_scenario_summary_{timestamp}.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with dl_cols[1]:
            df_inv_all = pd.DataFrame(
                res['inv_all'].T,
                columns=[f'Scenario_{i}' for i in range(params.num_scenarios)]
            )
            df_inv_all.insert(0, 'Period', t_ax)
            st.download_button(
                "⬇️ Inventory all scenarios",
                df_inv_all.to_csv(index=False),
                file_name=f"dp_inventory_all_{timestamp}.csv",
                mime="text/csv",
                use_container_width=True,
            )

            # Aggregate stats
            agg = pd.DataFrame.from_dict({
                'Expected_Objective_GHS': res['exp_obj'],
                'Mean_Yearend_Inventory_kg': df_scen['Yearend_Inventory_kg'].mean(),
                'Std_Yearend_Inventory_kg': df_scen['Yearend_Inventory_kg'].std(),
                'Mean_Total_Purchase_kg': df_scen['Total_Purchase_kg'].mean(),
                'Std_Total_Purchase_kg': df_scen['Total_Purchase_kg'].std(),
                'Mean_Total_Release_kg': df_scen['Total_Release_kg'].mean(),
                'Mean_Total_Release_BOG_kg': df_scen['Total_Release_BOG_kg'].mean(),
                'Mean_Total_Release_Market_kg': df_scen['Total_Release_Market_kg'].mean(),
                'Mean_Shortfall_kg': df_scen['Shortfall_kg'].mean(),
                'Std_Shortfall_kg': df_scen['Shortfall_kg'].std(),
                'Scenarios_Meeting_Q': int((df_scen['Shortfall_kg'] == 0).sum()),
                'Mean_Total_Disc_GHS': df_scen['Total_Abs_Disc_GHS'].mean(),
                'Std_Total_Disc_GHS': df_scen['Total_Abs_Disc_GHS'].std(),
                'Mean_Over_Supply_Periods': df_scen['Over_Supply_Periods'].mean(),
                'Mean_Under_Supply_Periods': df_scen['Under_Supply_Periods'].mean(),
                'N_Optimal_Scenarios': res['n_opt'],
                'N_Infeasible_Scenarios': res['n_inf'],
                'Solve_Time_s': res['solve_time'],
            }, orient='index', columns=['Value'])

            st.download_button(
                "⬇️ Aggregate statistics",
                agg.to_csv(),
                file_name=f"dp_aggregate_stats_{timestamp}.csv",
                mime="text/csv",
                use_container_width=True,
            )

        # Show settings used
        st.markdown("### Model Configuration")
        config = {
            "Gold forecast (USD/oz)": f"Q1={q1}, Q2={q2}, Q3={q3}, Q4={q4}",
            "FX range (GHS/USD)": f"U[{fx_low}, {fx_high}], threshold={fx_threshold}",
            "ASM volume (kg/week)": f"U[{v_min:,.0f}, {v_max:,.0f}]",
            "Q_ANNUAL (kg)": f"{q_annual:,.0f}",
            "Supply ratio ρ": rho,
            "Working buffer S̄ (kg)": s_bar,
            "Weights (α, β, γ, φ)": f"({alpha}, {beta}, {gamma}, {phi})",
            "Scenarios": num_scenarios,
            "Seed": seed,
            "DP grid": f"{inv_grid} × {x_grid} × {y_grid}",
            "Solve time (s)": f"{res['solve_time']:.2f}",
        }
        st.json(config)

else:
    st.info("👈 Configure inputs in the sidebar and click **Run DP Solver** to generate the schedule.")
    st.markdown(
        """
        ### How to use this tool

        1. **Enter gold price forecast** — JP Morgan or your own quarterly forecasts (USD/oz).
        2. **Set FX range** — exchange rate band is sampled uniformly for each scenario.
        3. **Set ASM volume range** — weekly small-scale miner supply availability.
        4. **Tune weights** — α (under-release penalty), β (over-release), γ (inventory reward),
           φ (shortfall penalty).
        5. **Click Run** — results include the optimal weekly schedule, policy maps,
           value function, and outcome distributions across scenarios.

        ### Model summary

        - **State**: inventory buffer sₜ ∈ [0, S̄]
        - **Decisions**: purchase xₜ, release yₜ at each week t
        - **Uncertainty**: FX rₜ ~ U[low, high], ASM volume Vₜ ~ U[min, max]
        - **Gold price**: fixed from quarterly forecasts (linearly interpolated to weekly)
        - **Solve method**: backward DP per scenario, expected-value aggregation
        """
    )
