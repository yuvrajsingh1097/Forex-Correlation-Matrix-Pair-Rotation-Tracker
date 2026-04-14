"""
Forex Correlation Matrix + Pair Rotation Tracker
=================================================
Quant tool that:
  1. Computes rolling correlation matrix across 8 major forex pairs
  2. Detects DIVERGENCE events — when historically correlated pairs
     temporarily decouple (z-score of spread diverges > threshold)
  3. Tracks pair rotation momentum — which pairs are leading vs lagging
  4. Visualises: Correlation Heatmap | Divergence Chart |
                 Rotation Radar | Rolling Corr Timeline

Divergence Edge:
  When EURUSD and GBPUSD (typically 0.85+ corr) diverge sharply,
  one is mispriced relative to the other → mean-reversion opportunity.

Author : Your Name
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────
PAIRS = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD",
         "USDCAD", "USDCHF", "USDJPY", "EURGBP"]

CORR_WINDOW        = 30    # days for rolling correlation
DIV_ZSCORE_THRESH  = 2.0   # z-score threshold to flag divergence
SPREAD_WINDOW      = 20    # lookback for spread z-score
ROTATION_WINDOW    = 10    # days for momentum rotation
N_DAYS             = 180   # total simulation days

# Base correlations (realistic forex market structure)
TRUE_CORR = {
    ("EURUSD", "GBPUSD"): 0.87,
    ("EURUSD", "AUDUSD"): 0.72,
    ("EURUSD", "NZDUSD"): 0.68,
    ("EURUSD", "USDCAD"): -0.75,
    ("EURUSD", "USDCHF"): -0.91,
    ("EURUSD", "USDJPY"): -0.45,
    ("EURUSD", "EURGBP"):  0.55,
    ("GBPUSD", "AUDUSD"):  0.71,
    ("GBPUSD", "NZDUSD"):  0.65,
    ("GBPUSD", "USDCAD"): -0.72,
    ("GBPUSD", "USDCHF"): -0.83,
    ("GBPUSD", "USDJPY"): -0.40,
    ("GBPUSD", "EURGBP"): -0.20,
    ("AUDUSD", "NZDUSD"):  0.92,
    ("AUDUSD", "USDCAD"): -0.65,
    ("AUDUSD", "USDCHF"): -0.68,
    ("AUDUSD", "USDJPY"): -0.30,
    ("NZDUSD", "USDCAD"): -0.60,
    ("USDCAD", "USDCHF"):  0.70,
    ("USDCAD", "USDJPY"):  0.55,
    ("USDCHF", "USDJPY"):  0.60,
}

# ─────────────────────────────────────────────────────
#  1. GENERATE CORRELATED FOREX RETURNS
# ─────────────────────────────────────────────────────
def build_corr_matrix():
    n = len(PAIRS)
    C = np.eye(n)
    for (p1, p2), c in TRUE_CORR.items():
        if p1 in PAIRS and p2 in PAIRS:
            i, j = PAIRS.index(p1), PAIRS.index(p2)
            C[i, j] = c
            C[j, i] = c
    # Ensure positive semi-definite
    eigvals = np.linalg.eigvals(C)
    if np.any(eigvals < 0):
        C += np.eye(n) * (abs(eigvals.min()) + 0.01)
        D = np.diag(1 / np.sqrt(np.diag(C)))
        C = D @ C @ D
    return C

def generate_forex_returns(n_days=180, seed=99):
    np.random.seed(seed)
    C   = build_corr_matrix()
    L   = np.linalg.cholesky(C)
    raw = np.random.normal(0, 1, (n_days, len(PAIRS)))
    corr_returns = raw @ L.T

    vols = {
        "EURUSD": 0.0060, "GBPUSD": 0.0072, "AUDUSD": 0.0065,
        "NZDUSD": 0.0070, "USDCAD": 0.0055, "USDCHF": 0.0058,
        "USDJPY": 0.0065, "EURGBP": 0.0045,
    }
    starts = {
        "EURUSD": 1.0850, "GBPUSD": 1.2650, "AUDUSD": 0.6520,
        "NZDUSD": 0.6020, "USDCAD": 1.3580, "USDCHF": 0.9050,
        "USDJPY": 149.50, "EURGBP": 0.8580,
    }

    dates = pd.date_range("2024-01-01", periods=n_days, freq="1D")
    prices = {}
    for k, pair in enumerate(PAIRS):
        vol   = vols[pair]
        rets  = corr_returns[:, k] * vol
        # Inject a divergence event for EURUSD/GBPUSD around day 80-100
        if pair == "GBPUSD":
            rets[80:95] += 0.0035
        price = [starts[pair]]
        for r in rets[1:]:
            price.append(round(price[-1] * (1 + r), 5))
        prices[pair] = price

    return pd.DataFrame(prices, index=dates)

# ─────────────────────────────────────────────────────
#  2. ROLLING CORRELATION MATRIX
# ─────────────────────────────────────────────────────
def compute_rolling_corr(prices_df, window=30):
    returns = prices_df.pct_change().dropna()
    rolling_corrs = {}
    for i in range(window, len(returns) + 1):
        window_ret = returns.iloc[i - window: i]
        corr = window_ret.corr()
        rolling_corrs[returns.index[i - 1]] = corr
    return rolling_corrs

def get_latest_corr(rolling_corrs):
    last_key = list(rolling_corrs.keys())[-1]
    return rolling_corrs[last_key]

# ─────────────────────────────────────────────────────
#  3. DIVERGENCE DETECTION
# ─────────────────────────────────────────────────────
def detect_divergences(prices_df, pair1="EURUSD", pair2="GBPUSD",
                        spread_window=20, z_thresh=2.0):
    p1 = prices_df[pair1]
    p2 = prices_df[pair2]

    # Normalise both to index = 100 at start
    p1_n = p1 / p1.iloc[0] * 100
    p2_n = p2 / p2.iloc[0] * 100

    spread = p1_n - p2_n
    roll_mean = spread.rolling(spread_window).mean()
    roll_std  = spread.rolling(spread_window).std()
    z_score   = (spread - roll_mean) / roll_std.replace(0, np.nan)

    divergences = []
    in_div = False
    for i in range(len(z_score)):
        z = z_score.iloc[i]
        if not np.isnan(z) and abs(z) >= z_thresh and not in_div:
            divergences.append({
                "time"   : z_score.index[i],
                "z_score": round(z, 2),
                "spread" : round(spread.iloc[i], 4),
                "type"   : "SHORT Spread" if z > 0 else "Long Spread",
            })
            in_div = True
        elif not np.isnan(z) and abs(z) < 0.5:
            in_div = False

    return z_score, spread, roll_mean, roll_std, pd.DataFrame(divergences)

# ─────────────────────────────────────────────────────
#  4. PAIR ROTATION (MOMENTUM RANKING)
# ─────────────────────────────────────────────────────
def compute_rotation(prices_df, window=10):
    returns = prices_df.pct_change(window).iloc[-1]
    ranked  = returns.rank(ascending=False)
    return returns.round(4), ranked.astype(int)

# ─────────────────────────────────────────────────────
#  5. ROLLING CORR TIMELINE FOR TOP PAIRS
# ─────────────────────────────────────────────────────
def rolling_corr_series(prices_df, pair1, pair2, window=30):
    r1 = prices_df[pair1].pct_change()
    r2 = prices_df[pair2].pct_change()
    return r1.rolling(window).corr(r2)

# ─────────────────────────────────────────────────────
#  6. PLOT
# ─────────────────────────────────────────────────────
def plot_all(prices_df, rolling_corrs, z_score, spread,
             roll_mean, div_events, mom_returns, mom_ranks):

    fig = plt.figure(figsize=(22, 15))
    fig.patch.set_facecolor("#0d1117")

    gs = gridspec.GridSpec(3, 3, hspace=0.42, wspace=0.35,
                           height_ratios=[1.4, 1.2, 1.2])

    ax_heat   = fig.add_subplot(gs[0, :2])   # Correlation heatmap (wide)
    ax_rot    = fig.add_subplot(gs[0, 2])    # Rotation bar chart
    ax_div    = fig.add_subplot(gs[1, :])    # Divergence z-score
    ax_rc1    = fig.add_subplot(gs[2, 0])    # Rolling corr EURUSD/GBPUSD
    ax_rc2    = fig.add_subplot(gs[2, 1])    # Rolling corr AUDUSD/NZDUSD
    ax_rc3    = fig.add_subplot(gs[2, 2])    # Rolling corr USDCAD/USDCHF

    for ax in [ax_heat, ax_rot, ax_div, ax_rc1, ax_rc2, ax_rc3]:
        ax.set_facecolor("#0d1117")
        for sp in ax.spines.values():
            sp.set_edgecolor("#30363d")
        ax.tick_params(colors="#8b949e", labelsize=8)

    # ── HEATMAP ──
    corr_mat = get_latest_corr(rolling_corrs)
    n = len(PAIRS)
    cmap = plt.get_cmap("RdYlGn")

    for i in range(n):
        for j in range(n):
            val = corr_mat.iloc[i, j]
            color = cmap((val + 1) / 2)
            rect = mpatches.FancyBboxPatch(
                (j + 0.05, n - i - 1 + 0.05), 0.90, 0.90,
                boxstyle="round,pad=0.05",
                facecolor=color, edgecolor="#0d1117", lw=0.8
            )
            ax_heat.add_patch(rect)
            txt_col = "black" if 0.3 < (val + 1) / 2 < 0.85 else "white"
            ax_heat.text(j + 0.5, n - i - 0.5, f"{val:.2f}",
                         ha="center", va="center", fontsize=7.5,
                         color=txt_col, fontweight="bold")

    ax_heat.set_xlim(0, n)
    ax_heat.set_ylim(0, n)
    ax_heat.set_xticks(np.arange(n) + 0.5)
    ax_heat.set_yticks(np.arange(n) + 0.5)
    ax_heat.set_xticklabels(PAIRS, rotation=30, ha="right",
                             color="#e6edf3", fontsize=8.5)
    ax_heat.set_yticklabels(reversed(PAIRS), color="#e6edf3", fontsize=8.5)
    ax_heat.set_title(f"Rolling {CORR_WINDOW}-Day Correlation Matrix  |  8 Major Pairs",
                      color="#e6edf3", fontsize=11, fontweight="bold", pad=8)

    # Colorbar manually
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-1, 1))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax_heat, fraction=0.025, pad=0.02)
    cb.ax.tick_params(colors="#8b949e", labelsize=7)
    cb.set_label("Correlation", color="#8b949e", fontsize=8)

    # ── ROTATION BAR ──
    colors_rot = ["#26a69a" if r > 0 else "#ef5350" for r in mom_returns.values]
    bars = ax_rot.barh(PAIRS, mom_returns.values * 100,
                       color=colors_rot, edgecolor="#30363d", height=0.6)
    ax_rot.axvline(0, color="#8b949e", lw=0.8)
    for bar, rank in zip(bars, [mom_ranks[p] for p in PAIRS]):
        w = bar.get_width()
        ax_rot.text(w + (0.03 if w >= 0 else -0.03),
                    bar.get_y() + bar.get_height() / 2,
                    f"#{rank}", va="center", ha="left" if w >= 0 else "right",
                    color="#e6edf3", fontsize=7.5, fontweight="bold")
    ax_rot.set_title(f"{ROTATION_WINDOW}D Momentum Rotation",
                     color="#e6edf3", fontsize=10, fontweight="bold", pad=8)
    ax_rot.set_xlabel("Return %", color="#8b949e", fontsize=8)
    ax_rot.yaxis.set_tick_params(colors="#e6edf3")
    ax_rot.grid(color="#21262d", ls="--", lw=0.5, axis="x")
    ax_rot.set_xlim(mom_returns.values.min() * 120,
                    mom_returns.values.max() * 150)

    # ── DIVERGENCE Z-SCORE ──
    xs = np.arange(len(z_score.dropna()))
    zv = z_score.dropna().values
    xi = z_score.dropna().index

    ax_div.plot(xs, zv, color="#7c8cf8", lw=1.5, zorder=3, label="Z-Score")
    ax_div.axhline( DIV_ZSCORE_THRESH, color="#ef5350", lw=1, ls="--",
                    alpha=0.8, label=f"+{DIV_ZSCORE_THRESH}σ threshold")
    ax_div.axhline(-DIV_ZSCORE_THRESH, color="#26a69a", lw=1, ls="--",
                    alpha=0.8, label=f"-{DIV_ZSCORE_THRESH}σ threshold")
    ax_div.axhline(0, color="#8b949e", lw=0.6, ls=":")
    ax_div.fill_between(xs, DIV_ZSCORE_THRESH,  zv,
                        where=zv >=  DIV_ZSCORE_THRESH,
                        color="#ef5350", alpha=0.25, label="Short Spread Zone")
    ax_div.fill_between(xs, -DIV_ZSCORE_THRESH, zv,
                        where=zv <= -DIV_ZSCORE_THRESH,
                        color="#26a69a", alpha=0.25, label="Long Spread Zone")

    # Divergence event markers
    for _, ev in div_events.iterrows():
        if ev["time"] in xi:
            xi_pos = list(xi).index(ev["time"])
            ax_div.scatter(xi_pos, ev["z_score"],
                           color="#ffe066", s=90, zorder=5)
            ax_div.annotate(
                f"DIV\n{ev['z_score']}σ",
                xy=(xi_pos, ev["z_score"]),
                xytext=(xi_pos + 3, ev["z_score"] + 0.3 * np.sign(ev["z_score"])),
                color="#ffe066", fontsize=7, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color="#ffe066", lw=0.8)
            )

    tick_step = max(1, len(xi) // 8)
    ax_div.set_xticks(np.arange(0, len(xi), tick_step))
    ax_div.set_xticklabels(
        [xi[i].strftime("%b %d") for i in range(0, len(xi), tick_step)],
        rotation=20, ha="right", color="#8b949e", fontsize=8
    )
    ax_div.set_title("EURUSD / GBPUSD Spread Divergence  (Z-Score)",
                     color="#e6edf3", fontsize=11, fontweight="bold", pad=8)
    ax_div.set_ylabel("Z-Score (σ)", color="#8b949e", fontsize=9)
    ax_div.legend(facecolor="#161b22", edgecolor="#30363d",
                  labelcolor="#e6edf3", fontsize=7.5, loc="upper left", ncol=3)
    ax_div.grid(color="#21262d", ls="--", lw=0.5)

    # ── ROLLING CORR PANELS ──
    pair_combos = [
        ("EURUSD", "GBPUSD", "#7c8cf8"),
        ("AUDUSD", "NZDUSD", "#26a69a"),
        ("USDCAD", "USDCHF", "#f0a040"),
    ]
    for ax_rc, (p1, p2, col) in zip([ax_rc1, ax_rc2, ax_rc3], pair_combos):
        rc = rolling_corr_series(prices_df, p1, p2, CORR_WINDOW).dropna()
        rc_xs = np.arange(len(rc))
        ax_rc.plot(rc_xs, rc.values, color=col, lw=1.5)
        ax_rc.axhline(0, color="#8b949e", lw=0.5, ls=":")
        ax_rc.fill_between(rc_xs, 0, rc.values,
                           where=rc.values >= 0, color=col, alpha=0.12)
        ax_rc.fill_between(rc_xs, 0, rc.values,
                           where=rc.values < 0,  color="#ef5350", alpha=0.12)
        ax_rc.set_ylim(-1.1, 1.1)
        ax_rc.set_title(f"{p1} / {p2}", color="#e6edf3",
                        fontsize=9.5, fontweight="bold", pad=6)
        ax_rc.set_ylabel("Correlation", color="#8b949e", fontsize=8)
        ax_rc.grid(color="#21262d", ls="--", lw=0.5)
        # current value badge
        cur = rc.values[-1]
        badge_col = "#26a69a" if cur >= 0 else "#ef5350"
        ax_rc.text(0.97, 0.93, f"Now: {cur:.2f}",
                   transform=ax_rc.transAxes, ha="right", va="top",
                   color=badge_col, fontsize=8.5, fontweight="bold",
                   bbox=dict(fc="#161b22", ec=badge_col, lw=0.8,
                             boxstyle="round,pad=0.3"))
        tick_s = max(1, len(rc) // 5)
        ax_rc.set_xticks(rc_xs[::tick_s])
        ax_rc.set_xticklabels(
            [rc.index[i].strftime("%b %d") for i in range(0, len(rc), tick_s)],
            rotation=20, ha="right", color="#8b949e", fontsize=7
        )

    fig.suptitle("Forex Correlation Matrix + Pair Rotation Tracker  —  8 Major Pairs",
                 color="#e6edf3", fontsize=14, fontweight="bold", y=0.98)

    plt.savefig("output.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.show()
    print("✅  Chart saved → output.png")


# ─────────────────────────────────────────────────────
#  CONSOLE SUMMARY
# ─────────────────────────────────────────────────────
def print_summary(latest_corr, div_events, mom_returns, mom_ranks):
    print("\n" + "=" * 60)
    print("  Forex Correlation + Rotation Tracker — Summary")
    print("=" * 60)

    print(f"\n  📊 Current Rolling {CORR_WINDOW}D Correlations (top pairs):")
    pairs_to_show = [("EURUSD", "GBPUSD"), ("AUDUSD", "NZDUSD"),
                     ("USDCAD", "USDCHF"), ("EURUSD", "USDCHF")]
    for p1, p2 in pairs_to_show:
        if p1 in latest_corr.columns and p2 in latest_corr.columns:
            c = latest_corr.loc[p1, p2]
            bar = "█" * int(abs(c) * 15)
            print(f"  {p1}/{p2:8s}  {c:+.3f}  {bar}")

    print(f"\n  ⚡ Divergence Events Detected : {len(div_events)}")
    if not div_events.empty:
        print(div_events[["time", "z_score", "type"]].to_string(index=False))

    print(f"\n  🔄 Pair Rotation Ranking ({ROTATION_WINDOW}D momentum):")
    sorted_pairs = mom_returns.sort_values(ascending=False)
    for rank, (pair, ret) in enumerate(sorted_pairs.items(), 1):
        arrow = "▲" if ret > 0 else "▼"
        print(f"  #{rank}  {pair:8s}  {arrow}  {ret*100:+.3f}%")
    print("=" * 60)


# ─────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("📊  Generating correlated forex price data (8 pairs, 180 days)...")
    prices_df = generate_forex_returns(n_days=N_DAYS, seed=99)

    print("🔗  Computing rolling correlation matrices...")
    rolling_corrs = compute_rolling_corr(prices_df, CORR_WINDOW)
    latest_corr   = get_latest_corr(rolling_corrs)

    print("📉  Detecting EURUSD/GBPUSD spread divergences...")
    z_score, spread, roll_mean, roll_std, div_events = detect_divergences(
        prices_df, "EURUSD", "GBPUSD",
        SPREAD_WINDOW, DIV_ZSCORE_THRESH
    )

    print("🔄  Computing pair rotation momentum...")
    mom_returns, mom_ranks = compute_rotation(prices_df, ROTATION_WINDOW)

    print_summary(latest_corr, div_events, mom_returns, mom_ranks)

    print("\n🎨  Plotting full dashboard...")
    plot_all(prices_df, rolling_corrs, z_score, spread,
             roll_mean, div_events, mom_returns, mom_ranks)