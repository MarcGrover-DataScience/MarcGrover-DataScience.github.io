"""
Bootstrap Confidence Intervals - Part B: Coverage Simulation Study
==================================================================
Question
  Part A built bootstrap intervals from a single pilot sample. A single sample
  cannot show whether the METHOD is trustworthy: an interval either happens to
  contain the truth or it does not. This script answers the question properly
  by repeating the exercise thousands of times on data where the true value is
  known by construction, and counting how often a nominal 95% interval really
  does contain it ("empirical coverage").

Scenarios (true parameter known exactly)
  - Normal(0, 1)             -> mean, median   well-behaved baseline
  - Lognormal(0, 1)          -> mean, median   skewed, heavy right tail
  - Uniform(0, 1)            -> sample maximum known bootstrap failure; true max = 1
  - Lognormal(0, 1.5)        -> mean, median   heavier skew (added in v2)

Interval methods
  Percentile, Basic, BCa, and Studentised (bootstrap-t, mean only)

Stages
  1. Configuration
  2. Scenarios and statistics
  3. Confidence-interval engine
  4. Simulation loop
  5. Summary tables
  6. Charts 06-08   (numbering continues from Part A's charts 01-05)

v2 changes from the first version of this script
  - Adds the heavier-skew Lognormal(0, 1.5) scenario. It is appended LAST in the
    scenario list so the original scenarios consume the random stream exactly as
    before and reproduce the first version's results.
  - Sectioned and commented to portfolio conventions; "Studentised" spelling.
  - Charts numbered 06-08; coverage figure widened to a 2 x 4 grid; the tail-miss
    chart now shows both lognormal skew levels side by side.

Usage:  python bootstrap_simulation_v2.py [R] [B]
        R = simulation replications per setting (default 1000)
        B = bootstrap resamples per replication (default 999)
Runs on CPU only. Outputs go to ./bootstrap_sim_v2_output/
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # save charts to file; remove this line for interactive windows
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm


# ======================================================================
# 1. CONFIGURATION
# ======================================================================
R = int(sys.argv[1]) if len(sys.argv) > 1 else 1000   # replications per setting
B = int(sys.argv[2]) if len(sys.argv) > 2 else 999    # resamples per replication
# R = 1,000 gives a Monte Carlo standard error of about 0.7 percentage points on a
# coverage of 95%, so differences smaller than roughly +/-1.4 points are noise.
# B = 999 (rather than Part A's 9,999) because the loop below builds hundreds of
# thousands of intervals; 999 also puts the 2.5% and 97.5% positions on whole
# order statistics, since (B + 1) * 0.025 = 25.

ALPHA = 0.05                  # nominal 95% intervals
NS = [10, 20, 50, 100, 200]   # sample sizes, from "too small" to "comfortable"
SEED = 2026
OUT = Path(__file__).resolve().parent / "bootstrap_sim_v2_output"
OUT.mkdir(exist_ok=True)

# One shared random stream. Fixing the seed makes every run identical, and keeping
# a single stream (rather than one per scenario) is what lets v2 reproduce the
# first version's numbers for the original scenarios.
rng = np.random.default_rng(SEED)

sns.set_theme(style="whitegrid", context="notebook")
# Same colours as Part A so each method keeps its identity across the whole page
COLOURS = {"Percentile": "#1f77b4", "Basic": "#ff7f0e", "BCa": "#2ca02c", "Studentised": "#d62728"}
MARKERS = {"Percentile": "o", "Basic": "s", "BCa": "^", "Studentised": "D"}
MC_SE = np.sqrt(0.95 * 0.05 / R)   # Monte Carlo standard error of a coverage near 95%


# ======================================================================
# 2. SCENARIOS AND STATISTICS
# ======================================================================
# Each scenario supplies a data generator and the TRUE value of every statistic
# being estimated. Knowing the truth exactly is the whole point of a simulation
# study: it is what allows "did the interval contain it?" to be answered.
# Lognormal(0, sigma): median = exp(0) = 1 and mean = exp(sigma^2 / 2).
SCENARIOS = {
    "Normal": dict(draw=lambda n: rng.normal(0, 1, n),
                   truth={"mean": 0.0, "median": 0.0}),
    "Lognormal (sigma=1)": dict(draw=lambda n: rng.lognormal(0, 1, n),
                                truth={"mean": np.exp(0.5), "median": 1.0}),
    "Uniform": dict(draw=lambda n: rng.uniform(0, 1, n),
                    truth={"max": 1.0}),
    # Appended last on purpose (see the docstring): heavier skew, so the mean's
    # sampling distribution is far from normal and the t-style intervals struggle most.
    "Lognormal (sigma=1.5)": dict(draw=lambda n: rng.lognormal(0, 1.5, n),
                                  truth={"mean": np.exp(1.125), "median": 1.0}),
}

# Statistics act on a 2-D array (rows = resamples) so that all B resamples are
# evaluated in one vectorised call; this is what keeps the study CPU-feasible.
STAT_FUNCS = {
    "mean": lambda a: a.mean(axis=1),
    "median": lambda a: np.median(a, axis=1),
    "max": lambda a: a.max(axis=1),
}


# ======================================================================
# 3. CONFIDENCE-INTERVAL ENGINE
# ======================================================================
def bootstrap_cis(x, stat, rng):
    """Return {method: (lower, upper)} for one sample and one statistic."""
    n = x.size
    f = STAT_FUNCS[stat]
    theta = f(x[None, :])[0]                         # the sample estimate
    xb = x[rng.integers(0, n, size=(B, n))]          # B resamples drawn with replacement
    boot = f(xb)                                     # the statistic on every resample
    cis = {}

    # Percentile: take the middle 95% of the bootstrap distribution directly.
    lo, hi = np.quantile(boot, [ALPHA / 2, 1 - ALPHA / 2])
    cis["Percentile"] = (lo, hi)

    # Basic (reverse percentile): reflect the percentile interval about the
    # estimate. It assumes the bootstrap error (boot - theta) mimics the true
    # error (theta - truth), which fails when the statistic is skewed or discrete.
    cis["Basic"] = (2 * theta - hi, 2 * theta - lo)

    # BCa: shifts the percentile cut-points to correct for bias (z0) and skewness
    # (acceleration a, estimated by the jackknife).
    p0 = np.mean(boot < theta) + 0.5 * np.mean(boot == theta)   # mid-rank for ties
    p0 = np.clip(p0, 1 / (B + 1), B / (B + 1))                  # keep norm.ppf finite
    z0 = norm.ppf(p0)
    # Leave-one-out samples: n rows, each with one observation removed
    loo = np.broadcast_to(x, (n, n))[~np.eye(n, dtype=bool)].reshape(n, n - 1)
    jack = f(loo)
    d = jack.mean() - jack
    denom = 6.0 * (np.sum(d ** 2) ** 1.5)
    a = np.sum(d ** 3) / denom if denom > 0 else 0.0            # guard: constant jackknife
    za = norm.ppf([ALPHA / 2, 1 - ALPHA / 2])
    adj = norm.cdf(z0 + (z0 + za) / (1 - a * (z0 + za)))        # adjusted percentile positions
    cis["BCa"] = tuple(np.quantile(boot, adj))

    # Studentised (bootstrap-t): only coded for the mean, where every resample has
    # a cheap standard error. Pivoting on t = (boot - theta) / se_boot is what
    # lets it cope with skewness better than the other three.
    if stat == "mean":
        se_hat = x.std(ddof=1) / np.sqrt(n)
        se_b = xb.std(axis=1, ddof=1) / np.sqrt(n)
        t = (boot - theta) / se_b
        t_lo, t_hi = np.quantile(t, [ALPHA / 2, 1 - ALPHA / 2])
        cis["Studentised"] = (theta - t_hi * se_hat, theta - t_lo * se_hat)
    return cis


# ======================================================================
# 4. SIMULATION LOOP
# ======================================================================
# For every scenario x sample size x replication: draw a fresh sample, build every
# interval, and record whether it contained the truth, and which way it missed.
# The direction matters: a method can have acceptable overall error while being
# badly one-sided, which is what "miss_above" / "miss_below" expose.
records = []
t0 = time.time()
print(f"Simulation: R = {R}, B = {B}, sample sizes {NS}")
for dist, cfg in SCENARIOS.items():
    for n in NS:
        for _ in range(R):
            x = cfg["draw"](n)
            for stat, truth in cfg["truth"].items():
                for method, (lo, hi) in bootstrap_cis(x, stat, rng).items():
                    records.append((dist, stat, n, method,
                                    lo <= truth <= hi,   # covered
                                    truth < lo,          # interval entirely ABOVE the truth
                                    truth > hi,          # interval entirely BELOW the truth
                                    hi - lo))            # width
        print(f"{dist:22s} n={n:<4d} done  ({time.time() - t0:5.0f}s)", flush=True)


# ======================================================================
# 5. SUMMARY TABLES
# ======================================================================
df = pd.DataFrame(records, columns=["dist", "stat", "n", "method",
                                    "covered", "miss_above", "miss_below", "width"])
df["scenario"] = df["dist"] + " - " + df["stat"]
summary = (df.groupby(["scenario", "n", "method"])
             .agg(coverage=("covered", "mean"),
                  miss_above=("miss_above", "mean"),
                  miss_below=("miss_below", "mean"),
                  mean_width=("width", "mean"))
             .reset_index())
summary.to_csv(OUT / "coverage_summary.csv", index=False)
print("\nCoverage table (nominal 95%):")
print(summary.pivot_table(index=["scenario", "method"], columns="n", values="coverage")
      .round(3).to_string())


# ======================================================================
# 6. CHARTS
# ======================================================================
# --- Chart 06: coverage by scenario, method and sample size ---
# Display order is chosen for the story (well-behaved -> skewed -> heavily skewed ->
# failure), which is different from the order the scenarios were simulated in.
scen = ["Normal - mean", "Normal - median",
        "Lognormal (sigma=1) - mean", "Lognormal (sigma=1) - median",
        "Lognormal (sigma=1.5) - mean", "Lognormal (sigma=1.5) - median",
        "Uniform - max"]
fig, axes = plt.subplots(2, 4, figsize=(20, 9))
axes = axes.ravel()
for ax, s in zip(axes, scen):
    sub = summary[summary.scenario == s]
    for m, g in sub.groupby("method"):
        ax.plot(g.n, g.coverage, marker=MARKERS[m], color=COLOURS[m], label=m, lw=1.8)
    ax.axhline(0.95, color="k", ls="--", lw=1)
    # Grey band = how far an exactly-correct method could stray by chance alone
    ax.axhspan(0.95 - 1.96 * MC_SE, 0.95 + 1.96 * MC_SE, color="grey", alpha=0.2)
    ax.set_xscale("log")
    ax.set_xticks(NS)
    ax.set_xticklabels(NS)
    ax.set_title(s)
    ax.set_xlabel("sample size n")
    ax.set_ylabel("empirical coverage")
    # Floor at 0.8 unless a method dips lower, so the interesting region stays visible
    ax.set_ylim(max(0, min(0.8, sub.coverage.min() - 0.03)), 1.0)
axes[-1].axis("off")   # the eighth panel holds the legend and notes instead of data
h, l = axes[0].get_legend_handles_labels()
axes[-1].legend(h, l, loc="center", bbox_to_anchor=(0.5, 0.68), fontsize=12, title="CI method")
axes[-1].text(0.5, 0.18, f"Dashed = nominal 95%\nGrey band = Monte Carlo error (R={R})\n"
                         f"B={B} resamples\nStudentised: mean only",
              ha="center", va="center", transform=axes[-1].transAxes)
# BCa and Percentile both sit at exactly 0% for the maximum, so one line hides the other
axes[6].text(0.5, 0.1, "BCa and Percentile both at 0%\n(lines overlap)", ha="center",
             transform=axes[6].transAxes, fontsize=9)
fig.suptitle("Coverage of 95% bootstrap confidence intervals", fontsize=14)
fig.tight_layout()
fig.savefig(OUT / "06_coverage_by_scenario.png", dpi=140)
plt.close(fig)

# --- Chart 07: interval width, and which side the misses fall on ---
fig, (axw, axm1, axm2) = plt.subplots(1, 3, figsize=(19, 5.2))

# Left: how wide the intervals are. Coverage is only half the story, because a
# method can reach nominal coverage simply by being very wide.
sub = summary[summary.scenario == "Lognormal (sigma=1) - mean"]
for m, g in sub.groupby("method"):
    axw.plot(g.n, g.mean_width, marker=MARKERS[m], color=COLOURS[m], label=m, lw=1.8)
axw.set_xscale("log"); axw.set_yscale("log")
axw.set_xticks(NS); axw.set_xticklabels(NS)
axw.set_xlabel("sample size n"); axw.set_ylabel("mean CI width")
axw.set_title("Lognormal (sigma=1) mean: average interval width")
axw.legend()

# Middle and right: direction of the errors at n = 20, for both skew levels. With
# skewed data the misses are strongly lopsided: intervals tend to sit BELOW the
# truth, because small samples rarely contain the rare very large values that
# dominate the true mean.
n_show = 20
for ax, scenario in [(axm1, "Lognormal (sigma=1) - mean"), (axm2, "Lognormal (sigma=1.5) - mean")]:
    sub = summary[(summary.scenario == scenario) & (summary.n == n_show)].reset_index(drop=True)
    x = np.arange(len(sub)); w = 0.38
    ax.bar(x - w / 2, sub.miss_below, w, color="#9467bd",
           label="interval entirely BELOW truth (misses high)")
    ax.bar(x + w / 2, sub.miss_above, w, color="#8c564b",
           label="interval entirely ABOVE truth (misses low)")
    ax.axhline(ALPHA / 2, color="k", ls="--", lw=1, label="ideal 2.5% per tail")
    ax.set_xticks(x); ax.set_xticklabels(sub.method)
    ax.set_ylabel("miss rate")
    ax.set_title(f"{scenario.replace(' - mean', '')} mean, n={n_show}: which side do errors fall on?")
axm1.legend(fontsize=8)
# Same vertical scale on both panels so the two skew levels compare fairly
ymax = max(axm1.get_ylim()[1], axm2.get_ylim()[1])
axm1.set_ylim(0, ymax); axm2.set_ylim(0, ymax)
fig.tight_layout()
fig.savefig(OUT / "07_width_and_tail_misses.png", dpi=140)
plt.close(fig)

# --- Chart 08: why the bootstrap fails for the sample maximum ---
# A single illustrative sample, with its own fixed seeds so this chart does not
# depend on the simulation above. The bootstrap can only ever return values already
# in the sample, so its maximum can never exceed the sample maximum, which itself
# always falls short of the true maximum of 1.
n = 50
x = np.random.default_rng(7).uniform(0, 1, n)
big = x[np.random.default_rng(8).integers(0, n, size=(20000, n))].max(axis=1)
frac = np.mean(big == x.max())     # share of resamples that simply reproduce the sample maximum
fig, ax = plt.subplots(figsize=(8, 4.8))
ax.hist(big, bins=60, color="#1f77b4", alpha=0.8)
ax.axvline(1.0, color="red", lw=2, label="true maximum = 1")
ax.axvline(x.max(), color="k", ls="--", lw=1.5, label=f"sample max = {x.max():.3f}")
ax.set_title(f"Bootstrap distribution of the max (Uniform(0,1), n={n})\n"
             f"{frac:.1%} of resamples equal the sample max (theory: {1 - (1 - 1 / n) ** n:.1%})")
ax.set_xlabel("bootstrap maximum"); ax.set_ylabel("count"); ax.legend()
fig.tight_layout()
fig.savefig(OUT / "08_bootstrap_max_failure.png", dpi=140)
plt.close(fig)

print(f"\nSaved charts and CSV to {OUT}  (total {time.time() - t0:.0f}s)")
