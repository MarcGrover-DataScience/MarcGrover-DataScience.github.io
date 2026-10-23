"""
Bootstrap Confidence Intervals - Part A: Health Insurance Pricing Pilot
=======================================================================
Business question
  A health insurer is designing a new regional product and has a small pilot
  sample of policyholders. The head of underwriting needs defensible
  uncertainty ranges for three quantities that drive pricing decisions:
    1. the MEDIAN annual charge          (the "typical" cost)
    2. the SMOKER : NON-SMOKER RATIO of mean charges (premium loading)
    3. the 90th PERCENTILE charge        (stop-loss / reinsurance threshold)
  The mean charge is also analysed, as the one case where a textbook
  t-interval exists and can be compared with the bootstrap.

Dataset
  Medical Cost Personal Dataset (insurance.csv), 1,338 rows, 7 columns.
  Source: https://www.kaggle.com/datasets/mirichoi0218/insurance
  Simulated and census-derived, NOT real claims data - results illustrate the
  technique and say nothing about real insurance pricing.

Design
  The full dataset is treated as a known "population" so that every interval
  can be checked against a true value, which a real pilot never allows.
  A pilot sample is drawn from it and bootstrap intervals are built from the
  pilot alone.

Stages
  1. Configuration
  2. Load, validate and profile                      -> charts 01, 02
  3. Statistics and bootstrap engine (functions)
  4. Population ("known truth") values
  5. Draw the pilot sample
  6. Textbook baseline: t-interval for the mean
  7. Bootstrap intervals on the pilot
  8. Cross-check against scipy.stats.bootstrap
  9. Results against the population values            -> charts 03, 04
 10. Pilot-size sweep: is the pilot big enough?       -> chart 05

Usage:  python bootstrap_pricing_pilot.py
Needs insurance.csv in the same folder as this script. CPU only.
Outputs go to ./bootstrap_pricing_output/
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # save charts to file; remove this line for interactive windows
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator, NullFormatter
import seaborn as sns
from scipy import stats
from scipy.stats import norm


# ======================================================================
# 1. CONFIGURATION
# ======================================================================
SEED = 2026
ALPHA = 0.05            # 95% intervals

PILOT_N = 100           # pilot size; gives roughly 20 smokers, enough to show
                        # the ratio's instability without being hopeless
B_MAIN = 9999           # resamples for the main analysis. B + 1 = 10,000 makes
                        # (B + 1) * 0.025 = 250 exactly, so the 2.5% and 97.5%
                        # percentile positions fall on whole order statistics
B_SWEEP = 999          # resamples inside the 600 repeated pilots of stage 10:
                        # cheaper, and Part B uses the same value
SWEEP_NS = [50, 100, 200]
SWEEP_REPS = 200        # repeated pilot draws per size
MIN_PER_GROUP = 2       # a ratio needs at least two observations in each group

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "insurance.csv"
OUT = HERE / "bootstrap_pricing_output"
OUT.mkdir(exist_ok=True)

# Independent random streams per stage. Spawning from one SeedSequence means
# changing the sweep (stage 10) can never alter the pilot sample or its
# bootstrap results, which keeps the headline numbers stable while iterating.
_seeds = np.random.SeedSequence(SEED).spawn(4)
rng_pilot, rng_boot, rng_check, rng_sweep = [np.random.default_rng(s) for s in _seeds]

# Colours match Part B so a method keeps the same colour across the whole page
COLOURS = {"Percentile": "#1f77b4", "Basic": "#ff7f0e", "BCa": "#2ca02c",
           "Studentised": "#d62728", "t-interval": "#555555"}
sns.set_theme(style="whitegrid", context="notebook")
money = FuncFormatter(lambda v, _: f"${v:,.0f}")
t0 = time.time()


# ======================================================================
# 2. LOAD, VALIDATE AND PROFILE
# ======================================================================
print("=" * 70)
print("2. LOAD, VALIDATE AND PROFILE")
print("=" * 70)
df = pd.read_csv(DATA_PATH)
print(f"Rows x columns: {df.shape}")
print(f"Missing values: {int(df.isna().sum().sum())}")

# The raw file contains one exact duplicate record (identical in all seven
# columns, including charges to four decimal places). Identical charges make a
# coincidence very unlikely, so it is treated as a duplicated entry and removed.
# One row in 1,338 barely matters; the point is that the decision is recorded.
n_dupes = int(df.duplicated().sum())
df = df.drop_duplicates().reset_index(drop=True)
print(f"Exact duplicate rows removed: {n_dupes}  ->  {len(df)} rows remain")

# Boolean smoker flag: the bootstrap engine works on plain NumPy arrays
df["is_smoker"] = df["smoker"].eq("yes")
x_all = df["charges"].to_numpy()
s_all = df["is_smoker"].to_numpy()

print(f"\nSmokers: {s_all.sum()} ({s_all.mean():.1%});  non-smokers: {(~s_all).sum()}")
print(f"Charges skewness, all policyholders: {stats.skew(x_all):.2f}")
print(f"Charges skewness, non-smokers:       {stats.skew(x_all[~s_all]):.2f}")
print(f"Charges skewness, smokers:           {stats.skew(x_all[s_all]):.2f}")
# Skewness comes almost entirely from the non-smokers; the overall shape is
# a MIXTURE of two groups, not a single lognormal-like distribution.

# --- Chart 01: distribution of charges, raw and log scale ---
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
for ax, log in zip(axes, [False, True]):
    sns.histplot(df["charges"], bins=50, log_scale=log, color="#1f77b4", ax=ax)
    ax.axvline(x_all.mean(), color="#d62728", ls="--", lw=1.6,
               label=f"mean ${x_all.mean():,.0f}")
    ax.axvline(np.median(x_all), color="#2ca02c", ls="-", lw=1.6,
               label=f"median ${np.median(x_all):,.0f}")
    ax.xaxis.set_major_formatter(money)
    ax.set_xlabel("annual charges" + (" (log scale)" if log else ""))
    ax.legend()
axes[0].set_title("Charges are right-skewed: mean sits above median")
axes[1].set_title("Log scale: still not a single clean bell")
fig.tight_layout()
fig.savefig(OUT / "01_charges_distribution.png", dpi=140)
plt.close(fig)

# --- Chart 02: charges by smoker status ---
fig, ax = plt.subplots(figsize=(8, 5.5))
order = ["no", "yes"]
sns.boxplot(data=df, x="smoker", y="charges", order=order, showfliers=False,
            palette={"no": "#9ecae1", "yes": "#fdae6b"}, hue="smoker", legend=False, ax=ax)
sns.stripplot(data=df, x="smoker", y="charges", order=order, color="k",
              alpha=0.15, size=2.5, jitter=0.25, ax=ax)
means = df.groupby("smoker")["charges"].mean()
ax.scatter([0, 1], [means["no"], means["yes"]], marker="D", color="#d62728",
           s=70, zorder=5, label="group mean")
ax.yaxis.set_major_formatter(money)
ax.set_xlabel("smoker")
ax.set_ylabel("annual charges")
ax.set_title(f"Smokers cost far more: ratio of means = {means['yes'] / means['no']:.2f}")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "02_charges_by_smoker.png", dpi=140)
plt.close(fig)


# ======================================================================
# 3. STATISTICS AND BOOTSTRAP ENGINE
# ======================================================================
# Every statistic is written as f(values, is_smoker) acting on 2-D arrays
# (rows = resamples, columns = observations). Computing all resamples in one
# vectorised call is what makes B = 9,999 feasible on a CPU-only laptop.
# Statistics that ignore smoker status simply ignore the second argument.

def stat_mean(v, s):
    return v.mean(axis=1)

def stat_median(v, s):
    return np.median(v, axis=1)

def stat_p90(v, s):
    return np.quantile(v, 0.90, axis=1)

def stat_ratio(v, s):
    """Mean charge of smokers divided by mean charge of non-smokers."""
    n_s = s.sum(axis=1)
    n_n = (~s).sum(axis=1)
    mean_s = np.where(s, v, 0.0).sum(axis=1) / n_s
    mean_n = np.where(~s, v, 0.0).sum(axis=1) / n_n
    return mean_s / mean_n

# Which interval methods each statistic gets, and whether it needs stratified
# resampling. Studentised is only offered for the mean: it needs a standard
# error inside every resample, and a nested bootstrap for the median, ratio
# and percentile is not worth the cost here.
# The ratio is STRATIFIED: resampling smokers and non-smokers separately keeps
# group sizes fixed. Plain resampling would sometimes draw a sample with very
# few (or zero) smokers, which would add noise that the real pilot did not have.
STATISTICS = [
    dict(name="Mean charge", func=stat_mean, stratified=False,
         methods=("Percentile", "Basic", "BCa", "Studentised")),
    dict(name="Median charge", func=stat_median, stratified=False,
         methods=("Percentile", "Basic", "BCa")),
    dict(name="Smoker : non-smoker ratio", func=stat_ratio, stratified=True,
         methods=("Percentile", "Basic", "BCa")),
    dict(name="90th percentile charge", func=stat_p90, stratified=False,
         methods=("Percentile", "Basic", "BCa")),
]


def resample_indices(s, B, rng, stratified):
    """Return a (B, n) matrix of indices for B bootstrap resamples."""
    n = s.size
    if not stratified:
        return rng.integers(0, n, size=(B, n))
    idx_s = np.flatnonzero(s)
    idx_n = np.flatnonzero(~s)
    draw_s = idx_s[rng.integers(0, idx_s.size, size=(B, idx_s.size))]
    draw_n = idx_n[rng.integers(0, idx_n.size, size=(B, idx_n.size))]
    return np.hstack([draw_s, draw_n])


def bca_interval(boot, theta, jack, alpha):
    """
    Bias-corrected and accelerated interval.
    z0 corrects for median bias in the bootstrap distribution; the acceleration
    constant 'a' (from the jackknife) corrects for skewness. Both shift the
    percentile cut-points rather than the endpoints themselves.
    """
    B = boot.size
    # Mid-rank handling of ties: the median and percentile of a small sample can
    # produce many resamples exactly equal to theta
    p0 = np.mean(boot < theta) + 0.5 * np.mean(boot == theta)
    p0 = np.clip(p0, 1 / (B + 1), B / (B + 1))   # keep norm.ppf finite
    z0 = norm.ppf(p0)
    d = jack.mean() - jack
    denom = 6.0 * (np.sum(d ** 2) ** 1.5)
    a = np.sum(d ** 3) / denom if denom > 0 else 0.0
    za = norm.ppf([alpha / 2, 1 - alpha / 2])
    adj = norm.cdf(z0 + (z0 + za) / (1 - a * (z0 + za)))
    return tuple(np.quantile(boot, adj))


def bootstrap_intervals(x, s, stat, B, rng, methods):
    """
    Bootstrap one statistic on one sample.
    Returns (theta, boot, cis): the sample estimate, the B bootstrap estimates,
    and a dict mapping method name -> (lower, upper).
    """
    f, n = stat["func"], x.size
    theta = f(x[None, :], s[None, :])[0]
    idx = resample_indices(s, B, rng, stat["stratified"])
    xb, sb = x[idx], s[idx]
    boot = f(xb, sb)
    cis = {}

    lo, hi = np.quantile(boot, [ALPHA / 2, 1 - ALPHA / 2])
    if "Percentile" in methods:
        cis["Percentile"] = (lo, hi)

    # Basic (reverse percentile): reflects the percentile interval about theta
    if "Basic" in methods:
        cis["Basic"] = (2 * theta - hi, 2 * theta - lo)

    if "BCa" in methods:
        # Leave-one-out samples for the jackknife. Deleting one observation
        # automatically deletes it from its own group, which is what the
        # stratified ratio needs. (The jackknife is a crude tool for the median
        # and percentile because those statistics are not smooth; BCa is
        # reported anyway because Part B found it reliable for medians.)
        loo = np.broadcast_to(np.arange(n), (n, n))[~np.eye(n, dtype=bool)].reshape(n, n - 1)
        jack = f(x[loo], s[loo])
        cis["BCa"] = bca_interval(boot, theta, jack, ALPHA)

    # Studentised (bootstrap-t): only coded for the mean, where the standard
    # error of each resample is simply its own sd / sqrt(n)
    if "Studentised" in methods:
        se_hat = x.std(ddof=1) / np.sqrt(n)
        se_b = xb.std(axis=1, ddof=1) / np.sqrt(n)
        t = (boot - theta) / se_b
        t_lo, t_hi = np.quantile(t, [ALPHA / 2, 1 - ALPHA / 2])
        cis["Studentised"] = (theta - t_hi * se_hat, theta - t_lo * se_hat)
    return theta, boot, cis


# ======================================================================
# 4. POPULATION ("KNOWN TRUTH") VALUES
# ======================================================================
# The full cleaned dataset acts as the population. It is a finite, fixed set of
# 1,337 rows rather than a true infinite population, so these are "truths" for
# the purposes of the exercise only.
pop = {st["name"]: st["func"](x_all[None, :], s_all[None, :])[0] for st in STATISTICS}
print("\n" + "=" * 70)
print("4. POPULATION VALUES (full dataset treated as known truth)")
print("=" * 70)
for k, v in pop.items():
    print(f"  {k:28s} {v:>12,.2f}")


# ======================================================================
# 5. DRAW THE PILOT SAMPLE
# ======================================================================
# A simple random sample WITHOUT replacement, not a stratified one: a real pilot
# could not guarantee how many smokers it would recruit, and the resulting
# uncertainty is part of the story.
pilot_idx = rng_pilot.choice(len(x_all), size=PILOT_N, replace=False)
xp, sp = x_all[pilot_idx], s_all[pilot_idx]
assert sp.sum() >= MIN_PER_GROUP and (~sp).sum() >= MIN_PER_GROUP, "pilot needs both groups"
print("\n" + "=" * 70)
print(f"5. PILOT SAMPLE (n = {PILOT_N})")
print("=" * 70)
print(f"  Smokers in pilot: {sp.sum()} ({sp.mean():.0%}); population share {s_all.mean():.1%}")
# This is ONE draw. Whether an interval happens to cover the population value
# here is an anecdote; coverage over many draws is the job of Part B.


# ======================================================================
# 6. TEXTBOOK BASELINE: t-INTERVAL FOR THE MEAN
# ======================================================================
# The mean is the one statistic with a standard textbook interval. It assumes the
# sampling distribution of the mean is roughly normal, which is questionable for
# a skewed variable at n = 100. (The ratio has only an approximate delta-method
# formula and the median and percentile have no simple one, which is why the
# bootstrap is attractive for them.)
t_lo, t_hi = stats.t.interval(1 - ALPHA, df=PILOT_N - 1, loc=xp.mean(), scale=stats.sem(xp))
print("\n" + "=" * 70)
print("6. TEXTBOOK t-INTERVAL FOR THE MEAN")
print("=" * 70)
print(f"  Mean charge ${xp.mean():,.0f},  95% t-interval (${t_lo:,.0f}, ${t_hi:,.0f})")


# ======================================================================
# 7. BOOTSTRAP INTERVALS ON THE PILOT
# ======================================================================
results = {}   # stat name -> dict(theta, boot, cis)
for st in STATISTICS:
    theta, boot, cis = bootstrap_intervals(xp, sp, st, B_MAIN, rng_boot, st["methods"])
    if st["name"] == "Mean charge":
        cis["t-interval"] = (t_lo, t_hi)
    results[st["name"]] = dict(theta=theta, boot=boot, cis=cis)
print(f"\n7. Bootstrap intervals built (B = {B_MAIN:,}; {time.time() - t0:.0f}s elapsed)")


# ======================================================================
# 8. CROSS-CHECK AGAINST scipy.stats.bootstrap
# ======================================================================
# Independent validation of the engine above. The two implementations use
# different random streams, so endpoints will differ by Monte Carlo noise; a
# difference of a few per cent of the interval width is expected, a large one
# would indicate a bug.
print("\n" + "=" * 70)
print("8. CROSS-CHECK AGAINST scipy.stats.bootstrap")
print("=" * 70)
check_rows = []
for name, fn in [("Mean charge", np.mean), ("Median charge", np.median)]:
    for method, label in [("percentile", "Percentile"), ("BCa", "BCa")]:
        sc = stats.bootstrap((xp,), fn, vectorized=True, n_resamples=B_MAIN,
                             confidence_level=1 - ALPHA, method=method,
                             random_state=rng_check).confidence_interval
        lo, hi = results[name]["cis"][label]
        width = hi - lo
        check_rows.append((name, label, lo, sc.low, hi, sc.high,
                           max(abs(lo - sc.low), abs(hi - sc.high)) / width))
check = pd.DataFrame(check_rows, columns=["statistic", "method", "own_lower", "scipy_lower",
                                          "own_upper", "scipy_upper", "max_diff_pct_of_width"])
check["max_diff_pct_of_width"] *= 100
print(check.round(1).to_string(index=False))
check.to_csv(OUT / "crosscheck_scipy.csv", index=False)


# ======================================================================
# 9. RESULTS AGAINST THE POPULATION VALUES
# ======================================================================
rows = []
for name, r in results.items():
    for method, (lo, hi) in r["cis"].items():
        rows.append(dict(statistic=name, method=method, estimate=r["theta"],
                         lower=lo, upper=hi, width=hi - lo,
                         lower_arm=r["theta"] - lo,       # distance estimate -> lower end
                         upper_arm=hi - r["theta"],       # distance estimate -> upper end
                         population=pop[name],
                         covers_population=bool(lo <= pop[name] <= hi)))
res = pd.DataFrame(rows)
res.to_csv(OUT / "pilot_intervals.csv", index=False)
print("\n" + "=" * 70)
print("9. PILOT INTERVALS vs POPULATION VALUES")
print("=" * 70)
print(res.round(2).to_string(index=False))
# Asymmetry is the key diagnostic: the t-interval is symmetric by construction
# (lower_arm == upper_arm), whereas bootstrap intervals can lean to one side.

# --- Chart 03: bootstrap distributions for median, ratio and P90 ---
panel_stats = ["Median charge", "Smoker : non-smoker ratio", "90th percentile charge"]
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
for ax, name in zip(axes, panel_stats):
    r = results[name]
    sns.histplot(r["boot"], bins=60, color="#9ecae1", ax=ax)
    lo, hi = r["cis"]["Percentile"]
    ax.axvspan(lo, hi, color="#1f77b4", alpha=0.18, label="95% percentile interval")
    for v in r["cis"]["BCa"]:
        ax.axvline(v, color=COLOURS["BCa"], ls=":", lw=2)
    ax.axvline(np.nan, color=COLOURS["BCa"], ls=":", lw=2, label="BCa endpoints")
    ax.axvline(r["theta"], color="k", ls="--", lw=1.5, label="pilot estimate")
    ax.axvline(pop[name], color="red", lw=2, label="population value")
    ax.set_title(name)
    ax.set_xlabel("bootstrap estimate")
    if "ratio" not in name:
        ax.xaxis.set_major_formatter(money)
        ax.xaxis.set_major_locator(MaxNLocator(5))   # fewer ticks: dollar labels collide otherwise
axes[0].legend(fontsize=8)
fig.suptitle(f"Bootstrap distributions from one pilot (n = {PILOT_N}, B = {B_MAIN:,})", fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "03_bootstrap_distributions.png", dpi=140)
plt.close(fig)

# --- Chart 04: all methods, all statistics, against the population value ---
fig, axes = plt.subplots(2, 2, figsize=(14, 7.5))
for ax, st in zip(axes.ravel(), STATISTICS):
    name = st["name"]
    sub = res[res.statistic == name].reset_index(drop=True)
    for i, row in sub.iterrows():
        ax.hlines(i, row.lower, row.upper, color=COLOURS[row.method], lw=5)
        ax.plot(row.estimate, i, "o", color="k", ms=5)
    ax.axvline(pop[name], color="red", lw=2, label="population value")
    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels(sub.method)
    ax.invert_yaxis()
    ax.set_title(name)
    if "ratio" not in name:
        ax.xaxis.set_major_formatter(money)
# One shared legend outside the panels: inside the mean panel it hid the t-interval bar
fig.legend([plt.Line2D([0], [0], color="red", lw=2)], ["population value"],
           loc="lower center", fontsize=9)
fig.suptitle("95% intervals by method (dot = pilot estimate)", fontsize=13)
fig.tight_layout(rect=[0, 0.04, 1, 1])
fig.savefig(OUT / "04_interval_comparison.png", dpi=140)
plt.close(fig)


# ======================================================================
# 10. PILOT-SIZE SWEEP: IS THE PILOT BIG ENOUGH?
# ======================================================================
# The decision-maker's practical question. For each pilot size, draw many
# independent pilots, build a percentile interval for each statistic, and record
# how wide it is RELATIVE to the population value. Relative width makes the four
# statistics comparable, and the median over repeats stops one lucky or unlucky
# draw from dominating.
# Caveat: samples are drawn without replacement from only 1,337 rows, so at
# n = 200 (15% of the data) the bootstrap, which ignores this finite-population
# effect, gives intervals that are slightly too wide.
print("\n" + "=" * 70)
print(f"10. PILOT-SIZE SWEEP ({SWEEP_REPS} pilots at each of n = {SWEEP_NS})")
print("=" * 70)
sweep = []
skipped = {n: 0 for n in SWEEP_NS}
for n in SWEEP_NS:
    for _ in range(SWEEP_REPS):
        idx = rng_sweep.choice(len(x_all), size=n, replace=False)
        x, s = x_all[idx], s_all[idx]
        for st in STATISTICS:
            if st["stratified"] and (s.sum() < MIN_PER_GROUP or (~s).sum() < MIN_PER_GROUP):
                skipped[n] += 1       # too few smokers for a ratio; count and move on
                continue
            _, _, cis = bootstrap_intervals(x, s, st, B_SWEEP, rng_sweep, ("Percentile",))
            lo, hi = cis["Percentile"]
            sweep.append((st["name"], n, hi - lo, (hi - lo) / pop[st["name"]]))
    print(f"  n = {n:<4d} done  ({time.time() - t0:5.0f}s)", flush=True)
sw = pd.DataFrame(sweep, columns=["statistic", "n", "width", "relative_width"])
sw_summary = (sw.groupby(["statistic", "n"])
                .agg(median_width=("width", "median"),
                     median_relative_width=("relative_width", "median"),
                     reps=("width", "size"))
                .reset_index())
sw_summary.to_csv(OUT / "pilot_size_sweep.csv", index=False)
print(sw_summary.round(3).to_string(index=False))
if any(skipped.values()):
    print(f"  Pilots skipped for the ratio (too few smokers): {skipped}")

# --- Chart 05: relative interval width against pilot size ---
fig, ax = plt.subplots(figsize=(9, 6))
palette = sns.color_palette("deep", len(STATISTICS))
for colour, st in zip(palette, STATISTICS):
    g = sw_summary[sw_summary.statistic == st["name"]]
    ax.plot(g.n, g.median_relative_width, marker="o", lw=2, color=colour, label=st["name"])
# Reference line: width falling as 1 / sqrt(n), anchored on the mean at n = 50.
# If a statistic runs parallel to it, quadrupling the pilot halves the interval.
g0 = sw_summary[(sw_summary.statistic == "Mean charge") & (sw_summary.n == SWEEP_NS[0])]
anchor = g0.median_relative_width.iloc[0]
ref = anchor * np.sqrt(SWEEP_NS[0] / np.array(SWEEP_NS))
ax.plot(SWEEP_NS, ref, color="grey", ls="--", label=r"$1/\sqrt{n}$ reference")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xticks(SWEEP_NS); ax.set_xticklabels(SWEEP_NS)
ax.set_yticks([0.25, 0.3, 0.4, 0.5, 0.6])     # explicit ticks: log axes default to 6x10^-1 style labels
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.xaxis.set_minor_formatter(NullFormatter())
ax.yaxis.set_minor_formatter(NullFormatter())
ax.set_xlabel("pilot size n")
ax.set_ylabel("median 95% interval width, as % of population value")
ax.set_title(f"How wide are the intervals as the pilot grows? ({SWEEP_REPS} pilots per size)")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "05_pilot_size_sweep.png", dpi=140)
plt.close(fig)

print(f"\nSaved charts and CSVs to {OUT}  (total {time.time() - t0:.0f}s)")
