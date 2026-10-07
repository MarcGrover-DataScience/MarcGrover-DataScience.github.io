"""
=============================================================================
SEASONAL FORECASTING WITH PROPHET
German daily electricity consumption (Open Power System Data), 2006-2017
=============================================================================
Portfolio category : Time-Series Analysis
Business scenario  : An energy retailer's procurement team buys power forward
                     against forecast demand. How accurately can daily demand
                     be forecast 90 and 365 days ahead, and how large a buffer
                     margin should be held against forecast error?
Dataset            : Open Power System Data, daily German electricity
                     consumption, wind and solar generation (GWh), 2006-2017
                     https://github.com/jenfly/opsd  (derived from the OPSD
                     time series package, https://open-power-system-data.org/)

BUILD STATUS - CHECKPOINT C (baselines, core Prophet, regressor experiment)
    Section 1  Data loading and validation                  -> chart 01
    Section 2  Chronological train/test split
    Section 3  Exploratory analysis (training only)         -> charts 02-05
    Section 4  Holiday calendar and evaluation helpers
    Section 5  Baselines M0: seasonal naive, SARIMA,
               SARIMAX                                      -> chart 06
    Section 6  Prophet core M1: walk-forward cross-validation
               and tuning (training data only)             -> charts 07-08
    Section 7  Final fits, test-set evaluation, diagnostics  -> charts 09-12
    Section 8  Wind and solar regressor experiment (M2, M3)
               with placebo and bootstrap noise floors, and
               a conservative-trend robustness rerun        -> charts 13-17
    Section 9: Air Passengers bridge (Prophet versus ARIMA), charts 19-21.

REQUIREMENTS: pandas, numpy, matplotlib, seaborn (0.12 or later), statsmodels,
prophet (which also installs `holidays`).

RUNTIME: the first run takes roughly 20 to 30 minutes on a CPU-only laptop,
almost all of it in the Section 6 cross-validation (about 350 Prophet fits).
Its results are cached to a CSV, so later runs take a few minutes. Set
FORCE_RETUNE = True to repeat the tuning.
=============================================================================
"""

# =============================================================================
# SECTION 0: IMPORTS AND CONFIGURATION
# =============================================================================
import urllib.request
import warnings
from pathlib import Path

import logging

import holidays
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dateutil.easter import easter  # ships with pandas; used for Easter/Whit Sunday
from prophet import Prophet
from prophet.utilities import regressor_coefficients
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Charts are written straight to PNG files and never displayed. The
# non-interactive backend avoids GUI problems when the script is run from
# PyCharm on Windows and guarantees that a chart never blocks the run.
matplotlib.use("Agg")
warnings.filterwarnings("ignore", category=FutureWarning)
# cmdstanpy (Prophet's optimiser) logs two INFO lines per fit, which would bury
# the console output across several hundred cross-validation fits. Setting the
# logger level alone does not work, because cmdstanpy resets it to INFO on first
# use unless the logger already has a handler. A handler that shows only
# warnings and errors is therefore attached first, so those are still reported.
for logger_name in ("cmdstanpy", "prophet"):
    package_logger = logging.getLogger(logger_name)
    if not package_logger.handlers:
        warning_handler = logging.StreamHandler()
        warning_handler.setLevel(logging.WARNING)
        package_logger.addHandler(warning_handler)
    package_logger.setLevel(logging.WARNING)
pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 20)

# --- Paths -------------------------------------------------------------------
# Paths are anchored to the script's own folder so the results are identical
# whichever working directory PyCharm happens to use.
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "prophet_outputs"  # named for its contents: every chart and table lands here
OUTPUT_DIR.mkdir(exist_ok=True)
DATA_URL = "https://raw.githubusercontent.com/jenfly/opsd/master/opsd_germany_daily.csv"
DATA_CACHE = BASE_DIR / "opsd_germany_daily.csv"  # local copy: one download only

# --- Chronological split -----------------------------------------------------
# Ten years to train on, two to test on. Two test years give two complete
# annual cycles, which is a more reliable verdict than a single year.
TRAIN_START, TRAIN_END = "2006-01-01", "2015-12-31"
TEST_START, TEST_END = "2016-01-01", "2017-12-31"

# --- Modelling constants (Checkpoint B) --------------------------------------
RANDOM_SEED = 42
# The level step found in the EDA lies between 23 Dec 2013 and 14 Jan 2014.
# The year boundary falls inside that window, so it is used as the step date
# for the intervention variable that SARIMAX and Prophet are given.
STEP_DATE = pd.Timestamp("2014-01-01")
# April to September 2009 is masked in the sensitivity variant (Section 7).
MASK_2009_START, MASK_2009_END = "2009-04-01", "2009-09-30"
# Walk-forward cross-validation inside the training data: first fit on five
# years, forecast 365 days ahead, then move the forecast origin on by 180 days.
CV_INITIAL_DAYS, CV_HORIZON_DAYS, CV_PERIOD_DAYS = 5 * 365, 365, 180
# Tuning is slow (a few hundred model fits), so its results are cached.
# Set to True after changing the grid or any modelling assumption.
FORCE_RETUNE = False
HORIZON_BANDS = [("1-90 days", 1, 90), ("91-365 days", 91, 365), ("366-731 days", 366, 731)]
# --- Regressor experiment and uncertainty (Checkpoint C) ----------------------
N_PLACEBOS = 10            # placebo regressors fitted per experiment (Section 8.3)
BOOTSTRAP_BLOCK_DAYS = 28  # block length for the paired bootstrap (Section 4.4)
N_BOOTSTRAP = 2000         # bootstrap resamples
# In the matched-range rerun, no trend changepoint is allowed in the last 730 days before
# the forecast origin, whatever the length of the training window (Section 8.6).
FREE_TREND_DAYS = 730

EXPECTED_COLUMNS = ["Consumption", "Wind", "Solar", "Wind+Solar"]
WEEKDAY_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# --- Colours: one consistent palette across every chart ----------------------
COL_TRAIN = "#1f5f8b"   # blue   - training data
COL_TEST = "#d9822b"    # orange - test data
COL_GREY = "#8c8c8c"    # neutral - raw daily values
COL_ACCENT = "#b03a48"  # red    - highlights and annotations
sns.set_theme(style="whitegrid", context="notebook")


def print_section(title):
    """Print a clear banner so the console output mirrors the script layout."""
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def save_figure(fig, filename):
    """Save a figure as a numbered PNG and close it to free memory."""
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved chart: {path.name}")


# =============================================================================
# SECTION 1: DATA LOADING AND VALIDATION
# =============================================================================
# Forecasting models fail silently on bad inputs (a duplicated date or a gap
# simply shifts the seasonal pattern), so the data is validated before any
# modelling. In production these checks would live in a validation suite such
# as the one built in the Great Expectations project.
print_section("SECTION 1: DATA LOADING AND VALIDATION")

if not DATA_CACHE.exists():
    print(f"  Downloading dataset from {DATA_URL}")
    urllib.request.urlretrieve(DATA_URL, DATA_CACHE)

raw = pd.read_csv(DATA_CACHE, parse_dates=["Date"])
print(f"  Loaded {len(raw):,} rows and {raw.shape[1]} columns from {DATA_CACHE.name}")

# --- 1.1 Structural checks (hard failures) -----------------------------------
# These conditions would make every downstream result meaningless, so the
# script stops rather than carrying on.
missing_columns = set(EXPECTED_COLUMNS) - set(raw.columns)
if missing_columns:
    raise ValueError(f"Expected columns are missing: {missing_columns}")

duplicate_dates = int(raw["Date"].duplicated().sum())
df = raw.set_index("Date").sort_index()
full_range = pd.date_range(df.index.min(), df.index.max(), freq="D")
missing_dates = full_range.difference(df.index)
non_positive = int((df["Consumption"] <= 0).sum())

if duplicate_dates or len(missing_dates) or non_positive:
    raise ValueError(
        f"Structural validation failed: {duplicate_dates} duplicate dates, "
        f"{len(missing_dates)} missing dates, {non_positive} non-positive values."
    )
df.index.freq = "D"  # a strictly regular daily index, confirmed by the checks above

# --- 1.2 Availability of the regressors (soft findings) ----------------------
# Wind and solar are not complete. Prophet refuses missing values in a
# regressor, so the availability windows decide how much data the regressor
# models in Checkpoint C can use.
wind_start = df["Wind"].first_valid_index()
solar_start = df["Solar"].first_valid_index()
solar_gap_days = df.loc[solar_start:, "Solar"].isna().sum()
wind_gap_days = df.loc[wind_start:, "Wind"].isna().sum()

# Wind+Solar should equal Wind plus Solar wherever all three are present.
both_present = df[["Wind", "Solar", "Wind+Solar"]].dropna()
max_sum_difference = (both_present["Wind"] + both_present["Solar"]
                      - both_present["Wind+Solar"]).abs().max()

validation = pd.DataFrame(
    [
        ("Rows", f"{len(df):,}"),
        ("Date range", f"{df.index.min().date()} to {df.index.max().date()}"),
        ("Duplicate dates", duplicate_dates),
        ("Missing calendar dates", len(missing_dates)),
        ("Non-positive consumption values", non_positive),
        ("Missing Consumption values", int(df["Consumption"].isna().sum())),
        ("Wind first available", str(wind_start.date())),
        ("Wind gap days after first available", int(wind_gap_days)),
        ("Solar first available", str(solar_start.date())),
        ("Solar gap days after first available", int(solar_gap_days)),
        ("Max |Wind + Solar - Wind+Solar|", f"{max_sum_difference:.4f}"),
    ],
    columns=["Check", "Result"],
)
print("\n  Validation summary:")
print(validation.to_string(index=False))
validation.to_csv(OUTPUT_DIR / "table_validation_summary.csv", index=False)

# --- 1.3 Plausibility screen --------------------------------------------------
# The lowest-demand days should be public holidays or the Christmas period.
# If they were ordinary working days that would point to a data fault.
national_holidays = holidays.Germany(years=range(2006, 2018))  # national only

# The national list omits Easter Sunday and Whit Sunday because they fall on
# a Sunday anyway. Consumption on those two days is nonetheless well below an
# ordinary Sunday (quantified in Section 3.3), so they are added as
# supplementary holidays. Both are computed from the date of Easter.
holiday_names = {pd.Timestamp(date): name for date, name in national_holidays.items()}
supplementary_names = set()
for year in range(2006, 2018):
    easter_sunday = pd.Timestamp(easter(year))
    holiday_names[easter_sunday] = "Ostersonntag"
    holiday_names[easter_sunday + pd.Timedelta(days=49)] = "Pfingstsonntag"  # Whit Sunday
supplementary_names.update({"Ostersonntag", "Pfingstsonntag"})

lowest = df["Consumption"].nsmallest(5)
print("\n  Five lowest-consumption days (expected: holidays / Christmas period):")
for date, value in lowest.items():
    label = holiday_names.get(date, "not a holiday")
    print(f"    {date.date()} ({WEEKDAY_LABELS[date.dayofweek]})  {value:7.1f} GWh  - {label}")
n_holiday_days = sum(date in holiday_names for date in lowest.index)
n_in_2009 = int((lowest.index.year == 2009).sum())
print(f"  {n_holiday_days} of the five are holidays and {n_in_2009} of the five fall in 2009. The 2009 "
      f"concentration is investigated in Section 3.1.")

# --- Chart 01: full series with split boundary and regressor availability ----
# A descriptive overview of the whole series. Every later analytical decision
# uses the training period only.
fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True,
                         gridspec_kw={"height_ratios": [2.2, 1, 1]})
ax = axes[0]
ax.plot(df.index, df["Consumption"], color=COL_GREY, lw=0.6, alpha=0.8, label="Daily consumption")
ax.plot(df.index, df["Consumption"].rolling(365, center=True).mean(),
        color=COL_TRAIN, lw=2.2, label="365-day centred mean")
ax.set_ylabel("Consumption (GWh/day)")
ax.set_title("German daily electricity consumption, wind and solar generation, 2006-2017")
ax.legend(loc="lower left", ncol=2)  # empty corner: keeps the labels clear

for a in axes:
    a.axvspan(pd.Timestamp(TRAIN_START), pd.Timestamp(TRAIN_END), color=COL_TRAIN, alpha=0.05)
    a.axvspan(pd.Timestamp(TEST_START), pd.Timestamp(TEST_END), color=COL_TEST, alpha=0.12)
    a.axvline(pd.Timestamp(TEST_START), color=COL_ACCENT, ls="--", lw=1)
axes[0].text(pd.Timestamp("2010-06-01"), axes[0].get_ylim()[1] * 0.97, "Training: 2006-2015",
             ha="center", va="top", color=COL_TRAIN, fontweight="bold")
axes[0].text(pd.Timestamp("2017-01-01"), axes[0].get_ylim()[1] * 0.97, "Test: 2016-2017",
             ha="center", va="top", color=COL_TEST, fontweight="bold")

axes[1].plot(df.index, df["Wind"], color="#3a8f6e", lw=0.7)
axes[1].set_ylabel("Wind (GWh/day)")
axes[1].text(pd.Timestamp("2006-06-01"), df["Wind"].max() * 0.5,
             f"No wind data before {wind_start.date()}", color=COL_GREY, style="italic")

axes[2].plot(df.index, df["Solar"], color="#c9a227", lw=0.7)
axes[2].set_ylabel("Solar (GWh/day)")
axes[2].text(pd.Timestamp("2006-06-01"), df["Solar"].max() * 0.5,
             f"No solar data before {solar_start.date()}", color=COL_GREY, style="italic")
axes[2].set_xlabel("Date")
fig.tight_layout()
save_figure(fig, "01_data_overview.png")


# =============================================================================
# SECTION 2: CHRONOLOGICAL TRAIN/TEST SPLIT
# =============================================================================
# The split comes BEFORE the exploratory analysis so that no modelling
# decision (seasonality mode, holiday treatment, regressor choice) can be
# influenced by the test years. A random split would be wrong for a time
# series: it would let the model train on days that surround the ones it is
# tested on and would overstate accuracy.
print_section("SECTION 2: CHRONOLOGICAL TRAIN/TEST SPLIT")

train = df.loc[TRAIN_START:TRAIN_END].copy()
test = df.loc[TEST_START:TEST_END].copy()
assert train.index.max() < test.index.min(), "Training data must precede test data"
assert len(train) + len(test) == len(df), "Split must cover every observation exactly once"

# Regressor windows inside the training period. Rows with a missing regressor
# are dropped, which affects only the few solar gap days in early 2014.
train_wind = train.loc[wind_start:, ["Consumption", "Wind"]].dropna()
train_solar = train.loc[solar_start:, ["Consumption", "Wind", "Solar"]].dropna()

# The regressor experiment in Checkpoint C supplies the ACTUAL test-period
# wind and solar values (a perfect-foresight upper bound), so the test set
# must be complete for both.
test_regressors_complete = bool(test[["Wind", "Solar"]].notna().all().all())

split_summary = pd.DataFrame(
    [
        ("Training (all)", train.index.min().date(), train.index.max().date(), len(train), len(train) / 365.25),
        ("Test", test.index.min().date(), test.index.max().date(), len(test), len(test) / 365.25),
        ("Training with Wind (M2 window)", train_wind.index.min().date(), train_wind.index.max().date(),
         len(train_wind), len(train_wind) / 365.25),
        ("Training with Wind + Solar (M3 window)", train_solar.index.min().date(), train_solar.index.max().date(),
         len(train_solar), len(train_solar) / 365.25),
    ],
    columns=["Set", "Start", "End", "Days", "Years"],
)
split_summary["Years"] = split_summary["Years"].round(2)
print(split_summary.to_string(index=False))
print(f"\n  Test period has complete Wind and Solar values: {test_regressors_complete}")
split_summary.to_csv(OUTPUT_DIR / "table_split_summary.csv", index=False)

# From this point on, only `train` is analysed.
consumption = train["Consumption"]


# =============================================================================
# SECTION 3: EXPLORATORY ANALYSIS (TRAINING DATA ONLY)
# =============================================================================
print_section("SECTION 3: EXPLORATORY ANALYSIS (TRAINING DATA ONLY)")

# Flags used throughout the EDA. Public holidays and the Christmas-New Year
# period behave unlike ordinary days, so they are separated out. Only
# NATIONAL holidays are used: regional holidays (which differ by federal
# state) are not in this national series and are a stated limitation.
holiday_dates = pd.DatetimeIndex(list(holiday_names))
train["is_holiday"] = train.index.isin(holiday_dates)
train["in_christmas_window"] = (((train.index.month == 12) & (train.index.day >= 24)) |
                                ((train.index.month == 1) & (train.index.day <= 6)))
train["regular_day"] = ~(train["is_holiday"] | train["in_christmas_window"])
print(f"  Public holidays in training data: {int(train['is_holiday'].sum())} days; "
      f"Christmas-New Year window: {int(train['in_christmas_window'].sum())} days")

# ----- 3.1 Level: annual means and the year-on-year signature of a step ------
print("\n  --- 3.1 Annual level and level shifts ---")
annual = consumption.groupby(consumption.index.year).agg(["mean", "std"])
annual["change_pct"] = annual["mean"].pct_change() * 100
print(annual.round(1).to_string())
annual.round(2).to_csv(OUTPUT_DIR / "table_yearly_level.csv")

# A level change is hard to see in raw data because seasonality dominates.
# Comparing each day with the same weekday 52 weeks earlier (a 364-day lag
# preserves the weekday) removes the seasonal pattern. A one-off step then
# shows up as a plateau of year-on-year change lasting about a year, and a
# temporary dip shows up as a negative plateau followed by a mirror-image
# positive one a year later.
roll7 = consumption.rolling(7).mean()
roll28 = consumption.rolling(28).mean()
yoy7 = (roll7 / roll7.shift(364) - 1) * 100
yoy28 = (roll28 / roll28.shift(364) - 1) * 100

# Rather than assume there is a single break, list EVERY stretch of at least
# 45 days in which the 7-day year-on-year change stays beyond +/-5%.
outside_band = yoy7.abs() > 5
run_id = (outside_band != outside_band.shift()).cumsum()
segments = []
for _, run in yoy7[outside_band].groupby(run_id[outside_band]):
    if len(run) >= 45:
        segments.append((run.index[0], run.index[-1], len(run), run.mean()))
segments = pd.DataFrame(segments, columns=["start", "end", "days", "mean_yoy_pct"])
print("\n  Sustained year-on-year departures (> +/-5% for at least 45 days):")
print(segments.assign(start=segments["start"].dt.date, end=segments["end"].dt.date)
      .round(1).to_string(index=False))

# Read the list as three separate events, not four independent ones:
#  * Apr-Sep 2009: a sharp, temporary shortfall. The global financial crisis is
#    one candidate, but its abrupt start and finish make a reporting gap in
#    the source equally plausible. The data alone cannot settle it.
#  * Mid-2010 and late 2014: mirror images. Each is simply the comparison
#    against a depressed or pre-step base a year earlier, not new events.
#  * Around the 2013/14 year-end: a persistent step (the largest and longest
#    plateau) after which the level stays high.
step_candidates = segments[segments["mean_yoy_pct"] > 0].copy()
step_candidates["score"] = step_candidates["days"] * step_candidates["mean_yoy_pct"]
step_row = step_candidates.sort_values("score").iloc[-1]
rough_start = step_row["start"]

# The 7-day comparison is distorted around Christmas, because the same weekday
# 52 weeks earlier sits at a different point in the holiday. The onset is
# therefore refined using REGULAR days only (holidays and the Christmas-New
# Year window excluded), smoothed over 14 days.
regular_series = consumption.where(train["regular_day"])
yoy_regular = (regular_series / regular_series.shift(364) - 1) * 100
yoy_regular_14 = yoy_regular.rolling(14, min_periods=8).mean()
near_step = yoy_regular_14[rough_start - pd.Timedelta(days=60): rough_start + pd.Timedelta(days=45)]
last_normal_day = near_step[near_step < 2].index.max()      # last date still close to last year's level
first_elevated_day = near_step[near_step > 8].index.min()   # first date clearly above it
print(f"\n  Largest persistent step. On regular days the year-on-year change is within +/-2% "
      f"up to {last_normal_day.date()}")
print(f"  and above +8% from {first_elevated_day.date()}. The step therefore occurred between "
      f"those dates, inside the Christmas-New Year")
print("  window, where it cannot be resolved to a single day.")

# Size of the step: 52 weeks either side of the midpoint of that interval.
step_date = last_normal_day + (first_elevated_day - last_normal_day) / 2
before = consumption.loc[step_date - pd.Timedelta(days=364): step_date - pd.Timedelta(days=1)]
after = consumption.loc[step_date: step_date + pd.Timedelta(days=363)]
step_size_pct = (after.mean() / before.mean() - 1) * 100
print(f"  Mean consumption, 52 weeks before: {before.mean():.1f} GWh/day; "
      f"52 weeks after: {after.mean():.1f} GWh/day ({step_size_pct:+.1f}%)")

# Once the warm/cold weather differences of early 2014 wash out, what level
# does the year-on-year change settle at? Averaged over the months 6 to 11
# after the step, it gives a cleaner reading of the step's size.
settled = yoy28[step_date + pd.Timedelta(days=182): step_date + pd.Timedelta(days=330)]
print(f"  Year-on-year change 6-11 months after the step: {settled.mean():+.1f}% on average")

# Candidate explanation to investigate, NOT a confirmed cause: the OPSD
# documentation reports a "representativity factor" for German load data of
# 91% until 2014 and 97% from 2015, i.e. a change in coverage that alone
# would lift reported values by roughly this much.
coverage_uplift_pct = (0.97 / 0.91 - 1) * 100
print(f"  Documented OPSD coverage change (91% -> 97%) would imply about "
      f"{coverage_uplift_pct:+.1f}% - a candidate explanation, to be treated with caution.")

# The 2009 shortfall, month by month, against the same month a year earlier
monthly = consumption.resample("MS").mean()
monthly_yoy_2009 = (monthly["2009-01":"2009-12"].values / monthly["2008-01":"2008-12"].values - 1) * 100
print("\n  2009 monthly change on 2008 (%):")
print("   ", ", ".join(f"{m}: {v:+.1f}" for m, v in
                       zip(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
                           monthly_yoy_2009)))

# --- Chart 02: annual level and year-on-year change ---------------------------
fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))
ax = axes[0]
bars = ax.bar(annual.index.astype(str), annual["mean"], color=COL_TRAIN, alpha=0.85)
for bar, pct in zip(bars, annual["change_pct"]):
    if not np.isnan(pct):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 6, f"{pct:+.1f}%",
                ha="center", va="bottom", fontsize=9)
ax.set_ylim(annual["mean"].min() * 0.9, annual["mean"].max() * 1.05)
ax.set_ylabel("Mean daily consumption (GWh)")
ax.set_title("Annual mean consumption (change on previous year)")

ax = axes[1]
ax.plot(yoy28.index, yoy28, color=COL_TRAIN, lw=1.4)
ax.axhline(0, color="black", lw=0.8)
for _, seg in segments.iterrows():
    colour = COL_ACCENT if seg["mean_yoy_pct"] > 0 else COL_TEST
    ax.axvspan(seg["start"], seg["end"], color=colour, alpha=0.15)
ax.annotate("2009 shortfall\n(crisis or reporting gap?)", xy=(pd.Timestamp("2009-07-15"), -9),
            xytext=(pd.Timestamp("2006-09-01"), -14), fontsize=9,
            arrowprops={"arrowstyle": "->", "color": "black"})
ax.annotate("2013/14 step:\nplateau of about +6% to +10%\nfor a year", xy=(pd.Timestamp("2014-03-01"), 10.5),
            xytext=(pd.Timestamp("2011-08-01"), 13.5), fontsize=9,
            arrowprops={"arrowstyle": "->", "color": "black"})
ax.text(pd.Timestamp("2010-04-01"), 15.5, "2010: base effect\nof the 2009 shortfall", fontsize=9, ha="center")
ax.set_ylim(-19, 19)
ax.set_ylabel("Year-on-year change, 28-day mean (%)")
ax.set_title("Year-on-year change (shaded: sustained departures beyond +/-5%)")
fig.tight_layout()
save_figure(fig, "02_yearly_level_shift.png")

# ----- 3.2 Weekly and annual seasonal profiles --------------------------------
print("\n  --- 3.2 Weekly and annual seasonal profiles ---")
# Each day is expressed relative to the mean of its own calendar year. That
# removes the year-to-year level differences (including the step above, which
# falls close to a year boundary) so that only the seasonal shape remains.
year_mean = consumption.groupby(consumption.index.year).transform("mean")
profile = pd.DataFrame({
    "weekday": train.index.dayofweek,
    "month": train.index.month,
    "pct_vs_year_mean": (consumption / year_mean - 1) * 100,
})[train["regular_day"]]  # regular days only: holidays would distort weekday means

weekday_means = profile.groupby("weekday")["pct_vs_year_mean"].mean()
month_means = profile.groupby("month")["pct_vs_year_mean"].mean()
print("  Weekday effect (% vs calendar-year mean, regular days):")
print("   ", ", ".join(f"{WEEKDAY_LABELS[d]} {v:+.1f}%" for d, v in weekday_means.items()))
print(f"  Monthly effect: peak {month_means.idxmax()} ({month_means.max():+.1f}%), "
      f"trough {month_means.idxmin()} ({month_means.min():+.1f}%)")

# Additive or multiplicative seasonality? If seasonal swings grow in
# proportion to the level of the series, the seasonality is multiplicative.
# The ARIMA project needed a Box-Cox transform for this reason; Prophet has a
# switch instead (seasonality_mode). Here the amplitude of each seasonal
# swing is measured year by year and compared with that year's level.
regular = train[train["regular_day"]]
amplitude_rows = []
for year, group in regular.groupby(regular.index.year):
    level = consumption[consumption.index.year == year].mean()
    weekly_amp = (group.loc[group.index.dayofweek < 5, "Consumption"].mean()
                  - group.loc[group.index.dayofweek == 6, "Consumption"].mean())
    annual_amp = (group.loc[group.index.month.isin([12, 1, 2]), "Consumption"].mean()
                  - group.loc[group.index.month.isin([6, 7, 8]), "Consumption"].mean())
    amplitude_rows.append((year, level, weekly_amp, annual_amp))
amplitude = pd.DataFrame(amplitude_rows, columns=["year", "level", "weekly_amp", "annual_amp"]).set_index("year")


def coefficient_of_variation(series):
    """Relative spread: standard deviation divided by mean."""
    return series.std() / series.mean()


print("\n  Seasonal amplitude by year (GWh):")
print(amplitude.round(1).to_string())
for name in ["weekly_amp", "annual_amp"]:
    absolute_cv = coefficient_of_variation(amplitude[name])
    relative_cv = coefficient_of_variation(amplitude[name] / amplitude["level"])
    corr = amplitude[name].corr(amplitude["level"])
    print(f"  {name}: CV of absolute amplitude {absolute_cv:.3f} vs CV of amplitude/level "
          f"{relative_cv:.3f}; correlation with level {corr:+.2f}")
print("  Note: ten annual points give little statistical power, so this is only a")
print("  guide. The seasonality mode is settled by cross-validation in Section 6.")

# --- Chart 03: seasonal profiles and amplitude against level ------------------
fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
sns.barplot(data=profile, x="weekday", y="pct_vs_year_mean", order=range(7), color=COL_TRAIN,
            errorbar=("ci", 95), ax=axes[0])
axes[0].set_xticks(range(7))
axes[0].set_xticklabels(WEEKDAY_LABELS)
axes[0].set_xlabel("")
axes[0].set_ylabel("% difference from calendar-year mean")
axes[0].set_title("Weekly profile (regular days)")
axes[0].axhline(0, color="black", lw=0.8)

sns.barplot(data=profile, x="month", y="pct_vs_year_mean", order=range(1, 13), color=COL_TRAIN,
            errorbar=("ci", 95), ax=axes[1])
axes[1].set_xticks(range(12))
axes[1].set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
axes[1].set_xlabel("Month")
axes[1].set_ylabel("")
axes[1].set_title("Annual profile (regular days)")
axes[1].axhline(0, color="black", lw=0.8)

ax = axes[2]
ax.scatter(amplitude["level"], amplitude["annual_amp"], color=COL_TRAIN, s=55, label="Annual (winter minus summer)")
ax.scatter(amplitude["level"], amplitude["weekly_amp"], color=COL_TEST, s=55, label="Weekly (weekday minus Sunday)")
for year, row in amplitude.iterrows():
    ax.annotate(str(year)[2:], (row["level"], row["annual_amp"]), textcoords="offset points",
                xytext=(4, 4), fontsize=8, color=COL_TRAIN)
    ax.annotate(str(year)[2:], (row["level"], row["weekly_amp"]), textcoords="offset points",
                xytext=(4, 4), fontsize=8, color=COL_TEST)
ax.set_xlabel("Annual mean consumption (GWh/day)")
ax.set_ylabel("Seasonal amplitude (GWh/day)")
ax.set_title("Does seasonal amplitude scale with level?")
ax.legend(loc="center left", fontsize=9)
fig.tight_layout()
save_figure(fig, "03_seasonal_profiles.png")

# ----- 3.3 Public holiday and Christmas-New Year effects ----------------------
print("\n  --- 3.3 Holiday effects ---")
# Effect of a holiday = actual consumption relative to what a normal day of
# the same weekday would have used. The expectation is the mean of the same
# weekday 1-4 weeks either side, using regular days only. This local baseline
# is unaffected by the level step and by seasonality.
clean = consumption.where(train["regular_day"])
neighbours = pd.concat([clean.shift(k) for k in (-28, -21, -14, -7, 7, 14, 21, 28)], axis=1)
baseline = neighbours.mean(axis=1)
baseline[neighbours.notna().sum(axis=1) < 4] = np.nan  # need at least four reference days
deviation_pct = (consumption / baseline - 1) * 100

holiday_rows = [(date, holiday_names[date].split(";")[0].strip(), deviation_pct.loc[date])
                for date in train.index[train["is_holiday"]]]
holiday_effects = pd.DataFrame(holiday_rows, columns=["date", "holiday", "deviation_pct"]).dropna()
holiday_summary = (holiday_effects.groupby("holiday")["deviation_pct"]
                   .agg(["mean", "std", "count"]).sort_values("mean"))
holiday_summary["in_national_list"] = ~holiday_summary.index.isin(supplementary_names)
print(holiday_summary.round(1).to_string())
print("\n  Ostersonntag and Pfingstsonntag are absent from the national holiday list but show")
print("  clear dips against an ordinary Sunday, so they are carried into the Prophet holiday")
print("  calendar in Checkpoint B as custom holidays.")
holiday_summary.round(2).to_csv(OUTPUT_DIR / "table_holiday_effects.csv")

# Christmas-New Year profile: 20 December to 8 January in each winter that
# lies fully inside the training period (December 2006 to December 2014).
window_records = []
for year in range(2006, 2015):
    start = pd.Timestamp(year=year, month=12, day=20)
    for offset in range(20):  # 20 December ... 8 January
        date = start + pd.Timedelta(days=offset)
        window_records.append((year, offset, date, deviation_pct.get(date, np.nan)))
window = pd.DataFrame(window_records, columns=["winter", "offset", "date", "deviation_pct"])
window_profile = window.groupby("offset")["deviation_pct"].agg(["mean", "std"])
lowest_offset = window_profile["mean"].idxmin()
lowest_date = pd.Timestamp(year=2013, month=12, day=20) + pd.Timedelta(days=int(lowest_offset))
print(f"\n  Deepest Christmas-New Year dip: {window_profile['mean'].min():.1f}% vs a normal "
      f"same-weekday day, around {lowest_date.strftime('%d %b')}")

# --- Chart 04: holiday effects -------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), gridspec_kw={"width_ratios": [1, 1.3]})
ax = axes[0]
bar_colours = [COL_TRAIN if listed else COL_TEST for listed in holiday_summary["in_national_list"]]
ax.barh(holiday_summary.index, holiday_summary["mean"], color=bar_colours, alpha=0.9)
for position, (name, row) in enumerate(holiday_summary.iterrows()):
    ax.text(row["mean"] - 0.6, position, f"n={int(row['count'])}", va="center", ha="right",
            fontsize=8, color="black")
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("Mean % difference from a normal same-weekday day")
ax.set_title("Holidays (blue: national list; orange: added Sundays)")
ax.set_xlim(holiday_summary["mean"].min() * 1.25, 3)

ax = axes[1]
for winter, group in window.groupby("winter"):
    ax.plot(group["offset"], group["deviation_pct"], color=COL_GREY, lw=0.8, alpha=0.5)
ax.plot(window_profile.index, window_profile["mean"], color=COL_ACCENT, lw=2.4, label="Mean across winters")
ax.axhline(0, color="black", lw=0.8)
tick_positions = [0, 4, 5, 6, 11, 12, 19]
tick_labels = ["20 Dec", "24", "25", "26", "31", "1 Jan", "8 Jan"]
ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels)
ax.set_ylabel("% difference from a normal same-weekday day")
ax.set_title("Christmas-New Year period (each grey line is one winter)")
ax.legend(loc="lower left")
fig.tight_layout()
save_figure(fig, "04_holiday_effects.png")

# ----- 3.4 Do wind and solar carry information about consumption? --------------
print("\n  --- 3.4 Wind and solar against consumption ---")
# Raw correlations are misleading here: solar peaks in summer, when demand is
# lowest, so it correlates NEGATIVELY with consumption purely through the
# annual cycle, and both renewables grow year on year because capacity is
# being built, not because demand changes. To see whether they carry any
# extra information, each variable is first stripped of (a) its year-level,
# (b) the weekday pattern and (c) the annual cycle (four Fourier pairs), and
# the leftover residuals are correlated.


def seasonal_design_matrix(index, fourier_order=4):
    """Year dummies + weekday dummies + annual Fourier terms."""
    columns = []
    for year in sorted(set(index.year)):
        columns.append((index.year == year).astype(float))       # year level (absorbs capacity growth)
    for weekday in range(1, 7):
        columns.append((index.dayofweek == weekday).astype(float))  # Monday is the reference day
    day_of_year = index.dayofyear.values / 365.25
    for k in range(1, fourier_order + 1):
        columns.append(np.sin(2 * np.pi * k * day_of_year))
        columns.append(np.cos(2 * np.pi * k * day_of_year))
    return np.column_stack(columns)


def residualise(frame, columns):
    """Return each column with the seasonal design-matrix fit removed."""
    design = seasonal_design_matrix(frame.index)
    residuals = {}
    for col in columns:
        beta, *_ = np.linalg.lstsq(design, frame[col].values, rcond=None)
        residuals[col] = frame[col].values - design @ beta
    return pd.DataFrame(residuals, index=frame.index)


def correlation_with_interval(x, y):
    """Pearson correlation with an approximate 95% interval (Fisher transform).
    Daily residuals are autocorrelated, so the effective sample size is smaller
    than n and the interval is optimistic: treat it as a lower bound on the
    uncertainty."""
    r = np.corrcoef(x, y)[0, 1]
    z, se = np.arctanh(r), 1 / np.sqrt(len(x) - 3)
    return r, np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)


correlation_rows = []
for label, frame, variable in [
    ("Wind (2010-2015 window)", train_wind, "Wind"),
    ("Wind (2012-2015 window)", train_solar, "Wind"),
    ("Solar (2012-2015 window)", train_solar, "Solar"),
]:
    resid = residualise(frame, ["Consumption", variable])
    raw_r = frame["Consumption"].corr(frame[variable])
    adj_r, lo, hi = correlation_with_interval(resid["Consumption"], resid[variable])
    correlation_rows.append((label, len(frame), raw_r, adj_r, lo, hi))

correlations = pd.DataFrame(correlation_rows,
                            columns=["Regressor and window", "Days", "Raw r", "Adjusted r", "CI low", "CI high"])
print(correlations.round(3).to_string(index=False))
correlations.round(4).to_csv(OUTPUT_DIR / "table_regressor_correlations.csv", index=False)

# Capacity growth: annual mean output of each renewable inside training data.
renewable_annual = pd.DataFrame({
    "Wind": train_wind["Wind"].groupby(train_wind.index.year).mean(),
    "Solar": train_solar["Solar"].groupby(train_solar.index.year).mean(),
})
print("\n  Annual mean output (GWh/day) - growth reflects installed capacity, not demand:")
print(renewable_annual.round(1).to_string())

# --- Chart 05: regressor relationships ------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
ax = axes[0]
x_positions = np.arange(len(correlations))
width = 0.36
ax.bar(x_positions - width / 2, correlations["Raw r"], width, color=COL_GREY, label="Raw correlation")
ax.bar(x_positions + width / 2, correlations["Adjusted r"], width, color=COL_TRAIN,
       label="After removing year level, weekday and annual cycle")
ax.errorbar(x_positions + width / 2, correlations["Adjusted r"],
            yerr=[correlations["Adjusted r"] - correlations["CI low"],
                  correlations["CI high"] - correlations["Adjusted r"]],
            fmt="none", ecolor="black", capsize=4, lw=1)
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(x_positions)
ax.set_xticklabels([label.replace(" (", "\n(") for label in correlations["Regressor and window"]])
ax.set_ylabel("Correlation with consumption")
ax.set_title("Correlation with consumption, before and after adjustment")
ax.legend(loc="upper right", fontsize=9)

ax = axes[1]
ax.plot(renewable_annual.index, renewable_annual["Wind"], marker="o", color="#3a8f6e", lw=2, label="Wind")
ax.plot(renewable_annual.index, renewable_annual["Solar"], marker="s", color="#c9a227", lw=2, label="Solar")
ax.set_xlabel("Year")
ax.set_ylabel("Mean output (GWh/day)")
ax.set_title("Renewable output grows with installed capacity")
ax.legend(loc="upper left")
fig.tight_layout()
save_figure(fig, "05_regressor_correlation.png")

# =============================================================================
# FINDINGS CARRIED FORWARD TO MODELLING (CHECKPOINT B)
# =============================================================================
# Three findings from the EDA change how Prophet has to be configured.
print_section("FINDINGS CARRIED FORWARD TO MODELLING")

# 1. Prophet only allows trend changepoints inside the first `changepoint_range`
#    share of the training history (default 0.8). Where does that boundary fall?
default_last_changepoint = train.index[int(0.8 * len(train)) - 1]
print(f"  1. With Prophet's default changepoint_range of 0.8, the last possible trend")
print(f"     changepoint is {default_last_changepoint.date()}. The level step occurred between "
      f"{last_normal_day.date()} and {first_elevated_day.date()},")
print("     so it falls at the very edge of the default range, and the trend alone may not")
print("     capture it. Two remedies are therefore tested by cross-validation in Section 6:")
print("     a wider changepoint_range (0.9 and 0.95) and an explicit level-step regressor.")
print("     (The result, reported in Section 6, is that the regressor is what works.)")

# 2. Holidays: national list + Easter/Whit Sunday + a Christmas-New Year window.
print("  2. The holiday calendar needs custom entries (Easter Sunday, Whit Sunday and a")
print("     Christmas-New Year window with day-specific effects) beyond the built-in list.")

# 3. The 2009 shortfall is unexplained. Deleting it silently would be an
#    undocumented data edit; keeping it may distort the trend. Both versions
#    will be fitted and compared as a sensitivity check.
print("  3. The Apr-Sep 2009 shortfall is kept in the primary model and masked in a")
print("     sensitivity variant, so its influence on the forecast is measured, not assumed.")

# =============================================================================
# SECTION 4: HOLIDAY CALENDAR AND EVALUATION HELPERS
# =============================================================================
print_section("SECTION 4: HOLIDAY CALENDAR AND EVALUATION HELPERS")

# --- 4.1 Calendar flags for any date range -----------------------------------
# The EDA flagged the training days only. The same flags are needed for the
# test days (to compare errors on holidays and ordinary days), so the logic is
# wrapped in a function. It reuses `holiday_dates` from Section 3.
def calendar_flags(index):
    """Return holiday and Christmas-New Year flags for a daily DatetimeIndex."""
    flags = pd.DataFrame(index=index)
    flags["is_holiday"] = index.isin(holiday_dates)
    flags["in_christmas_window"] = (((index.month == 12) & (index.day >= 24)) |
                                    ((index.month == 1) & (index.day <= 6)))
    return flags


test_flags = calendar_flags(test.index)
test_special_days = (test_flags["is_holiday"] | test_flags["in_christmas_window"]).values
train_special_days = (train["is_holiday"] | train["in_christmas_window"]).values
print(f"  Test days that are holidays or in the Christmas-New Year window: "
      f"{int(test_special_days.sum())} of {len(test)}")


# --- 4.2 Holiday table for Prophet ---------------------------------------------
# Prophet learns one effect per named holiday. Three decisions are taken from
# the EDA (chart 04):
#  * Easter Sunday and Whit Sunday are added to the national list.
#  * The Christmas-New Year period is split into groups of days that behave
#    alike, rather than treated as one block, because the dip is not uniform:
#    Boxing Day and 1 January are the deepest, 27-30 December is shallower,
#    and consumption recovers gradually through early January.
#  * Where two national holidays coincide (1 May 2008 was also Ascension Day)
#    the first name is used, so the two effects are not added together.
# Reformationstag (31 October 2017, a one-off national holiday) occurs only in
# the test period, so no effect can be learned for it; it is left in, and Prophet
# gives it no effect. That affects one test day and is noted as a limitation.
def build_prophet_holidays():
    rows = [(date, name.split(";")[0].strip()) for date, name in holiday_names.items()]
    for year in range(2005, 2018):
        rows += [
            (pd.Timestamp(year, 12, 22), "Vor_Weihnachten"),
            (pd.Timestamp(year, 12, 23), "Vor_Weihnachten"),
            (pd.Timestamp(year, 12, 24), "Heiligabend"),
            (pd.Timestamp(year, 12, 27), "Zwischen_den_Jahren"),
            (pd.Timestamp(year, 12, 28), "Zwischen_den_Jahren"),
            (pd.Timestamp(year, 12, 29), "Zwischen_den_Jahren"),
            (pd.Timestamp(year, 12, 30), "Zwischen_den_Jahren"),
            (pd.Timestamp(year, 12, 31), "Silvester"),
            (pd.Timestamp(year + 1, 1, 2), "Nach_Neujahr"),
            (pd.Timestamp(year + 1, 1, 3), "Nach_Neujahr"),
            (pd.Timestamp(year + 1, 1, 4), "Anfang_Januar"),
            (pd.Timestamp(year + 1, 1, 5), "Anfang_Januar"),
            (pd.Timestamp(year + 1, 1, 6), "Anfang_Januar"),
        ]
    table = pd.DataFrame(rows, columns=["ds", "holiday"]).drop_duplicates()
    # Keep only dates inside the data range (Prophet does not need the rest).
    table = table[(table["ds"] >= df.index.min()) & (table["ds"] <= df.index.max())]
    return table.sort_values("ds").reset_index(drop=True)


prophet_holidays = build_prophet_holidays()
print(f"  Prophet holiday table: {prophet_holidays['holiday'].nunique()} named holidays, "
      f"{len(prophet_holidays)} dated rows")


# --- 4.3 Evaluation metrics --------------------------------------------------------
# The same four headline metrics as the ARIMA project (MAE, RMSE, R-squared,
# MAPE), plus bias and the empirical coverage of the 95% prediction interval.
def forecast_metrics(actual, forecast, lower=None, upper=None):
    """Headline accuracy metrics for one forecast against the actual values."""
    actual, forecast = np.asarray(actual, float), np.asarray(forecast, float)
    error = actual - forecast
    ss_res, ss_tot = np.sum(error ** 2), np.sum((actual - actual.mean()) ** 2)
    coverage = width = np.nan
    if lower is not None and upper is not None and not np.isnan(np.asarray(lower, float)).all():
        coverage = np.mean((actual >= np.asarray(lower)) & (actual <= np.asarray(upper))) * 100
        # Coverage alone can be bought with very wide intervals, so the average
        # width (as a share of the forecast) is reported beside it.
        width = np.mean((np.asarray(upper) - np.asarray(lower)) / forecast) * 100
    return {
        "MAE (GWh)": np.mean(np.abs(error)),
        "RMSE (GWh)": np.sqrt(np.mean(error ** 2)),
        "MAPE (%)": np.mean(np.abs(error) / actual) * 100,
        "R2": 1 - ss_res / ss_tot,
        "Bias (GWh, forecast - actual)": -np.mean(error),   # positive = over-forecast
        "95% interval coverage (%)": coverage,
        "Mean 95% interval width (% of forecast)": width,
    }


def horizon_band_table(model_name, actual, forecast):
    """Accuracy and procurement buffer by forecast horizon (days after the origin).

    The buffer answers the business question directly. A retailer that buys
    forward exactly the forecast volume is short on the days when demand exceeds
    the forecast. The buffer is the 95th percentile of that shortfall: hold this
    much extra volume and demand exceeds the hedged amount on only 1 day in 20.

    The risk is two-sided, though. A model that over-forecasts needs a small
    buffer but leaves the retailer holding surplus volume. The 95th percentile
    of the over-forecast is therefore reported beside the buffer, and the two
    must be read together.
    """
    horizon = (actual.index - pd.Timestamp(TRAIN_END)).days.values
    rows = []
    for label, low, high in HORIZON_BANDS:
        mask = (horizon >= low) & (horizon <= high)
        a, f = actual.values[mask], forecast.values[mask]
        shortfall_pct = (a - f) / f * 100
        rows.append({
            "Model": model_name, "Horizon band": label, "Days": int(mask.sum()),
            "MAPE (%)": np.mean(np.abs(a - f) / a) * 100,
            "MAE (GWh)": np.mean(np.abs(a - f)),
            "Buffer: 95th pct shortfall (%)": np.percentile(shortfall_pct, 95),
            "Buffer: 95th pct shortfall (GWh/day)": np.percentile(a - f, 95),
            "Surplus: 95th pct over-forecast (%)": np.percentile((f - a) / f * 100, 95),
        })
    return pd.DataFrame(rows)


# --- 4.4 Is a difference between two models real? A paired block bootstrap ------
# A lower MAE on one test set is not proof of a better model: the 731 test days
# are a single sample of weather and calendar luck. The paired bootstrap asks
# how much the MAE difference between two models would vary if the test period
# were redrawn. Daily errors are strongly autocorrelated, so whole 28-day
# blocks are resampled rather than single days (which would make the interval
# far too narrow).
def paired_block_bootstrap(actual, forecast_a, forecast_b, seed=RANDOM_SEED):
    """Mean of |error A| - |error B| with a 95% block-bootstrap interval.

    Negative values mean model A has the smaller error.
    """
    actual = np.asarray(actual, float)
    diff = np.abs(actual - np.asarray(forecast_a, float)) - np.abs(actual - np.asarray(forecast_b, float))
    n, block = len(diff), BOOTSTRAP_BLOCK_DAYS
    n_blocks = int(np.ceil(n / block))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n - block + 1, size=(N_BOOTSTRAP, n_blocks))
    index = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(N_BOOTSTRAP, -1)[:, :n]
    resampled = diff[index].mean(axis=1)
    return {"mean_diff": diff.mean(), "ci_low": np.percentile(resampled, 2.5),
            "ci_high": np.percentile(resampled, 97.5), "share_a_better": np.mean(resampled < 0)}


def paired_comparison_table(comparisons):
    """Run the paired bootstrap for a list of (label, key_a, key_b) comparisons."""
    rows = []
    for label, key_a, key_b in comparisons:
        result = paired_block_bootstrap(test_actual, test_forecasts[key_a]["yhat"], test_forecasts[key_b]["yhat"])
        rows.append({"Comparison (A vs B)": label, "MAE(A) - MAE(B) (GWh)": result["mean_diff"],
                     "95% CI low": result["ci_low"], "95% CI high": result["ci_high"],
                     "Resamples where A better (%)": result["share_a_better"] * 100})
    return pd.DataFrame(rows)


# One place to collect every model's test-period forecast (columns: yhat, lower, upper).
test_actual = test["Consumption"]
test_forecasts = {}
MODEL_LABELS = {
    "M0a": "M0a Seasonal naive",
    "M0b": "M0b SARIMA (weekly)",
    "M0c": "M0c SARIMAX (Fourier + holidays + step)",
    "M1d": "M1-default Prophet (out of the box)",
    "M1": "M1 Prophet (tuned)",
    "M1s": "M1-s Prophet (tuned, 2009 masked)",
}
MODEL_COLOURS = {"M0a": "#7f7f7f", "M0b": "#8c564b", "M0c": "#2e8b57",
                 "M1d": "#c9a227", "M1": COL_TRAIN, "M1s": "#7b4ea3"}


# =============================================================================
# SECTION 5: BASELINES (M0)
# =============================================================================
# Three baselines of increasing sophistication, so that Prophet is compared
# with the best classical alternative and not with a straw man:
#   M0a  seasonal naive   - no model at all
#   M0b  SARIMA, weekly   - the standard classical model, one seasonal period
#   M0c  SARIMAX          - weekly seasonal terms plus annual Fourier terms,
#                           holiday indicators and the same level-step
#                           indicator that Prophet is given
# All three forecast the full 731 test days from a single origin at the end of
# 2015, exactly as the ARIMA project forecast its 29-month holdout.
print_section("SECTION 5: BASELINES (M0)")

forecast_origin = train.index.max()
horizon_days = (test.index - forecast_origin).days.values   # 1, 2, ..., 731

# --- 5.1 M0a: seasonal naive -------------------------------------------------------
# Each test day is forecast with the value from the same WEEKDAY in the most
# recent year that is already observed. A 364-day lag (52 weeks) keeps the
# weekday aligned; a 365-day lag would compare a Monday with a Sunday. For the
# second forecast year the lag is 728 days, because 364 days back would fall
# inside the test period, which is unknown at the forecast origin.
lag_blocks = np.ceil(horizon_days / 364).astype(int)
source_dates = test.index - pd.to_timedelta(364 * lag_blocks, unit="D")
assert (source_dates <= forecast_origin).all(), "Seasonal naive must only use observed data"
test_forecasts["M0a"] = pd.DataFrame(
    {"yhat": consumption.loc[source_dates].values, "lower": np.nan, "upper": np.nan}, index=test.index)

# --- 5.2 M0b: SARIMA with a weekly seasonal period ------------------------------
# SARIMA(1,0,1)(0,1,1)7: short-term dynamics (AR1, MA1) plus a seasonally
# differenced, seasonal-MA(1) component that learns the weekday pattern. The
# orders are the standard starting point for daily data with a weekly cycle.
# It has no annual cycle, no holidays and no level-step handling by design:
# that is what a single-period SARIMA is. The result shows how far such a model
# gets on a 2-year horizon.
train_series = consumption.asfreq("D")
m0b = SARIMAX(train_series, order=(1, 0, 1), seasonal_order=(0, 1, 1, 7)).fit(disp=False, maxiter=200)
m0b_forecast = m0b.get_forecast(len(test))
interval = m0b_forecast.conf_int(alpha=0.05)
test_forecasts["M0b"] = pd.DataFrame(
    {"yhat": m0b_forecast.predicted_mean.values, "lower": interval.iloc[:, 0].values,
     "upper": interval.iloc[:, 1].values}, index=test.index)
print(f"  M0b SARIMA(1,0,1)(0,1,1,7) fitted; optimiser converged: {m0b.mle_retvals.get('converged')}")


# --- 5.3 M0c: SARIMAX with annual Fourier terms, holidays and a step -----------
# Annual seasonality (period about 365.25) is too long for SARIMA's seasonal
# machinery, so it enters as Fourier terms, the standard "dynamic harmonic
# regression" device. Four sine/cosine pairs match the EDA. Holiday indicators
# and the level-step dummy give it the SAME information Prophet receives, so
# any difference in the results reflects the model, not the inputs. All of
# these regressors are deterministic calendar quantities, known at forecast time.
def sarimax_exog(index, fourier_order=4):
    exog = pd.DataFrame(index=index)
    day_of_year = index.dayofyear.values / 365.25
    for k in range(1, fourier_order + 1):
        exog[f"sin{k}"] = np.sin(2 * np.pi * k * day_of_year)
        exog[f"cos{k}"] = np.cos(2 * np.pi * k * day_of_year)
    flags = calendar_flags(index)
    exog["is_holiday"] = flags["is_holiday"].astype(float)
    exog["in_christmas_window"] = flags["in_christmas_window"].astype(float)
    exog["post_step"] = (index >= STEP_DATE).astype(float)
    return exog


m0c = SARIMAX(train_series, exog=sarimax_exog(train_series.index),
              order=(1, 0, 1), seasonal_order=(0, 1, 1, 7)).fit(disp=False, maxiter=200)
m0c_forecast = m0c.get_forecast(len(test), exog=sarimax_exog(test.index))
interval = m0c_forecast.conf_int(alpha=0.05)
test_forecasts["M0c"] = pd.DataFrame(
    {"yhat": m0c_forecast.predicted_mean.values, "lower": interval.iloc[:, 0].values,
     "upper": interval.iloc[:, 1].values}, index=test.index)
print(f"  M0c SARIMAX fitted; optimiser converged: {m0c.mle_retvals.get('converged')}")
print(f"  M0c estimated level step (post_step): {m0c.params['post_step']:+.1f} GWh/day; "
      f"holiday effect {m0c.params['is_holiday']:+.1f}; Christmas-window effect "
      f"{m0c.params['in_christmas_window']:+.1f}")

baseline_metrics = pd.DataFrame(
    {MODEL_LABELS[key]: forecast_metrics(test_actual, test_forecasts[key]["yhat"],
                                         test_forecasts[key]["lower"], test_forecasts[key]["upper"])
     for key in ["M0a", "M0b", "M0c"]}).T
print("\n  Baseline accuracy on the 731-day test set:")
print(baseline_metrics.round(2).to_string())

# --- Chart 06: baseline forecasts -----------------------------------------------------
fig = plt.figure(figsize=(15, 9))
grid = fig.add_gridspec(2, 2, height_ratios=[1.1, 1])
ax_full = fig.add_subplot(grid[0, :])
ax_full.plot(test.index, test_actual.rolling(7, center=True).mean(), color="black", lw=2, label="Actual (7-day mean)")
for key in ["M0a", "M0b", "M0c"]:
    ax_full.plot(test.index, test_forecasts[key]["yhat"].rolling(7, center=True).mean(),
                 color=MODEL_COLOURS[key], lw=1.6, label=MODEL_LABELS[key])
ax_full.set_ylabel("Consumption (GWh/day)")
ax_full.set_title("Baselines on the 2016-2017 test set (7-day centred means, so the weekly cycle does not obscure the annual one)")
ax_full.legend(loc="lower left", ncol=2, fontsize=9)

for position, (start, end, title) in enumerate([("2016-03-14", "2016-04-10", "Four weeks around Easter 2016"),
                                               ("2016-12-12", "2017-01-08", "Christmas and New Year 2016/17")]):
    ax = fig.add_subplot(grid[1, position])
    window = slice(start, end)
    ax.plot(test_actual.loc[window].index, test_actual.loc[window], color="black", lw=2, label="Actual")
    for key in ["M0a", "M0b", "M0c"]:
        ax.plot(test_forecasts[key].loc[window].index, test_forecasts[key].loc[window, "yhat"],
                color=MODEL_COLOURS[key], lw=1.4, label=MODEL_LABELS[key])
    ax.set_title(title)
    ax.set_ylabel("Consumption (GWh/day)")
    ax.tick_params(axis="x", rotation=30)
fig.tight_layout()
save_figure(fig, "06_baseline_forecasts.png")


# =============================================================================
# SECTION 6: PROPHET CORE (M1) - CROSS-VALIDATION AND TUNING
# =============================================================================
# Prophet's hyperparameters are chosen by walk-forward cross-validation inside
# the TRAINING data only. The 2016-2017 test set is not used in any choice.
print_section("SECTION 6: PROPHET CORE (M1) - CROSS-VALIDATION AND TUNING")

model_frame = pd.DataFrame({
    "ds": df.index,
    "y": df["Consumption"].values,
    # Level-step indicator: 0 before STEP_DATE and 1 afterwards. Offered to
    # Prophet as an optional extra regressor (see the tuning grid below).
    "post_step": (df.index >= STEP_DATE).astype(float),
})
train_frame = model_frame[model_frame["ds"] <= TRAIN_END].reset_index(drop=True)
test_frame = model_frame[model_frame["ds"] >= TEST_START].reset_index(drop=True)


def make_prophet(config, with_uncertainty, extra_regressors=()):
    """Build a Prophet model from a configuration dictionary.

    config["default"] = True gives the out-of-the-box model: Prophet's own
    defaults and its built-in German holiday list. Otherwise the tuned set-up
    is built: custom holidays, and optionally the level-step regressor.
    """
    samples = 1000 if with_uncertainty else 0   # intervals are not needed in CV, and skipping them is faster
    if config.get("default"):
        model = Prophet(interval_width=0.95, uncertainty_samples=samples)
        model.add_country_holidays(country_name="DE")
        return model
    model = Prophet(
        growth="linear",
        changepoint_prior_scale=config["cps"],      # flexibility of the trend: higher = more freedom to bend
        changepoint_range=config["range"],          # share of history in which trend changes may occur
        seasonality_mode=config["mode"],            # additive or multiplicative seasonal effects
        yearly_seasonality=int(config["fourier"]),  # Fourier order of the annual cycle
        weekly_seasonality=True,
        daily_seasonality=False,                    # meaningless for daily data
        holidays=prophet_holidays,
        interval_width=0.95,                        # matches the ARIMA project's 95% intervals
        uncertainty_samples=samples,
    )
    if config["step"] == "regressor":
        model.add_regressor("post_step")
    for name in extra_regressors:   # e.g. Wind and Solar in the Section 8 experiment
        model.add_regressor(name)
    return model


def fit_prophet(config, history, with_uncertainty=False, extra_regressors=()):
    model = make_prophet(config, with_uncertainty, extra_regressors)
    model.fit(history[["ds", "y", "post_step", *extra_regressors]])
    return model


# --- 6.1 Walk-forward cross-validation folds --------------------------------------------
# Each fold fits on all data up to a cutoff and forecasts the next 365 days.
# This mimics the real task (forecast a year ahead from today) at nine
# different "todays", and it never lets a model see data after its cutoff.
def make_cutoffs():
    cutoff = pd.Timestamp(TRAIN_END) - pd.Timedelta(days=CV_HORIZON_DAYS)
    earliest = pd.Timestamp(TRAIN_START) + pd.Timedelta(days=CV_INITIAL_DAYS)
    cutoffs = []
    while cutoff >= earliest:
        cutoffs.append(cutoff)
        cutoff -= pd.Timedelta(days=CV_PERIOD_DAYS)
    return sorted(cutoffs)


CV_CUTOFFS = make_cutoffs()
print(f"  {len(CV_CUTOFFS)} walk-forward folds, forecast origins from {CV_CUTOFFS[0].date()} "
      f"to {CV_CUTOFFS[-1].date()}, 365-day horizon")
print("  Folds with an origin before 2014 cannot see the level step in their history, and")
print("  their forecast year contains it, so every model fails on it alike. Only the latest")
print("  folds can show whether a configuration handles the step, which is why per-fold")
print("  results are reported alongside the average.")


def run_cv(config, collect_predictions=False):
    """Walk-forward cross-validation for one configuration.

    Returns a per-fold metrics table and, optionally, every out-of-sample
    prediction (needed for the error-by-horizon chart).
    """
    fold_rows, prediction_frames = [], []
    for cutoff in CV_CUTOFFS:
        history = train_frame[train_frame["ds"] <= cutoff]
        future = train_frame[(train_frame["ds"] > cutoff) &
                             (train_frame["ds"] <= cutoff + pd.Timedelta(days=CV_HORIZON_DAYS))]
        model = fit_prophet(config, history)
        predicted = model.predict(future[["ds", "post_step"]])["yhat"].values
        actual = future["y"].values
        ape = np.abs(actual - predicted) / actual * 100
        fold_rows.append({"cutoff": cutoff, "MAPE": ape.mean(), "MAE": np.mean(np.abs(actual - predicted)),
                          "RMSE": np.sqrt(np.mean((actual - predicted) ** 2))})
        if collect_predictions:
            prediction_frames.append(pd.DataFrame({"cutoff": cutoff, "horizon": np.arange(1, len(future) + 1),
                                                   "ape": ape, "error": predicted - actual}))
    folds = pd.DataFrame(fold_rows)
    return folds, (pd.concat(prediction_frames, ignore_index=True) if collect_predictions else None)


# --- 6.2 The tuning grid -------------------------------------------------------------------
# Four dimensions, every combination (36 configurations), annual Fourier order
# held at Prophet's default of 10 for this first pass:
#   cps   changepoint_prior_scale: how freely the trend may bend. 0.05 is the default.
#   range changepoint_range: the EDA showed the level step lies at about 85% of the
#         training history, beyond the default 0.8, so 0.9 and 0.95 are tried.
#   mode  additive or multiplicative seasonality (mixed evidence in the EDA).
#   step  whether the level-step indicator is offered as an extra regressor, or
#         Prophet must cope with the step using its trend changepoints alone.
# A second pass then varies the Fourier order of the annual cycle around the
# best configuration (a greedy search, to keep run time reasonable).
TUNING_CACHE = OUTPUT_DIR / "table_cv_tuning_results.csv"
GRID = [{"cps": cps, "range": rng, "mode": mode, "step": step, "fourier": 10}
        for cps in (0.01, 0.05, 0.2)
        for rng in (0.8, 0.9, 0.95)
        for mode in ("additive", "multiplicative")
        for step in ("none", "regressor")]


def tune(configs, label):
    rows = []
    for number, config in enumerate(configs, start=1):
        folds, _ = run_cv(config)
        rows.append({**config, "mean_MAPE": folds["MAPE"].mean(), "mean_MAE": folds["MAE"].mean(),
                     "mean_RMSE": folds["RMSE"].mean(), "last2_MAPE": folds["MAPE"].tail(2).mean()})
        print(f"    {label} {number:2d}/{len(configs)}  cps={config['cps']:<5} range={config['range']:<5} "
              f"{config['mode']:<14} step={config['step']:<9} fourier={config['fourier']:<2} "
              f"-> mean MAPE {rows[-1]['mean_MAPE']:.2f}%")
    return pd.DataFrame(rows)


if TUNING_CACHE.exists() and not FORCE_RETUNE:
    tuning = pd.read_csv(TUNING_CACHE)
    print(f"  Loaded cached tuning results ({len(tuning)} configurations) from {TUNING_CACHE.name}")
else:
    print(f"  Tuning pass 1: {len(GRID)} configurations x {len(CV_CUTOFFS)} folds (a few minutes) ...")
    pass_one = tune(GRID, "pass 1")
    best_one = pass_one.sort_values("mean_MAPE").iloc[0]
    base = {key: best_one[key] for key in ("cps", "range", "mode", "step")}
    print(f"  Tuning pass 2: annual Fourier order around the best pass-1 configuration ...")
    pass_two = tune([{**base, "fourier": order} for order in (6, 15)], "pass 2")
    tuning = pd.concat([pass_one, pass_two], ignore_index=True)
    tuning.to_csv(TUNING_CACHE, index=False)

best = tuning.sort_values("mean_MAPE").iloc[0]
best_config = {"cps": float(best["cps"]), "range": float(best["range"]), "mode": str(best["mode"]),
               "step": str(best["step"]), "fourier": int(best["fourier"])}
print(f"\n  Best configuration by mean cross-validated MAPE ({best['mean_MAPE']:.2f}%):")
print(f"    {best_config}")
print("\n  Ten best configurations:")
print(tuning.sort_values("mean_MAPE").head(10).round(2).to_string(index=False))

# How much do the individual choices matter? Average over all other settings.
print("\n  Average mean-MAPE by choice (fourier = 10 rows only):")
for column in ("mode", "step", "range", "cps"):
    print("   ", tuning[tuning["fourier"] == 10].groupby(column)["mean_MAPE"].mean().round(2).to_dict())

# --- 6.3 Cross-validated error by forecast horizon -------------------------------------------
# The business question asks about 90 and 365 days ahead. The cross-validation
# predictions give the error at each horizon, averaged over the nine origins.
default_config = {"default": True}
best_folds, best_predictions = run_cv(best_config, collect_predictions=True)
default_folds, default_predictions = run_cv(default_config, collect_predictions=True)
print("\n  Per-fold MAPE (%) at each forecast origin:")
fold_table = pd.DataFrame({"origin": [c.date() for c in CV_CUTOFFS],
                           "tuned": best_folds["MAPE"].round(2).values,
                           "default": default_folds["MAPE"].round(2).values})
print(fold_table.to_string(index=False))
fold_table.to_csv(OUTPUT_DIR / "table_cv_fold_results.csv", index=False)

horizon_summary = []
for label, predictions in (("Tuned", best_predictions), ("Default", default_predictions)):
    for limit in (90, 365):
        horizon_summary.append({"Model": label, "Horizon": f"1-{limit} days",
                                "CV MAPE (%)": predictions.loc[predictions["horizon"] <= limit, "ape"].mean()})
horizon_summary = pd.DataFrame(horizon_summary)
print("\n  Cross-validated MAPE by horizon window (mean over all folds):")
print(horizon_summary.round(2).to_string(index=False))
horizon_summary.round(3).to_csv(OUTPUT_DIR / "table_cv_horizon_summary.csv", index=False)

# --- Chart 07: tuning results ----------------------------------------------------------------
grid_rows = tuning[tuning["fourier"] == 10].copy()
grid_rows["row_label"] = grid_rows["mode"] + " / step: " + grid_rows["step"]
grid_rows["col_label"] = "cps " + grid_rows["cps"].astype(str) + "\nrange " + grid_rows["range"].astype(str)
heat = grid_rows.pivot(index="row_label", columns="col_label", values="mean_MAPE")
column_order = (grid_rows.sort_values(["cps", "range"])["col_label"].drop_duplicates().tolist())
heat = heat[column_order]

fig, axes = plt.subplots(1, 2, figsize=(17, 5.8), gridspec_kw={"width_ratios": [3, 1]})
sns.heatmap(heat, annot=True, fmt=".2f", cmap="viridis_r", cbar_kws={"label": "Mean cross-validated MAPE (%)"},
            ax=axes[0])
axes[0].set_xlabel("")
axes[0].set_ylabel("")
axes[0].set_title("Pass 1: 36 configurations (annual Fourier order 10)")
axes[0].tick_params(axis="x", rotation=0, labelsize=8)

fourier_rows = tuning[(tuning["cps"] == best_config["cps"]) & (tuning["range"] == best_config["range"]) &
                      (tuning["mode"] == best_config["mode"]) & (tuning["step"] == best_config["step"])
                      ].sort_values("fourier")
bar_colours = [COL_ACCENT if order == best_config["fourier"] else COL_TRAIN for order in fourier_rows["fourier"]]
axes[1].bar(fourier_rows["fourier"].astype(str), fourier_rows["mean_MAPE"], color=bar_colours)
# The axis starts at zero: the three orders differ by only about 0.02 percentage points,
# and a truncated axis would make that negligible difference look large.
axes[1].set_ylim(0, fourier_rows["mean_MAPE"].max() * 1.2)
for position, value in enumerate(fourier_rows["mean_MAPE"]):
    axes[1].text(position, value + 0.05, f"{value:.2f}", ha="center")
axes[1].set_xlabel("Annual Fourier order")
axes[1].set_ylabel("Mean cross-validated MAPE (%)")
axes[1].set_title("Pass 2: annual Fourier order")
fig.tight_layout()
save_figure(fig, "07_cv_tuning_results.png")

# --- Chart 08: cross-validated error by horizon -------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
for label, predictions, colour in (("Tuned", best_predictions, COL_TRAIN), ("Default", default_predictions, "#c9a227")):
    by_horizon = predictions.groupby("horizon")["ape"].mean().rolling(14, center=True, min_periods=1).mean()
    axes[0].plot(by_horizon.index, by_horizon.values, color=colour, lw=2, label=label)
axes[0].axvline(90, color="black", ls=":", lw=1)
axes[0].text(92, axes[0].get_ylim()[1] * 0.95, "90 days", va="top")
axes[0].set_xlabel("Forecast horizon (days after the origin)")
axes[0].set_ylabel("Mean absolute percentage error (%)")
axes[0].set_title("Cross-validated error by horizon (14-day smoothing)")
axes[0].legend()

fold_long = fold_table.melt(id_vars="origin", var_name="Configuration", value_name="MAPE (%)")
fold_long["origin"] = fold_long["origin"].astype(str)
sns.barplot(data=fold_long, x="origin", y="MAPE (%)", hue="Configuration",
            palette={"tuned": COL_TRAIN, "default": "#c9a227"}, ax=axes[1])
axes[1].set_xlabel("Forecast origin")
axes[1].set_title("365-day MAPE at each forecast origin")
axes[1].tick_params(axis="x", rotation=45)
fig.tight_layout()
save_figure(fig, "08_cv_error_by_horizon.png")


# =============================================================================
# SECTION 7: FINAL FITS, TEST-SET EVALUATION AND DIAGNOSTICS
# =============================================================================
# The test set is touched for the first time here. Every model has already been
# fully specified; nothing below changes a modelling choice.
print_section("SECTION 7: FINAL FITS, TEST-SET EVALUATION AND DIAGNOSTICS")

# --- 7.1 Fit the Prophet models on the full training period --------------------------------
np.random.seed(RANDOM_SEED)  # Prophet draws random samples for its intervals
all_dates = pd.concat([train_frame, test_frame], ignore_index=True)


def fit_and_forecast(config, history):
    """Fit on `history`, then predict every date (in-sample and test)."""
    np.random.seed(RANDOM_SEED)  # reseeded per model, so each model's intervals are reproducible on their own
    model = fit_prophet(config, history, with_uncertainty=True)
    full = model.predict(all_dates[["ds", "post_step"]])
    return model, full


model_m1, full_m1 = fit_and_forecast(best_config, train_frame)
model_m1d, full_m1d = fit_and_forecast(default_config, train_frame)
masked_history = train_frame.copy()
masked = (masked_history["ds"] >= MASK_2009_START) & (masked_history["ds"] <= MASK_2009_END)
masked_history.loc[masked, "y"] = np.nan   # Prophet ignores rows with a missing y
model_m1s, full_m1s = fit_and_forecast(best_config, masked_history)
print(f"  Sensitivity variant: {int(masked.sum())} days ({MASK_2009_START} to {MASK_2009_END}) masked")

for key, full in (("M1", full_m1), ("M1d", full_m1d), ("M1s", full_m1s)):
    part = full[full["ds"] >= TEST_START].set_index("ds")
    test_forecasts[key] = pd.DataFrame({"yhat": part["yhat"].values, "lower": part["yhat_lower"].values,
                                        "upper": part["yhat_upper"].values}, index=test.index)

if best_config["step"] == "regressor":
    coefficients = regressor_coefficients(model_m1)
    print("\n  Step regressor learned by M1:")
    print(coefficients[["regressor", "regressor_mode", "coef"]].round(4).to_string(index=False))

# --- 7.2 Components: what has Prophet learned? -----------------------------------------------
# The decomposition is Prophet's main interpretability advantage over ARIMA.
scale, unit = (100, "%") if best_config["mode"] == "multiplicative" else (1, " GWh")
fig, axes = plt.subplots(2, 2, figsize=(15, 9.5))
ax = axes[0, 0]
ax.plot(full_m1["ds"], full_m1["trend"], color=COL_TRAIN, lw=2, label="Trend")
if best_config["step"] == "regressor":
    # The level step is carried by the regressor, not the trend, so the trend alone
    # sits well below actual consumption after 2014. Adding the step effect shows
    # the underlying level that the model actually forecasts from.
    step_effect = full_m1["post_step"]
    level_with_step = (full_m1["trend"] * (1 + step_effect) if best_config["mode"] == "multiplicative"
                       else full_m1["trend"] + step_effect)
    ax.plot(full_m1["ds"], level_with_step, color=COL_ACCENT, lw=2, ls="--", label="Trend including step effect")
ax.axvspan(pd.Timestamp(TEST_START), pd.Timestamp(TEST_END), color=COL_TEST, alpha=0.12)
ax.set_ylabel("Level (GWh/day)")
ax.set_title("Trend (shaded: test period, trend extrapolated)")
ax.legend(loc="lower left", fontsize=9)

ax = axes[0, 1]
weekday_effect = full_m1.assign(weekday=full_m1["ds"].dt.dayofweek).groupby("weekday")["weekly"].mean() * scale
ax.bar(WEEKDAY_LABELS, weekday_effect.values, color=COL_TRAIN)
ax.axhline(0, color="black", lw=0.8)
ax.set_ylabel(f"Effect ({unit.strip()})")
ax.set_title("Weekly seasonality")

ax = axes[1, 0]
one_year = full_m1[(full_m1["ds"] >= "2016-01-01") & (full_m1["ds"] <= "2016-12-31")]
ax.plot(one_year["ds"], one_year["yearly"] * scale, color=COL_TRAIN, lw=2)
ax.axhline(0, color="black", lw=0.8)
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b"))
ax.set_ylabel(f"Effect ({unit.strip()})")
ax.set_title("Annual seasonality")

ax = axes[1, 1]
holiday_on_dates = full_m1.merge(prophet_holidays, on="ds")
holiday_on_dates = holiday_on_dates[holiday_on_dates["ds"] <= TRAIN_END]
holiday_effect = holiday_on_dates.groupby("holiday")["holidays"].mean().sort_values() * scale
ax.barh(holiday_effect.index, holiday_effect.values, color=COL_TRAIN)
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel(f"Effect ({unit.strip()})")
ax.set_title("Holiday effects learned from the training data")
ax.tick_params(axis="y", labelsize=8)
fig.tight_layout()
save_figure(fig, "09_prophet_components.png")

# --- Chart 10: trend and changepoints ------------------------------------------------------------
deltas = model_m1.params["delta"].mean(axis=0)
changepoint_table = pd.DataFrame({"date": model_m1.changepoints.values, "delta": deltas})
significant = changepoint_table[changepoint_table["delta"].abs() > 0.01]   # Prophet's own plotting threshold
print(f"\n  M1 trend changepoints with a non-negligible slope change: {len(significant)} of "
      f"{len(changepoint_table)}")
print(significant.assign(date=significant["date"].dt.date).round(3).to_string(index=False))

# With a flexible trend (changepoint_prior_scale 0.2) almost every changepoint
# passes Prophet's 0.01 threshold, and drawing them all as vertical lines would
# bury the chart. The six largest slope changes are marked on the trend itself.
largest = changepoint_table.reindex(changepoint_table["delta"].abs().sort_values(ascending=False).index).head(6)
trend_by_date = full_m1.set_index("ds")["trend"]
fig, ax = plt.subplots(figsize=(14, 6.6))
ax.plot(df.index, df["Consumption"].rolling(365, center=True).mean(), color=COL_GREY, lw=1.6,
        label="365-day centred mean of actual")
ax.plot(full_m1["ds"], full_m1["trend"], color=COL_TRAIN, lw=2.2, label="Prophet trend")
if best_config["step"] == "regressor":
    ax.plot(full_m1["ds"], level_with_step, color=COL_ACCENT, lw=2, ls="--", label="Trend including step effect")
ax.scatter(largest["date"], trend_by_date.loc[largest["date"]].values, s=90, color="black", zorder=5,
           label="Six largest slope changes")
ax.axvspan(pd.Timestamp("2013-12-23"), pd.Timestamp("2014-01-14"), color=COL_ACCENT, alpha=0.25)
ax.axvspan(pd.Timestamp(MASK_2009_START), pd.Timestamp(MASK_2009_END), color=COL_TEST, alpha=0.25,
           label="2009 shortfall (masked in M1-s)")
ax.axvspan(pd.Timestamp(TEST_START), pd.Timestamp(TEST_END), color=COL_TEST, alpha=0.08, label="Test period")
ax.set_ylabel("Level (GWh/day)")
ax.set_title("Trend, step effect and changepoints (shaded red: level-step window found in the EDA)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=3, fontsize=9)
fig.tight_layout()
save_figure(fig, "10_trend_changepoints.png")

# --- 7.3 Test-set evaluation ------------------------------------------------------------------------
print("\n  ACCURACY ON THE 731-DAY TEST SET (forecast origin: 31 December 2015)")
model_order = ["M0a", "M0b", "M0c", "M1d", "M1", "M1s"]
test_metrics = pd.DataFrame(
    {MODEL_LABELS[key]: forecast_metrics(test_actual, test_forecasts[key]["yhat"],
                                         test_forecasts[key]["lower"], test_forecasts[key]["upper"])
     for key in model_order}).T
print(test_metrics.round(2).to_string())
test_metrics.round(3).to_csv(OUTPUT_DIR / "table_test_metrics.csv")

band_tables = pd.concat([horizon_band_table(MODEL_LABELS[key], test_actual, test_forecasts[key]["yhat"])
                         for key in model_order], ignore_index=True)
print("\n  By forecast horizon band, with the procurement buffer:")
print(band_tables.round(2).to_string(index=False))
band_tables.round(3).to_csv(OUTPUT_DIR / "table_horizon_bands_and_buffer.csv", index=False)
print("  Caution: every band comes from ONE forecast origin (31 December 2015), so the 1-90 day")
print("  band is always January to March (winter peak), 91-365 days is mostly spring to autumn,")
print("  and so on. Horizon is therefore confounded with season, and the bands are descriptive")
print("  only. The cross-validated error by horizon (Section 6.3: nine origins at different")
print("  times of year) is the cleaner evidence on how error grows with horizon. Note also that")
print("  2016-2017 contains no level shift, so test MAPE is more favourable than the")
print("  cross-validated MAPE, whose folds include the 2013/14 step.")

special_rows = []
for key in model_order:
    ape = np.abs(test_actual.values - test_forecasts[key]["yhat"].values) / test_actual.values * 100
    special_rows.append({"Model": MODEL_LABELS[key],
                         "MAPE on ordinary days (%)": ape[~test_special_days].mean(),
                         f"MAPE on holiday / Christmas-window days (%)": ape[test_special_days].mean()})
special_table = pd.DataFrame(special_rows)
print("\n  Ordinary days versus holidays and the Christmas-New Year window:")
print(special_table.round(2).to_string(index=False))
special_table.round(3).to_csv(OUTPUT_DIR / "table_ordinary_vs_holiday_error.csv", index=False)

pd.concat({MODEL_LABELS[key]: test_forecasts[key] for key in model_order}, axis=1).assign(
    actual=test_actual.values).round(2).to_csv(OUTPUT_DIR / "table_test_forecasts.csv")

# Sensitivity of the forecast to the 2009 shortfall
m1_mape = test_metrics.loc[MODEL_LABELS["M1"], "MAPE (%)"]
m1s_mape = test_metrics.loc[MODEL_LABELS["M1s"], "MAPE (%)"]
print(f"\n  2009 sensitivity: test MAPE {m1_mape:.2f}% with 2009 included versus {m1s_mape:.2f}% "
      f"with it masked (difference {m1s_mape - m1_mape:+.2f} percentage points)")

# --- Chart 11: the tuned Prophet forecast --------------------------------------------------------
m1_forecast = test_forecasts["M1"]
fig = plt.figure(figsize=(16, 10))
grid = fig.add_gridspec(2, 2, height_ratios=[1.15, 1])
ax = fig.add_subplot(grid[0, :])
ax.fill_between(test.index, m1_forecast["lower"], m1_forecast["upper"], color=COL_TRAIN, alpha=0.18,
                label="95% prediction interval")
ax.plot(test.index, test_actual, color="black", lw=0.7, alpha=0.8, label="Actual (daily)")
ax.plot(test.index, m1_forecast["yhat"], color=COL_TRAIN, lw=1.4, label="M1 Prophet forecast")
coverage = test_metrics.loc[MODEL_LABELS["M1"], "95% interval coverage (%)"]
ax.set_title(f"Tuned Prophet forecast of 2016-2017 from a 31 December 2015 origin "
             f"(interval covers {coverage:.1f}% of days)")
ax.set_ylabel("Consumption (GWh/day)")
ax.legend(loc="lower left", ncol=3)

ax = fig.add_subplot(grid[1, 0])
for key in ["M0a", "M0b", "M0c", "M1"]:
    error = (test_forecasts[key]["yhat"] - test_actual).rolling(28, center=True).mean()
    ax.plot(test.index, error, color=MODEL_COLOURS[key], lw=1.8, label=MODEL_LABELS[key])
ax.axhline(0, color="black", lw=0.8)
ax.set_ylabel("Forecast minus actual, 28-day mean (GWh)")
ax.set_title("Bias as the horizon lengthens")
ax.legend(fontsize=8, loc="lower left")

ax = fig.add_subplot(grid[1, 1])
window = slice("2016-12-12", "2017-01-08")
ax.plot(test_actual.loc[window].index, test_actual.loc[window], color="black", lw=2, label="Actual")
for key in ["M0c", "M1d", "M1"]:
    ax.plot(test_forecasts[key].loc[window].index, test_forecasts[key].loc[window, "yhat"],
            color=MODEL_COLOURS[key], lw=1.5, label=MODEL_LABELS[key])
ax.set_title("Christmas and New Year 2016/17")
ax.set_ylabel("Consumption (GWh/day)")
ax.tick_params(axis="x", rotation=30)
ax.legend(fontsize=8, loc="lower left")
fig.tight_layout()
save_figure(fig, "11_prophet_forecast_test.png")

# --- 7.4 Residual diagnostics (training data) -----------------------------------------------------------
# The ARIMA project's residuals were close to white noise. Prophet fits trend,
# seasonality and holidays but has no mechanism for day-to-day dependence, so
# its residuals are expected to be autocorrelated. The Ljung-Box test checks.
in_sample = full_m1[full_m1["ds"] <= TRAIN_END]
residuals = pd.Series(train_frame["y"].values - in_sample["yhat"].values, index=train_frame["ds"])
ljung_box = acorr_ljungbox(residuals, lags=[7, 14, 28], return_df=True)
print("\n  In-sample residuals of M1 (training data):")
print(f"    mean {residuals.mean():+.2f} GWh, standard deviation {residuals.std():.1f} GWh, "
      f"lag-1 autocorrelation {residuals.autocorr(1):.2f}, lag-7 {residuals.autocorr(7):.2f}")
print("    Ljung-Box test (null hypothesis: no autocorrelation):")
print(ljung_box.round(4).to_string())

fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
axes[0].plot(residuals.index, residuals.values, color=COL_GREY, lw=0.5)
axes[0].plot(residuals.index, residuals.rolling(28, center=True).mean(), color=COL_ACCENT, lw=1.8,
             label="28-day mean")
axes[0].axhline(0, color="black", lw=0.8)
axes[0].set_ylabel("Residual (GWh)")
axes[0].set_title("In-sample residuals over time")
axes[0].legend()
sns.histplot(residuals, bins=50, kde=True, color=COL_TRAIN, ax=axes[1])
axes[1].set_xlabel("Residual (GWh)")
axes[1].set_title("Residual distribution")
plot_acf(residuals, lags=60, ax=axes[2], color=COL_TRAIN)
axes[2].set_xlabel("Lag (days)")
axes[2].set_title("Residual autocorrelation")
fig.tight_layout()
save_figure(fig, "12_residual_diagnostics.png")

# --- 7.5 Interim ranking and uncertainty about it ---------------------------------------------------
print_section("INTERIM RANKING (SECTIONS 5-7)")
ranking = test_metrics["MAPE (%)"].sort_values()
print("  Test-set MAPE, best to worst:")
for name, value in ranking.items():
    print(f"    {value:5.2f}%  {name}")

print("\n  Are these differences real? Paired block bootstrap (negative = first model better):")
paired_b = paired_comparison_table([
    ("M1 Prophet vs M0c SARIMAX", "M1", "M0c"),
    ("M1 Prophet vs M1-default Prophet", "M1", "M1d"),
    ("M1 Prophet vs M0a seasonal naive", "M1", "M0a"),
    ("M0c SARIMAX vs M1-default Prophet", "M0c", "M1d"),
    ("M1-s (2009 masked) vs M1", "M1s", "M1"),
])
print(paired_b.round(2).to_string(index=False))
paired_b.round(3).to_csv(OUTPUT_DIR / "table_paired_comparisons_core.csv", index=False)

# =============================================================================
# SECTION 8: THE WIND AND SOLAR REGRESSOR EXPERIMENT (M2, M3)
# =============================================================================
print_section("SECTION 8: THE WIND AND SOLAR REGRESSOR EXPERIMENT (M2, M3)")

# --- 8.1 What is being tested, written down before any result exists -------------
# Hypotheses are recorded here, in the script, before the experiment is run, so
# that the write-up cannot quietly reshape them around the results.
print("  Hypotheses (recorded before the experiment was run):")
print("    H1  Wind and solar output add no forecasting value beyond trend, seasonality and")
print("        calendar effects: each regressor model matches its like-for-like baseline")
print("        within the noise floor set by placebo regressors.")
print("    H2  Shortening the training window so that wind and solar can be used costs more")
print("        accuracy than the regressors could win back.")
print("  The regressor models are given the ACTUAL wind and solar of the test period, which a")
print("  real forecaster would not have at the forecast origin. Their results are therefore a")
print("  perfect-foresight UPPER BOUND on what these regressors could contribute.")
print("  The experiment is run three times: with the tuned configuration, then with a less flexible")
print("  trend and with a changepoint range matched to each window (Section 8.6), to find out what")
print("  is behind any instability.")

MODEL_LABELS.update({
    "W10": "M1-W10 Prophet core, 2010-15",
    "M2": "M2 Prophet + Wind, 2010-15",
    "W12": "M1-W12 Prophet core, 2012-15",
    "M2b": "M2b Prophet + Wind, 2012-15",
    "M3": "M3 Prophet + Wind + Solar, 2012-15",
    "M1t": "M1-t Prophet (tuned on last two folds)",
})
MODEL_COLOURS.update({"W10": "#6f8fb0", "M2": COL_TEST, "W12": "#9db4cc", "M2b": "#e0a050",
                      "M3": "#b8651b", "M1t": "#4f9d9d"})
REGRESSOR_MODELS = ["M2", "M2b", "M3"]

# --- 8.2 The experiment, written as a function so it can be run twice ----------------------
# The design isolates the regressor effect from the cost of a shorter history:
#   M1-W10 / M1-W12   the core model (no regressors) retrained on the SAME shorter window
#   M2   core + Wind,           trained 2010-2015 (wind exists from 2010)
#   M2b  core + Wind,           trained 2012-2015 (same window as M3)
#   M3   core + Wind + Solar,   trained 2012-2015 (solar exists from 2012)
# Every model keeps the hyperparameters of the configuration being tested, so the
# only thing that changes between a model and its baseline is the regressors. Rows
# with a missing regressor (a handful of days) are dropped from training.
#
# The experiment is run three times (Section 8.3 and 8.6). The first run uses the
# configuration tuned in Section 6 and showed that the four-year models fail: their
# trend ends on a steep slope that is then extrapolated for two years, which makes any
# regressor comparison on that window meaningless. Two reruns then test the cause:
# a less flexible trend (changepoint_prior_scale 0.05), and a changepoint range matched to
# each window, so that the last two years before the forecast carry no changepoints, as
# they do for M1.
assert test_regressors_complete, "The test period must have complete wind and solar values"
regressor_source = df[["Wind", "Solar"]]
effect_scale, effect_unit = (100, "% of consumption") if best_config["mode"] == "multiplicative" else (1, "GWh/day")
metric_columns = ["MAE (GWh)", "RMSE (GWh)", "MAPE (%)", "R2", "Bias (GWh, forecast - actual)"]

# A fitted trend that moves by more than this share of the average level per year over
# the test period is treated as an unstable extrapolation, and comparisons that
# involve such a model are not used to judge the hypotheses.
TREND_DRIFT_LIMIT_PCT = 1.5
trend_drift_limit = TREND_DRIFT_LIMIT_PCT / 100 * consumption.mean()   # GWh per year


def run_window_model(start, regressors, source, config):
    """Fit Prophet with `config` on one training window with the given regressors.

    Returns the fitted model, the 731-day test forecast, the training history and
    the drift of the fitted trend over the test period (GWh per year). The drift
    exposes a failure mode of Prophet's linear trend: a flexible trend that ends its
    history on a steep slope extrapolates that slope for the whole forecast.
    `source` supplies the regressor values (the real series, or a placebo).
    """
    regressors = list(regressors)
    if config.get("matched_range"):
        # Prophet only allows trend changepoints in the first `changepoint_range` share of the
        # history. M1's range of 0.8 over ten years leaves the last two years free of
        # changepoints. A fixed 0.8 on a four-year window would leave only 9 months, so
        # the trend would be fitted to a few months of data at its end and extrapolated
        # for two years. The range is therefore set per window to keep 730 days free.
        window_days = (pd.Timestamp(TRAIN_END) - pd.Timestamp(start)).days + 1
        config = {**config, "range": round(1 - FREE_TREND_DAYS / window_days, 2)}
    joined = model_frame.join(source[regressors], on="ds") if regressors else model_frame.copy()
    history = joined[(joined["ds"] >= start) & (joined["ds"] <= TRAIN_END)].dropna(subset=regressors)
    future = joined[joined["ds"] >= TEST_START]
    model = fit_prophet(config, history, extra_regressors=tuple(regressors))
    forecast = model.predict(future[["ds", "post_step", *regressors]])
    trend_drift = (forecast["trend"].iloc[-1] - forecast["trend"].iloc[0]) / (len(forecast) / 365.25)
    return model, pd.Series(forecast["yhat"].values, index=test.index), history, trend_drift


def effects_per_sd(model, history, regressors):
    """Effect of a one-standard-deviation rise in each regressor (units: effect_unit)."""
    coefficients = regressor_coefficients(model).set_index("regressor")["coef"]
    return {name: coefficients[name] * history[name].std() * effect_scale for name in regressors}


# --- Placebo regressors: the noise floor --------------------------------------------------
# A tiny change in MAE after adding a regressor could be real signal, or just the
# optimiser landing somewhere slightly different. To find out how large a change
# pure noise produces, the real wind and solar series are replaced by PLACEBOS:
# the same series rotated by a random number of days. A placebo keeps the real
# distribution, seasonal shape, autocorrelation and capacity growth, but its timing
# no longer matches the demand it is meant to explain. Any gain from the real series
# that is not larger than the placebos' gains is not evidence of signal.
def shifted_placebo(series, rng):
    available = series.dropna()
    full_range = pd.date_range(available.index.min(), available.index.max(), freq="D")
    filled = available.reindex(full_range).interpolate()   # the few gap days are filled for the placebo only
    shift = int(rng.integers(180, len(filled) - 180))       # at least 180 days away from the real alignment
    return pd.Series(np.roll(filled.values, shift), index=full_range)


EXPERIMENT = [
    ("W10", wind_start, []),
    ("M2", wind_start, ["Wind"]),
    ("W12", solar_start, []),
    ("M2b", solar_start, ["Wind"]),
    ("M3", solar_start, ["Wind", "Solar"]),
]
PLACEBO_SPECS = [("M2", wind_start, ["Wind"]), ("M2b", solar_start, ["Wind"]), ("M3", solar_start, ["Wind", "Solar"])]
MATCHED_BASELINE = {"M2": "W10", "M2b": "W12", "M3": "W12"}


def run_regressor_experiment(config, suffix, title, note=""):
    """Run the whole regressor experiment for one Prophet configuration.

    `suffix` is appended to every model key ("" for the primary run, "c" and "m" for
    the reruns) so that all the forecasts live side by side in `test_forecasts`.
    `note` is added to the model labels. Returns a dictionary of the result tables.
    """
    key = lambda base: base + suffix
    for base in ["W10", "M2", "W12", "M2b", "M3"] + ([] if not suffix else ["M1"]):
        MODEL_LABELS[key(base)] = MODEL_LABELS[base] + note
        MODEL_COLOURS[key(base)] = MODEL_COLOURS[base]
    range_text = "matched to each window" if config.get("matched_range") else str(config["range"])
    print(f"\n  ===== {title} (changepoint_prior_scale {config['cps']}, changepoint_range {range_text}) =====")

    # The full-history core model. For the primary run this is M1 from Section 7; for
    # a rerun it has to be refitted with the new configuration.
    drifts = {}
    if suffix:
        _, forecast_series, _, drifts[key("M1")] = run_window_model(pd.Timestamp(TRAIN_START), [], regressor_source, config)
        test_forecasts[key("M1")] = pd.DataFrame({"yhat": forecast_series.values, "lower": np.nan, "upper": np.nan},
                                                 index=test.index)
    else:
        m1_trend = full_m1[full_m1["ds"] >= TEST_START]["trend"]
        drifts["M1"] = (m1_trend.iloc[-1] - m1_trend.iloc[0]) / (len(m1_trend) / 365.25)

    real_rows = []
    for base, start, regressors in EXPERIMENT:
        model, forecast_series, history, drifts[key(base)] = run_window_model(start, regressors, regressor_source, config)
        test_forecasts[key(base)] = pd.DataFrame({"yhat": forecast_series.values, "lower": np.nan, "upper": np.nan},
                                                 index=test.index)
        for name, effect in effects_per_sd(model, history, regressors).items():
            real_rows.append({"model": base, "regressor": name, "effect": effect})
        print(f"  Fitted {MODEL_LABELS[key(base)]}: {len(history):,} training days")
    real_effects = pd.DataFrame(real_rows)

    order = ["M1", "W10", "M2", "W12", "M2b", "M3"]
    metrics = pd.DataFrame({MODEL_LABELS[key(b)]: forecast_metrics(test_actual, test_forecasts[key(b)]["yhat"])
                            for b in order}).T[metric_columns]
    print("\n  Test-set accuracy (regressor models use perfect-foresight wind/solar):")
    print(metrics.round(2).to_string())

    # Why can one window do so much worse than another? Compare how far each model's
    # trend drifts over the test period.
    drift_table = pd.DataFrame({"Trend drift over test period (GWh/year)": {MODEL_LABELS[key(b)]: drifts[key(b)] for b in order}})
    drift_table["Within limit"] = drift_table.iloc[:, 0].abs() <= trend_drift_limit
    print(f"\n  Trend extrapolation behind the accuracy differences (limit: +/-{trend_drift_limit:.0f} GWh/year, "
          f"{TREND_DRIFT_LIMIT_PCT}% of the average level):")
    print(drift_table.round(1).to_string())
    print(f"\n  Learned effect of a one-standard-deviation rise ({effect_unit}):")
    print(real_effects.assign(effect=real_effects["effect"].round(3)).to_string(index=False))

    # Placebo draws: identical shifts in every run (same seed), so runs are comparable.
    rng = np.random.default_rng(RANDOM_SEED)
    fit_rows, effect_rows = [], []
    print(f"\n  Fitting {N_PLACEBOS} placebo draws x {len(PLACEBO_SPECS)} model types ...")
    for draw in range(N_PLACEBOS):
        placebo_source = pd.DataFrame({name: shifted_placebo(df[name], rng)
                                       for name in ("Wind", "Solar")}).reindex(df.index)
        for base, start, regressors in PLACEBO_SPECS:
            model, forecast_series, history, _ = run_window_model(start, regressors, placebo_source, config)
            fit_rows.append({"model": base, "draw": draw,
                             "MAE": float(np.mean(np.abs(test_actual.values - forecast_series.values)))})
            for name, effect in effects_per_sd(model, history, regressors).items():
                effect_rows.append({"model": base, "draw": draw, "regressor": name, "effect": effect})
    placebo_fits, placebo_effects = pd.DataFrame(fit_rows), pd.DataFrame(effect_rows)

    noise_rows = []
    for base in REGRESSOR_MODELS:
        baseline_mae = metrics.loc[MODEL_LABELS[key(MATCHED_BASELINE[base])], "MAE (GWh)"]
        real_mae = metrics.loc[MODEL_LABELS[key(base)], "MAE (GWh)"]
        placebo_mae = placebo_fits.loc[placebo_fits["model"] == base, "MAE"]
        noise_rows.append({
            "Model": MODEL_LABELS[key(base)], "Baseline MAE": baseline_mae, "Real MAE": real_mae,
            "Real change": real_mae - baseline_mae,
            "Placebo change, min": (placebo_mae - baseline_mae).min(),
            "Placebo change, median": (placebo_mae - baseline_mae).median(),
            "Placebo change, max": (placebo_mae - baseline_mae).max(),
            "Placebos at least as good (%)": (placebo_mae <= real_mae).mean() * 100,
        })
    noise_table = pd.DataFrame(noise_rows)
    print("\n  Real regressors against the placebo noise floor (change in test MAE, GWh; negative = better):")
    print(noise_table.round(2).to_string(index=False))

    # Paired block bootstrap: regressor effect, and the cost of a shorter history
    comparisons = [
        ("M2 (+Wind) vs M1-W10", "M2", "W10"),
        ("M2b (+Wind) vs M1-W12", "M2b", "W12"),
        ("M3 (+Wind+Solar) vs M1-W12", "M3", "W12"),
        ("M1-W10 (6 years) vs M1 (10 years)", "W10", "M1"),
        ("M1-W12 (4 years) vs M1 (10 years)", "W12", "M1"),
        ("M2 (+Wind) vs M1 (10 years)", "M2", "M1"),
        ("M3 (+Wind+Solar) vs M1 (10 years)", "M3", "M1"),
    ]
    paired = paired_comparison_table([(label, key(a), key(b)) for label, a, b in comparisons])
    paired["Both trends within limit"] = [abs(drifts[key(a)]) <= trend_drift_limit and abs(drifts[key(b)]) <= trend_drift_limit
                                          for _, a, b in comparisons]
    print("\n  Paired block bootstrap, change in MAE in GWh (negative = first model better):")
    print(paired.round(2).to_string(index=False))

    # The hypotheses, judged only on comparisons whose models have a stable trend
    regressor_rows = paired.iloc[:3]
    sound = regressor_rows[regressor_rows["Both trends within limit"]]
    print("\n  Verdicts use only comparisons in which both models' trends are within the drift limit.")
    if len(sound) == 0:
        print("  H1: no regressor comparison is sound in this run, so H1 cannot be judged.")
    else:
        includes_zero = int(((sound["95% CI low"] <= 0) & (sound["95% CI high"] >= 0)).sum())
        sound_models = [a for (_, a, _), ok in zip(comparisons[:3], regressor_rows["Both trends within limit"]) if ok]
        beaten = noise_table[noise_table["Model"].isin([MODEL_LABELS[key(a)] for a in sound_models])]
        within_noise = int((beaten["Placebos at least as good (%)"] > 5).sum())
        print(f"  H1: of {len(sound)} sound regressor comparison(s), the 95% interval for the change in MAE")
        print(f"      includes zero in {includes_zero}, and the real regressors are not better than 95% of "
              f"placebos in {within_noise}.")
    truncation = paired.iloc[3:5]
    print("  H2: cost of a shorter history, change in MAE (GWh): "
          + "; ".join(f"{row['Comparison (A vs B)']}: {row['MAE(A) - MAE(B) (GWh)']:+.1f} "
                      f"[{row['95% CI low']:+.1f}, {row['95% CI high']:+.1f}]"
                      + ("" if row["Both trends within limit"] else " (unstable trend, not interpretable)")
                      for _, row in truncation.iterrows()))

    tag = {"": "", "c": "_conservative", "m": "_matched_range"}[suffix]
    metrics.round(3).to_csv(OUTPUT_DIR / f"table_regressor_experiment{tag}.csv")
    drift_table.round(2).to_csv(OUTPUT_DIR / f"table_trend_drift{tag}.csv")
    placebo_fits.round(3).to_csv(OUTPUT_DIR / f"table_placebo_fits{tag}.csv", index=False)
    noise_table.round(3).to_csv(OUTPUT_DIR / f"table_placebo_noise_floor{tag}.csv", index=False)
    paired.round(3).to_csv(OUTPUT_DIR / f"table_paired_comparisons_regressors{tag}.csv", index=False)
    return {"suffix": suffix, "config": config, "metrics": metrics, "drift_table": drift_table, "drifts": drifts,
            "real_effects": real_effects, "placebo_fits": placebo_fits, "placebo_effects": placebo_effects,
            "noise": noise_table, "paired": paired}


# --- 8.3 Primary run: the configuration tuned in Section 6 ---------------------------------
primary = run_regressor_experiment(best_config, "", "PRIMARY RUN")
experiment_metrics, real_effects = primary["metrics"], primary["real_effects"]
placebo_fits, placebo_effects = primary["placebo_fits"], primary["placebo_effects"]
noise_table, paired_c = primary["noise"], primary["paired"]
print("\n  In the primary run the three 2012-2015 models carry a steep extrapolated trend (see the drift")
print("  table), so their results say more about trend extrapolation than about wind or solar. The")
print("  six-year comparison (M2 against M1-W10) is the clean one in this run. Section 8.6 repeats the")
print("  experiment twice to test what is causing the instability.")

# --- 8.4 Do the regressors explain what the core model gets wrong? -----------------------------
# A second, independent look: correlate the core model's out-of-sample errors with
# wind and solar after removing year level, weekday pattern and annual cycle (the same
# adjustment as in the EDA). If the regressors carried information the core model
# lacks, its errors would be related to them.
m1_error = test_actual - test_forecasts["M1"]["yhat"]
error_frame = pd.DataFrame({"M1 forecast error": m1_error.values, "Wind": test["Wind"].values,
                            "Solar": test["Solar"].values}, index=test.index)
adjusted_errors = residualise(error_frame, ["M1 forecast error", "Wind", "Solar"])
error_correlations = []
for name in ("Wind", "Solar"):
    raw_r = error_frame["M1 forecast error"].corr(error_frame[name])
    adj_r, low, high = correlation_with_interval(adjusted_errors["M1 forecast error"], adjusted_errors[name])
    error_correlations.append({"Regressor": name, "Raw r": raw_r, "Adjusted r": adj_r, "CI low": low, "CI high": high})
error_correlations = pd.DataFrame(error_correlations)
print("\n  Correlation of M1's test-set forecast errors with wind and solar (test period, 731 days):")
print(error_correlations.round(3).to_string(index=False))
print("  (The interval is optimistic: daily residuals are autocorrelated.)")
error_correlations.round(4).to_csv(OUTPUT_DIR / "table_error_vs_regressors.csv", index=False)

# --- 8.5 Sensitivity to the tuning rule -----------------------------------------------------------
# M1 was chosen by the lowest MAPE averaged over all nine folds, a rule fixed before
# the test set was touched. The two latest folds are the only ones that can show
# whether a configuration copes with the level step, and they favour a different
# configuration. It is fitted here as a clearly labelled SENSITIVITY check. It is not
# a replacement for M1, because switching rule after seeing test results would be
# a forking path.
alt_row = tuning.sort_values("last2_MAPE").iloc[0]
alt_config = {"cps": float(alt_row["cps"]), "range": float(alt_row["range"]), "mode": str(alt_row["mode"]),
              "step": str(alt_row["step"]), "fourier": int(alt_row["fourier"])}
_, full_alt = fit_and_forecast(alt_config, train_frame)
alt_part = full_alt[full_alt["ds"] >= TEST_START].set_index("ds")
test_forecasts["M1t"] = pd.DataFrame({"yhat": alt_part["yhat"].values, "lower": alt_part["yhat_lower"].values,
                                      "upper": alt_part["yhat_upper"].values}, index=test.index)
alt_metrics = forecast_metrics(test_actual, test_forecasts["M1t"]["yhat"])
print(f"\n  Sensitivity to the tuning rule: configuration with the best last-two-fold MAPE")
print(f"    {alt_config}")
print(f"    test MAPE {alt_metrics['MAPE (%)']:.2f}% and MAE {alt_metrics['MAE (GWh)']:.1f} GWh, against "
      f"{test_metrics.loc[MODEL_LABELS['M1'], 'MAPE (%)']:.2f}% and "
      f"{test_metrics.loc[MODEL_LABELS['M1'], 'MAE (GWh)']:.1f} GWh for M1")
paired_t = paired_comparison_table([("M1-t vs M1", "M1t", "M1")])
print(paired_t.round(2).to_string(index=False))

# --- Shared drawing helper: paired-bootstrap forest plot with placebo noise floor ------------
def draw_forest(ax, experiment, jitter_generator):
    """One row per paired comparison: point = MAE change, whisker = 95% bootstrap interval.

    Red rows involve a model whose trend drifts beyond the limit (not interpretable).
    Grey dots are the MAE changes produced by placebo regressors: the noise floor.
    """
    paired, metrics, suffix = experiment["paired"], experiment["metrics"], experiment["suffix"]
    y_positions = np.arange(len(paired))[::-1]
    labelled = set()
    for y, (_, row) in zip(y_positions, paired.iterrows()):
        sound = bool(row["Both trends within limit"])
        label = None if sound in labelled else ("Both trends stable" if sound else "A trend is unstable")
        labelled.add(sound)
        mean = row["MAE(A) - MAE(B) (GWh)"]
        ax.errorbar(mean, y, xerr=[[mean - row["95% CI low"]], [row["95% CI high"] - mean]], fmt="o",
                    color=COL_TRAIN if sound else COL_ACCENT, capsize=4, lw=1.6, ms=8, zorder=4, label=label)
    for y, base in zip(y_positions[:3], ["M2", "M2b", "M3"]):
        baseline_mae = metrics.loc[MODEL_LABELS[MATCHED_BASELINE[base] + suffix], "MAE (GWh)"]
        change = experiment["placebo_fits"].loc[experiment["placebo_fits"]["model"] == base, "MAE"] - baseline_mae
        ax.scatter(change, y + jitter_generator.uniform(-0.18, 0.18, len(change)), color=COL_GREY, s=28, alpha=0.8,
                   label="Placebo regressors" if base == "M2" else None, zorder=3)
    ax.axvline(0, color="black", lw=0.8)
    # Differences can range from fractions of a GWh to tens of GWh, so the axis is linear
    # within +/-5 GWh and logarithmic beyond that. Without this the small differences,
    # which are the point of the experiment, would be squashed against zero.
    ax.set_xscale("symlog", linthresh=5)
    ax.set_xticks([-5, -2, 0, 2, 5, 10, 30, 60, 100])
    ax.set_xticklabels(["-5", "-2", "0", "2", "5", "10", "30", "60", "100"])
    ax.set_yticks(y_positions)
    ax.set_yticklabels(paired["Comparison (A vs B)"], fontsize=9)
    ax.set_xlabel("Change in test MAE (GWh/day): negative = first model better\n(axis linear within +/-5, logarithmic beyond)")
    ax.legend(loc="upper right", fontsize=9)


# --- Chart 13: learned regressor effects against placebo effects ---------------------------------
categories = [("M2", "Wind", "M2: Wind\n(2010-15)"), ("M2b", "Wind", "M2b: Wind\n(2012-15)"),
              ("M3", "Wind", "M3: Wind\n(2012-15)"), ("M3", "Solar", "M3: Solar\n(2012-15)")]
jitter_rng = np.random.default_rng(RANDOM_SEED)
fig, ax = plt.subplots(figsize=(11, 6))
for position, (key, name, label) in enumerate(categories):
    placebo_values = placebo_effects[(placebo_effects["model"] == key) & (placebo_effects["regressor"] == name)]["effect"]
    real_value = real_effects[(real_effects["model"] == key) & (real_effects["regressor"] == name)]["effect"].iloc[0]
    ax.scatter(position + jitter_rng.uniform(-0.13, 0.13, len(placebo_values)), placebo_values, color=COL_GREY,
               s=45, alpha=0.8, label="Placebo (rotated series)" if position == 0 else None)
    ax.scatter(position, real_value, marker="D", s=140, color=COL_ACCENT, zorder=5,
               label="Real regressor" if position == 0 else None)
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(range(len(categories)))
ax.set_xticklabels([label for _, _, label in categories])
ax.set_ylabel(f"Effect of a one standard deviation rise ({effect_unit})")
ax.set_title("Learned regressor effects against placebo effects")
ax.legend(loc="best")
fig.tight_layout()
save_figure(fig, "13_regressor_coefficients.png")

# --- Chart 14: accuracy and uncertainty of the regressor comparisons ----------------------------------
fig, axes = plt.subplots(1, 2, figsize=(17, 6.2), gridspec_kw={"width_ratios": [1, 1.25]})
bar_keys = ["M1", "W10", "M2", "W12", "M2b", "M3"]
bar_values = [experiment_metrics.loc[MODEL_LABELS[key], "MAE (GWh)"] for key in bar_keys]
bars = axes[0].bar(range(len(bar_keys)), bar_values, color=[MODEL_COLOURS[key] for key in bar_keys],
                   hatch=None)
for bar, key in zip(bars, bar_keys):
    if key in REGRESSOR_MODELS:
        bar.set_hatch("//")
        bar.set_edgecolor("white")
for position, value in enumerate(bar_values):
    axes[0].text(position, value + 0.6, f"{value:.1f}", ha="center")
axes[0].set_xticks(range(len(bar_keys)))
axes[0].set_xticklabels(["M1\n10 yrs\ncore", "M1-W10\n6 yrs\ncore", "M2\n6 yrs\n+Wind", "M1-W12\n4 yrs\ncore",
                         "M2b\n4 yrs\n+Wind", "M3\n4 yrs\n+Wind\n+Solar"])
axes[0].set_ylabel("Test-set MAE (GWh/day)")
axes[0].set_ylim(0, max(bar_values) * 1.2)
axes[0].set_title("Test accuracy (hatched: perfect-foresight regressors)")

draw_forest(axes[1], primary, jitter_rng)
axes[1].set_title("Paired bootstrap 95% intervals, with the placebo noise floor")
fig.tight_layout()
save_figure(fig, "14_regressor_experiment_comparison.png")

# --- Chart 15: do wind and solar explain the core model's errors? ------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
for ax, name, colour in zip(axes, ("Wind", "Solar"), ("#3a8f6e", "#c9a227")):
    sns.regplot(x=adjusted_errors[name], y=adjusted_errors["M1 forecast error"], scatter_kws={"s": 12, "alpha": 0.4, "color": colour},
                line_kws={"color": COL_ACCENT, "lw": 2}, ax=ax)
    row = error_correlations[error_correlations["Regressor"] == name].iloc[0]
    ax.set_title(f"{name}: adjusted r = {row['Adjusted r']:+.2f} (raw r = {row['Raw r']:+.2f})")
    ax.set_xlabel(f"{name}, adjusted for year level, weekday and annual cycle (GWh)")
    ax.set_ylabel("M1 forecast error, adjusted (GWh)")
fig.suptitle("Test period: is the core model's error related to wind or solar output?")
fig.tight_layout()
save_figure(fig, "15_residuals_vs_regressors.png")

# --- 8.6 Robustness: what is causing the four-year instability? ------------------------------
# Two reruns, each changing ONE thing relative to the primary run:
#   conservative  changepoint_prior_scale 0.05 (Prophet's default) instead of 0.2: a trend
#                 that bends less. Tests whether trend flexibility is the cause.
#   matched range changepoint_range set per window so that the last 730 days carry no
#                 changepoints, as in M1. Tests whether the cause is that a four-year
#                 window under the fixed range of 0.8 fits its trend to the last few months
#                 and extrapolates that slope for two years.
# These are robustness checks on the experiment, not a new tuning of M1: M1 itself keeps
# the configuration chosen before the test set was used.
neighbour = tuning[(tuning["cps"] == 0.05) & (tuning["range"] == best_config["range"]) &
                   (tuning["mode"] == best_config["mode"]) & (tuning["step"] == best_config["step"]) &
                   (tuning["fourier"] == 10)]
if len(neighbour):
    print(f"\n  For reference, the nearest configuration in the Section 6 grid (cps 0.05, Fourier order 10) had "
          f"cross-validated MAPE {neighbour['mean_MAPE'].iloc[0]:.2f}% against {best['mean_MAPE']:.2f}% for M1.")
conservative = run_regressor_experiment({**best_config, "cps": 0.05}, "c", "RERUN 1: CONSERVATIVE TREND",
                                        note=" [cps 0.05]")
matched = run_regressor_experiment({**best_config, "matched_range": True}, "m", "RERUN 2: MATCHED CHANGEPOINT RANGE",
                                   note=" [matched range]")

# Side-by-side summary of the three runs
order = ["M1", "W10", "M2", "W12", "M2b", "M3"]
runs = [("cps 0.2, range 0.8", primary, ""), ("cps 0.05, range 0.8", conservative, "c"),
        ("cps 0.2, matched range", matched, "m")]
comparison_rows = []
for base in order:
    row = {"Model": MODEL_LABELS[base]}
    for run_label, result, suffix in runs:
        row[f"MAE: {run_label}"] = result["metrics"].loc[MODEL_LABELS[base + suffix], "MAE (GWh)"]
        row[f"Drift: {run_label}"] = result["drifts"][base + suffix]
    comparison_rows.append(row)
run_comparison = pd.DataFrame(comparison_rows)
print("\n  THE THREE RUNS SIDE BY SIDE (MAE in GWh/day, trend drift in GWh/year)")
print(run_comparison.round(1).to_string(index=False))
run_comparison.round(2).to_csv(OUTPUT_DIR / "table_runs_comparison.csv", index=False)

# --- Chart 17: the reruns -----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(23, 6.6), gridspec_kw={"width_ratios": [1, 1, 1.35]})
x = np.arange(len(order))
short_labels = ["M1\n10 yrs", "M1-W10\n6 yrs", "M2\n+Wind", "M1-W12\n4 yrs", "M2b\n+Wind", "M3\n+Wind\n+Solar"]
run_colours = [COL_TRAIN, COL_TEST, "#2e8b57"]
for axis, prefix, ylabel, title in ((axes[0], "Drift", "Trend drift over test period (GWh/year)", "Fitted trend drift"),
                                    (axes[1], "MAE", "Test-set MAE (GWh/day)", "Test accuracy")):
    for offset, (run_label, _, _), colour in zip((-0.27, 0, 0.27), runs, run_colours):
        axis.bar(x + offset, run_comparison[f"{prefix}: {run_label}"], 0.27, color=colour, label=run_label)
    axis.axhline(0, color="black", lw=0.8)
    axis.set_xticks(x)
    axis.set_xticklabels(short_labels, fontsize=9)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
axes[0].axhline(-trend_drift_limit, color=COL_ACCENT, ls="--", lw=1, label="Stability limit")
axes[0].axhline(trend_drift_limit, color=COL_ACCENT, ls="--", lw=1)
axes[0].legend(fontsize=8, loc="lower left")
axes[1].legend(fontsize=8, loc="upper left")
draw_forest(axes[2], matched, jitter_rng)
axes[2].set_title("Matched changepoint range: paired bootstrap with placebo noise floor")
fig.tight_layout()
save_figure(fig, "17_trend_instability_reruns.png")

# --- 8.7 Every model, one table and one chart ----------------------------------------------------------
# The shorter-window models shown here are the MATCHED-RANGE versions (suffix "m"). The primary-run
# versions of the 2012-2015 models fail because of a trend-extrapolation artefact (Section 8.3), so
# they would mislead in a model comparison.
all_keys = ["M0a", "M0b", "M0c", "M1d", "M1", "M1s", "M1t", "W10m", "M2m", "W12m", "M2bm", "M3m"]
all_metrics = pd.DataFrame(
    {MODEL_LABELS[key]: forecast_metrics(test_actual, test_forecasts[key]["yhat"]) for key in all_keys}).T[metric_columns]
all_metrics = all_metrics.sort_values("MAPE (%)")
print_section("ALL MODELS ON THE 731-DAY TEST SET (best to worst MAPE)")
print(all_metrics.round(2).to_string())
print("\n  Models M2, M2b and M3 use perfect-foresight wind and solar: an upper bound, not a forecast.")
print("  The shorter-window models shown are the matched-range versions (Section 8.6).")
all_metrics.round(3).to_csv(OUTPUT_DIR / "table_all_models_test_metrics.csv")
pd.concat({MODEL_LABELS[key]: test_forecasts[key] for key in all_keys}, axis=1).assign(
    actual=test_actual.values).round(2).to_csv(OUTPUT_DIR / "table_test_forecasts.csv")

label_to_key = {MODEL_LABELS[key]: key for key in all_keys}
fig, ax = plt.subplots(figsize=(13, 7.2))
order = list(all_metrics.index)[::-1]
bars = ax.barh(order, all_metrics.loc[order, "MAPE (%)"], color=[MODEL_COLOURS[label_to_key[name]] for name in order])
for bar, name in zip(bars, order):
    if label_to_key[name].rstrip("m") in REGRESSOR_MODELS:   # M2m, M2bm, M3m: perfect-foresight regressors
        bar.set_hatch("//")
        bar.set_edgecolor("white")
    ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
            f"{all_metrics.loc[name, 'MAPE (%)']:.2f}%  (MAE {all_metrics.loc[name, 'MAE (GWh)']:.1f})",
            va="center", fontsize=9)
ax.set_xlabel("Test-set MAPE (%)")
ax.set_xlim(0, all_metrics["MAPE (%)"].max() * 1.22)
ax.set_title("All models on the 2016-2017 test set (hatched: perfect-foresight regressors, an upper bound)")
fig.tight_layout()
save_figure(fig, "16_model_comparison_all.png")

# print_section("CHECKPOINT C COMPLETE")
# print(f"  Charts 13-17 and the tables are in: {OUTPUT_DIR}")


# =============================================================================
# 8.8 ADDENDUM TO SECTION 8: REGRESSOR EFFECTS FROM THE SOUND (MATCHED-RANGE) RUN
# =============================================================================
# Chart 13 was drawn from the primary run, whose 2012-2015 models carry an unstable
# trend, so its placebo comparison for M2b and M3 is less trustworthy. This chart
# repeats it with the matched-range run, in which all six models are stable. It is
# the version to quote.
print_section("8.8 ADDENDUM: REGRESSOR EFFECTS FROM THE MATCHED-RANGE RUN")

matched_real, matched_placebo = matched["real_effects"], matched["placebo_effects"]
fig, ax = plt.subplots(figsize=(11, 6))
for position, (key, name, label) in enumerate(categories):   # `categories` was defined for chart 13
    placebo_values = matched_placebo[(matched_placebo["model"] == key) &
                                     (matched_placebo["regressor"] == name)]["effect"]
    real_value = matched_real[(matched_real["model"] == key) & (matched_real["regressor"] == name)]["effect"].iloc[0]
    ax.scatter(position + jitter_rng.uniform(-0.13, 0.13, len(placebo_values)), placebo_values,
               color=COL_GREY, s=45, alpha=0.8, label="Placebo (rotated series)" if position == 0 else None)
    ax.scatter(position, real_value, marker="D", s=140, color=COL_ACCENT, zorder=5,
               label="Real regressor" if position == 0 else None)
    outside = real_value < placebo_values.min() or real_value > placebo_values.max()
    print(f"  {label.replace(chr(10), ' ')}: real effect {real_value:+.3f} {effect_unit} per standard deviation; "
          f"placebo range {placebo_values.min():+.3f} to {placebo_values.max():+.3f}: "
          f"{'OUTSIDE the placebo range' if outside else 'within the placebo range'}")
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(range(len(categories)))
ax.set_xticklabels([label for _, _, label in categories])
ax.set_ylabel(f"Effect of a one standard deviation rise ({effect_unit})")
ax.set_title("Learned regressor effects against placebo effects (matched-range run)")
ax.legend(loc="best")
fig.tight_layout()
save_figure(fig, "18_regressor_coefficients_matched_range.png")
print("  A real effect outside the placebo range suggests the series is associated with demand. Whether that")
print("  association is large enough to improve the forecast is a separate question, answered in Section 8.6.")


# =============================================================================
# SECTION 9: THE AIR PASSENGERS BRIDGE (PROPHET VERSUS ARIMA)
# =============================================================================
# The ARIMA project forecast the Air Passengers series with ARIMA and suggested, in
# its Next Steps, benchmarking Prophet on the same train/test split. This section
# does that. The aim is a like-for-like methodological comparison, not a claim about
# which technique is better: 144 monthly points of one benchmark series say little about
# business data with holidays, regressors and level shifts, and the sections above
# are where Prophet's features are actually exercised.
print_section("SECTION 9: THE AIR PASSENGERS BRIDGE (PROPHET VERSUS ARIMA)")

# Imports used only in this section
from scipy.special import inv_boxcox
from scipy.stats import boxcox
from statsmodels.tsa.arima.model import ARIMA

# --- 9.1 Load the data and apply the ARIMA project's split ---------------------------
# The ARIMA project read 'AirPassengers.csv' (the Kaggle file) from its working folder.
# If that file is next to this script it is used, so the data is identical. Otherwise a
# public copy of the same series is downloaded.
AP_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
ap_candidates = [BASE_DIR / "AirPassengers.csv", BASE_DIR / "airline-passengers.csv"]
ap_path = next((path for path in ap_candidates if path.exists()), None)
if ap_path is None:
    ap_path = ap_candidates[1]
    print(f"  Downloading Air Passengers from {AP_URL}")
    urllib.request.urlretrieve(AP_URL, ap_path)
ap_raw = pd.read_csv(ap_path)
ap_df = pd.DataFrame({"Passengers": ap_raw.iloc[:, 1].astype(float).values},
                     index=pd.to_datetime(ap_raw.iloc[:, 0]))
ap_df.index.name = "Month"
ap_df.index.freq = "MS"   # strictly regular month-start index (also silences a statsmodels warning)

# Validation, in the same spirit as Section 1: the series must be the classic one.
assert len(ap_df) == 144, f"Expected 144 monthly observations, found {len(ap_df)}"
assert ap_df.index.min() == pd.Timestamp("1949-01-01") and ap_df.index.max() == pd.Timestamp("1960-12-01")
assert int(ap_df["Passengers"].sum()) == 40363, "The series does not match the classic Air Passengers data"
print(f"  Loaded {len(ap_df)} monthly observations from {ap_path.name} "
      f"({ap_df.index.min():%Y-%m} to {ap_df.index.max():%Y-%m}); checks passed")

# Chronological 80/20 split, computed exactly as in the ARIMA project: 115 training, 29 test months.
ap_train_size = int(len(ap_df) * 0.8)
ap_train, ap_test = ap_df.iloc[:ap_train_size], ap_df.iloc[ap_train_size:]
ap_actual = ap_test["Passengers"].values
print(f"  Training: {len(ap_train)} months ({ap_train.index.min():%Y-%m} to {ap_train.index.max():%Y-%m}); "
      f"test: {len(ap_test)} months ({ap_test.index.min():%Y-%m} to {ap_test.index.max():%Y-%m})")

# Figures published on the ARIMA project page, kept for reference
ARIMA_PUBLISHED = {
    "A_base": {"MAE": 32.34, "RMSE": 41.04, "R2": 0.7241, "MAPE": 6.96},
    "A_opt": {"MAE": 14.89, "RMSE": 17.98, "R2": 0.9470, "MAPE": 3.46},
}

ap_forecasts = {}   # key -> DataFrame(yhat, lower, upper) over the 29 test months
AP_LABELS = {
    "A_base": "ARIMA(12,1,12), untransformed",
    "A_opt": "ARIMA(12,2,12), Box-Cox",
    "P_add": "Prophet, additive (defaults)",
    "P_mult": "Prophet, multiplicative (defaults)",
    "P_bc": "Prophet, Box-Cox + additive",
    "P_tuned": "Prophet, tuned by cross-validation",
}
AP_COLOURS = {"A_base": "#8c564b", "A_opt": "#b5654f", "P_add": "#9db4cc", "P_mult": "#6f8fb0",
              "P_bc": "#4f9d9d", "P_tuned": COL_TRAIN}

# --- 9.2 The ARIMA models, re-run so the comparison is like for like -------------------
# The two ARIMA models are refitted here with the same code as the ARIMA project, which
# (a) checks that the data and split match and (b) puts both techniques through the same
# library versions. ARIMA(12,2,12) has 24 parameters on 115 observations and the
# optimiser does not fully converge, so small differences from the published figures
# are expected and are reported rather than hidden.
ap_lambda = None
for key, order, use_boxcox in (("A_base", (12, 1, 12), False), ("A_opt", (12, 2, 12), True)):
    if use_boxcox:
        transformed, ap_lambda = boxcox(ap_train["Passengers"])   # lambda estimated on the TRAINING data only
        series = pd.Series(transformed, index=ap_train.index)
    else:
        series = ap_train["Passengers"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # statsmodels convergence warnings; the flag is printed below instead
        fitted = ARIMA(series, order=order).fit()
        result = fitted.get_forecast(steps=len(ap_test))
    mean, interval = result.predicted_mean.values, result.conf_int(alpha=0.05).values
    if use_boxcox:
        mean, interval = inv_boxcox(mean, ap_lambda), inv_boxcox(interval, ap_lambda)
    ap_forecasts[key] = pd.DataFrame({"yhat": mean, "lower": interval[:, 0], "upper": interval[:, 1]},
                                     index=ap_test.index)
    print(f"  {AP_LABELS[key]}: optimiser converged: {fitted.mle_retvals.get('converged')}"
          + (f"; Box-Cox lambda {ap_lambda:.3f}" if use_boxcox else ""))

# --- 9.3 The Prophet models ----------------------------------------------------------------
# Monthly data, so weekly and daily seasonality are switched off. The three fixed variants
# map onto the ARIMA project's story:
#   P_add    additive seasonality on the raw series        <-> the untransformed ARIMA baseline
#   P_mult   Prophet's own answer to growing seasonal swings: multiplicative seasonality
#   P_bc     the SAME Box-Cox transformation the optimal ARIMA used, then additive Prophet
# A fourth model, P_tuned, is chosen by cross-validation inside the training data (9.4).
def ap_make_prophet(mode, cps, fourier, with_uncertainty):
    return Prophet(growth="linear", seasonality_mode=mode, changepoint_prior_scale=cps,
                   yearly_seasonality=int(fourier), weekly_seasonality=False, daily_seasonality=False,
                   interval_width=0.95, uncertainty_samples=1000 if with_uncertainty else 0)


def ap_frame(index, values):
    return pd.DataFrame({"ds": index, "y": values})


def ap_fit_forecast(mode, cps, fourier, use_boxcox=False):
    """Fit on the training months and forecast the 29 test months with 95% intervals."""
    np.random.seed(RANDOM_SEED)   # Prophet draws random samples for its intervals
    if use_boxcox:
        transformed, lam = boxcox(ap_train["Passengers"])      # lambda from the training data only
        history = ap_frame(ap_train.index, transformed)
    else:
        history = ap_frame(ap_train.index, ap_train["Passengers"].values)
    model = ap_make_prophet(mode, cps, fourier, with_uncertainty=True)
    model.fit(history)
    prediction = model.predict(pd.DataFrame({"ds": ap_test.index}))
    columns = prediction[["yhat", "yhat_lower", "yhat_upper"]].values
    if use_boxcox:
        columns = inv_boxcox(columns, lam)
    return model, pd.DataFrame(columns, columns=["yhat", "lower", "upper"], index=ap_test.index)


# --- 9.4 Tuning by walk-forward cross-validation (training data only) --------------------------
# Rolling origins inside the 115 training months, each forecasting 29 months ahead (the
# same horizon as the test set). Every origin has at least 48 months of history, and the
# origins are 6 months apart, working back from the last one that leaves 29 months to
# forecast inside the training data. The test months play no part.
AP_CV_HORIZON, AP_CV_INITIAL, AP_CV_STEP = len(ap_test), 48, 6
ap_cutoffs = list(range(ap_train_size - AP_CV_HORIZON, AP_CV_INITIAL - 1, -AP_CV_STEP))[::-1]
print(f"\n  Cross-validation: {len(ap_cutoffs)} rolling origins, each forecasting {AP_CV_HORIZON} months, "
      f"first origin after {ap_cutoffs[0]} months of history")


def ap_cv_mape(mode, cps, fourier):
    scores = []
    for cutoff in ap_cutoffs:
        history = ap_frame(ap_train.index[:cutoff], ap_train["Passengers"].values[:cutoff])
        future = ap_train.iloc[cutoff:cutoff + AP_CV_HORIZON]
        model = ap_make_prophet(mode, cps, fourier, with_uncertainty=False)
        model.fit(history)
        predicted = model.predict(pd.DataFrame({"ds": future.index}))["yhat"].values
        scores.append(np.mean(np.abs(future["Passengers"].values - predicted) / future["Passengers"].values) * 100)
    return float(np.mean(scores))


AP_TUNING_CACHE = OUTPUT_DIR / "table_airpassengers_cv_tuning.csv"
AP_GRID = [(mode, cps, fourier) for mode in ("additive", "multiplicative")
           for cps in (0.01, 0.05, 0.1, 0.5) for fourier in (3, 6, 10)]
if AP_TUNING_CACHE.exists() and not FORCE_RETUNE:
    ap_tuning = pd.read_csv(AP_TUNING_CACHE)
    print(f"  Loaded cached tuning results ({len(ap_tuning)} configurations) from {AP_TUNING_CACHE.name}")
else:
    print(f"  Tuning {len(AP_GRID)} configurations x {len(ap_cutoffs)} origins ...")
    ap_tuning = pd.DataFrame([{"mode": mode, "cps": cps, "fourier": fourier,
                               "mean_cv_MAPE": ap_cv_mape(mode, cps, fourier)} for mode, cps, fourier in AP_GRID])
    ap_tuning.to_csv(AP_TUNING_CACHE, index=False)
ap_best = ap_tuning.sort_values("mean_cv_MAPE").iloc[0]
ap_best_config = (str(ap_best["mode"]), float(ap_best["cps"]), int(ap_best["fourier"]))
print(f"  Best configuration by mean cross-validated MAPE ({ap_best['mean_cv_MAPE']:.2f}%): "
      f"mode {ap_best_config[0]}, changepoint_prior_scale {ap_best_config[1]}, Fourier order {ap_best_config[2]}")
print("  Average cross-validated MAPE (%) by choice:")
for column in ("mode", "cps", "fourier"):
    print("   ", ap_tuning.groupby(column)["mean_cv_MAPE"].mean().round(2).to_dict())
default_cv = ap_tuning[(ap_tuning["cps"] == 0.05) & (ap_tuning["fourier"] == 10)].set_index("mode")["mean_cv_MAPE"]
print(f"  Cross-validated MAPE of the two default configurations: "
      f"additive {default_cv['additive']:.2f}%, multiplicative {default_cv['multiplicative']:.2f}%")
ap_tuning.sort_values("mean_cv_MAPE").round(3).to_csv(AP_TUNING_CACHE, index=False)

# --- 9.5 Fit the Prophet variants and forecast the test months -----------------------------------
_, ap_forecasts["P_add"] = ap_fit_forecast("additive", 0.05, 10)
_, ap_forecasts["P_mult"] = ap_fit_forecast("multiplicative", 0.05, 10)
_, ap_forecasts["P_bc"] = ap_fit_forecast("additive", 0.05, 10, use_boxcox=True)
ap_best_model, ap_forecasts["P_tuned"] = ap_fit_forecast(*ap_best_config)

# --- 9.6 Evaluation on the 29 test months ----------------------------------------------------------------
# Same metrics as the ARIMA project (MAE, RMSE, R-squared, MAPE), computed by the same
# function used for the electricity models. The "(GWh)" in its column names is renamed
# because these data are in thousands of passengers.
order = ["A_base", "A_opt", "P_add", "P_mult", "P_bc", "P_tuned"]
ap_metrics = pd.DataFrame({AP_LABELS[key]: forecast_metrics(ap_actual, ap_forecasts[key]["yhat"],
                                                            ap_forecasts[key]["lower"], ap_forecasts[key]["upper"])
                           for key in order}).T
ap_metrics = ap_metrics.rename(columns=lambda name: name.replace("GWh", "000 passengers"))
print_section("AIR PASSENGERS: ACCURACY ON THE 29-MONTH TEST SET (thousands of passengers)")
print(ap_metrics.round(2).to_string())
ap_metrics.round(3).to_csv(OUTPUT_DIR / "table_airpassengers_metrics.csv")

# Published ARIMA figures against the re-run, so any environment difference is visible
published_check = pd.DataFrame({
    "Published MAE": [ARIMA_PUBLISHED[k]["MAE"] for k in ("A_base", "A_opt")],
    "Re-run MAE": [ap_metrics.loc[AP_LABELS[k], "MAE (000 passengers)"] for k in ("A_base", "A_opt")],
    "Published MAPE (%)": [ARIMA_PUBLISHED[k]["MAPE"] for k in ("A_base", "A_opt")],
    "Re-run MAPE (%)": [ap_metrics.loc[AP_LABELS[k], "MAPE (%)"] for k in ("A_base", "A_opt")],
    "Published R2": [ARIMA_PUBLISHED[k]["R2"] for k in ("A_base", "A_opt")],
    "Re-run R2": [ap_metrics.loc[AP_LABELS[k], "R2"] for k in ("A_base", "A_opt")],
}, index=[AP_LABELS["A_base"], AP_LABELS["A_opt"]])
print("\n  ARIMA: figures published on the ARIMA page against the re-run in this script:")
print(published_check.round(3).to_string())
if abs(published_check["Published MAE"] - published_check["Re-run MAE"]).max() > 0.5:
    print("  The re-run differs from the published figures by more than 0.5 on MAE. ARIMA(12,2,12) does not")
    print("  fully converge, so its results can move slightly with the library version. Quote the published")
    print("  figures when citing the ARIMA page, and the re-run figures for the like-for-like comparison.")
else:
    print("  The re-run is within 0.5 of the published MAE. Small differences are still expected, because")
    print("  ARIMA(12,2,12) does not fully converge (see the optimiser flags above).")
published_check.round(3).to_csv(OUTPUT_DIR / "table_airpassengers_arima_published_vs_rerun.csv")

# --- 9.7 Are the differences real? A paired block bootstrap on 29 points -----------------------
# With 29 test months the answer will often be "not clearly". Months are resampled in blocks
# of three, because neighbouring forecast errors are correlated.
def ap_paired_bootstrap(actual, forecast_a, forecast_b, block=3, seed=RANDOM_SEED):
    """Mean of |error A| - |error B| with a 95% block-bootstrap interval (negative = A better)."""
    diff = np.abs(actual - np.asarray(forecast_a)) - np.abs(actual - np.asarray(forecast_b))
    n, n_blocks = len(diff), int(np.ceil(len(diff) / block))
    starts = np.random.default_rng(seed).integers(0, n - block + 1, size=(N_BOOTSTRAP, n_blocks))
    index = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(N_BOOTSTRAP, -1)[:, :n]
    resampled = diff[index].mean(axis=1)
    return diff.mean(), np.percentile(resampled, 2.5), np.percentile(resampled, 97.5), np.mean(resampled < 0) * 100


ap_comparisons = [
    ("Prophet additive vs ARIMA baseline (both untransformed)", "P_add", "A_base"),
    ("Prophet multiplicative vs Prophet additive", "P_mult", "P_add"),
    ("Prophet Box-Cox vs ARIMA optimal (same transformation)", "P_bc", "A_opt"),
    ("Prophet tuned vs ARIMA optimal", "P_tuned", "A_opt"),
    ("Prophet tuned vs Prophet multiplicative (defaults)", "P_tuned", "P_mult"),
    ("Prophet Box-Cox vs Prophet multiplicative", "P_bc", "P_mult"),
]
ap_pair_rows = []
for label, key_a, key_b in ap_comparisons:
    mean_diff, low, high, share = ap_paired_bootstrap(ap_actual, ap_forecasts[key_a]["yhat"], ap_forecasts[key_b]["yhat"])
    ap_pair_rows.append({"Comparison (A vs B)": label, "MAE(A) - MAE(B) (000 passengers)": mean_diff,
                         "95% CI low": low, "95% CI high": high, "Resamples where A better (%)": share})
ap_pairs = pd.DataFrame(ap_pair_rows)
print("\n  Paired block bootstrap on the 29 test months (negative = first model better):")
print(ap_pairs.round(2).to_string(index=False))
ap_pairs.round(3).to_csv(OUTPUT_DIR / "table_airpassengers_paired_comparisons.csv", index=False)

# --- 9.8 What did the tuned Prophet learn? -------------------------------------------------------------------
# Prophet's decomposition against the classical decomposition on the ARIMA page (July and
# August about 20% above trend, November and January about 10% below).
ap_all_dates = pd.DataFrame({"ds": ap_df.index})
np.random.seed(RANDOM_SEED)
ap_components = ap_best_model.predict(ap_all_dates)
ap_multiplicative = ap_best_config[0] == "multiplicative"
ap_scale, ap_unit = (100, "% of trend") if ap_multiplicative else (1, "thousand passengers")
ap_year = ap_components[(ap_components["ds"] >= "1958-01-01") & (ap_components["ds"] <= "1958-12-01")]
month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
seasonal_effect = pd.Series(ap_year["yearly"].values * ap_scale, index=month_names)
print(f"\n  Annual seasonal effect learned by the tuned Prophet ({ap_unit}):")
print("   ", ", ".join(f"{name} {value:+.1f}" for name, value in seasonal_effect.items()))
print("  (Compare with the multiplicative decomposition on the ARIMA page.)")

# --- Chart 19: the six forecasts -----------------------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(18, 9), sharex=True, sharey=True)
for axis, key in zip(axes.ravel(), order):
    forecast = ap_forecasts[key]
    axis.plot(ap_train.index, ap_train["Passengers"], color=COL_GREY, lw=1.2, label="Training data")
    axis.plot(ap_test.index, ap_actual, color="black", lw=2, label="Actual (test)")
    axis.fill_between(ap_test.index, forecast["lower"], forecast["upper"], color=AP_COLOURS[key], alpha=0.2,
                      label="95% interval")
    axis.plot(ap_test.index, forecast["yhat"], color=AP_COLOURS[key], lw=2.2, label="Forecast")
    mae = ap_metrics.loc[AP_LABELS[key], "MAE (000 passengers)"]
    mape = ap_metrics.loc[AP_LABELS[key], "MAPE (%)"]
    axis.set_title(f"{AP_LABELS[key]}\nMAE {mae:.1f}, MAPE {mape:.2f}%", fontsize=11)
    axis.tick_params(axis="x", rotation=30)
axes[0, 0].legend(loc="upper left", fontsize=9)
for axis in axes[:, 0]:
    axis.set_ylabel("Passengers (thousands)")
fig.suptitle("Air Passengers: the same 29 test months forecast by ARIMA and Prophet", fontsize=14)
fig.tight_layout()
save_figure(fig, "19_airpassengers_forecast.png")

# --- Chart 20: accuracy comparison ---------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 5.8))
labels = [AP_LABELS[key] for key in order]
for axis, column, title, reference in (
        (axes[0], "MAE (000 passengers)", "Test-set MAE (thousand passengers)", "MAE"),
        (axes[1], "MAPE (%)", "Test-set MAPE (%)", "MAPE")):
    values = ap_metrics.loc[labels, column].values
    bars = axis.barh(labels[::-1], values[::-1], color=[AP_COLOURS[key] for key in order][::-1])
    for bar, value in zip(bars, values[::-1]):
        axis.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2, f"{value:.2f}", va="center")
    axis.axvline(ARIMA_PUBLISHED["A_opt"][reference], color=COL_ACCENT, ls="--", lw=1.2,
                 label="ARIMA optimal, as published")
    axis.set_xlim(0, values.max() * 1.15)
    axis.set_title(title)
axes[0].legend(loc="lower right", fontsize=9)
fig.tight_layout()
save_figure(fig, "20_airpassengers_comparison.png")

# --- Chart 21: the tuned Prophet's components --------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
axes[0].plot(ap_df.index, ap_df["Passengers"], color=COL_GREY, lw=1, label="Actual")
axes[0].plot(ap_components["ds"], ap_components["trend"], color=COL_TRAIN, lw=2.2, label="Prophet trend")
axes[0].axvspan(ap_test.index.min(), ap_test.index.max(), color=COL_TEST, alpha=0.12, label="Test months")
axes[0].set_ylabel("Passengers (thousands)")
axes[0].set_title("Trend")
axes[0].legend(loc="upper left")
axes[1].bar(month_names, seasonal_effect.values, color=COL_TRAIN)
axes[1].axhline(0, color="black", lw=0.8)
axes[1].set_ylabel(f"Seasonal effect ({ap_unit})")
axes[1].set_title("Annual seasonality")
fig.tight_layout()
save_figure(fig, "21_airpassengers_components.png")

print_section("CHECKPOINT D COMPLETE")
print(f"  Charts 18-21 and the tables are in: {OUTPUT_DIR}")
