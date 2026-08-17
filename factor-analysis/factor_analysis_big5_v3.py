"""
================================================================================
FACTOR ANALYSIS - VALIDATING A 50-ITEM PERSONALITY ASSESSMENT (BIG FIVE / OCEAN)
================================================================================

Business scenario:
    An HR-tech company has licensed a 50-item personality assessment, built on
    the International Personality Item Pool (IPIP-50), as part of its
    recruitment screening product. The assessment claims to measure five
    distinct personality constructs: Openness, Conscientiousness, Extraversion,
    Agreeableness, and Neuroticism (OCEAN). Before deploying it at scale, the
    business needs to know whether the instrument actually measures five
    distinct, coherent constructs - or whether items overlap and blur across
    traits in a way that would make the resulting scores unreliable for hiring
    decisions.

Why Factor Analysis (and not PCA):
    This question is explicitly about whether real, underlying constructs
    exist and are being cleanly captured by the observed survey items - not
    about compressing variance for convenience. Factor Analysis (FA) is a
    latent-variable model: it treats the five personality traits as real,
    unobserved dispositions that cause a person's answers to the 50 items,
    with each item response also containing item-specific "noise" that is not
    shared with the underlying trait. This is fundamentally different to the
    Principal Component Analysis (PCA) project elsewhere in this portfolio,
    where the principal components were abstract mathematical directions of
    maximum variance in the 30 features, with no obligation - or expectation
    - to correspond to anything real. In PCA, the components are a
    by-product of compressing the data. In Factor Analysis, the factors are
    the entire point of the analysis: the question is whether they exist.

Author: Marc Grover
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

# ------------------------------------------------------------------------------
# Compatibility shim: factor_analyzer 0.5.1 calls sklearn's check_array() with
# the keyword argument `force_all_finite`. Recent scikit-learn versions
# (>= 1.6) renamed this to `ensure_all_finite` and removed the old name,
# which raises a TypeError when factor_analyzer is imported/used against a
# newer scikit-learn install. Rather than pinning an older scikit-learn
# version (which could conflict with other packages in the environment),
# this shim detects the mismatch and transparently maps the old keyword to
# the new one, so the script runs unmodified regardless of which
# scikit-learn version is installed.
# ------------------------------------------------------------------------------
import inspect
from sklearn.utils.validation import check_array as _sklearn_check_array

_check_array_params = inspect.signature(_sklearn_check_array).parameters
if "force_all_finite" not in _check_array_params and "ensure_all_finite" in _check_array_params:
    def _check_array_compat(*args, **kwargs):
        if "force_all_finite" in kwargs:
            kwargs["ensure_all_finite"] = kwargs.pop("force_all_finite")
        return _sklearn_check_array(*args, **kwargs)

    import sklearn.utils.validation
    sklearn.utils.validation.check_array = _check_array_compat
    # factor_analyzer imports check_array directly into its own namespace, so
    # the patch must also be applied there, not just on the sklearn module
    import factor_analyzer.factor_analyzer as _fa_internal
    _fa_internal.check_array = _check_array_compat

from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo

# ------------------------------------------------------------------------------
# Global plotting configuration, consistent with portfolio conventions
# ------------------------------------------------------------------------------

sns.set_theme(style="whitegrid")
PLOT_DIR = "plots/"
FIG_DPI = 150


# ==============================================================================
# SECTION 1: DATA LOADING AND VALIDATION
# ==============================================================================
# The raw response data is sourced from the Open-Source Psychometrics Project
# (openpsychometrics.org), which hosts anonymised responses to the IPIP-50
# personality inventory. Each of the 50 items is rated on a five-point scale
# (1 = Disagree, 3 = Neutral, 5 = Agree), with 10 items per trait. A response
# of 0 indicates the item was skipped, per the dataset's codebook.

raw = pd.read_csv("big5_raw.csv", sep="\t")

# Identify the 50 item columns programmatically (e.g. E1-E10, N1-N10, etc.)
# rather than hard-coding the list, so the script is robust to column reordering.
item_cols = [c for c in raw.columns if c[0] in "ENACO" and c[1:].isdigit()]

trait_names = {
    "E": "Extraversion",
    "N": "Neuroticism",
    "A": "Agreeableness",
    "C": "Conscientiousness",
    "O": "Openness",
}

print("=" * 80)
print("SECTION 1: DATA LOADING AND VALIDATION")
print("=" * 80)
print(f"Raw dataset shape: {raw.shape}")
print(f"Number of personality items identified: {len(item_cols)}")

df = raw.copy()

# Treat 0-coded ("skipped") responses as missing, per the dataset codebook
df[item_cols] = df[item_cols].replace(0, np.nan)

n_incomplete = df[item_cols].isna().any(axis=1).sum()
print(f"Respondents with at least one skipped item: {n_incomplete} of {len(df)}")

# Given the very large sample size and the negligible proportion of incomplete
# responses, listwise deletion (dropping incomplete rows) is the appropriate
# and simplest missing-data strategy here - imputation would add complexity
# for no meaningful benefit at this scale.
df = df.dropna(subset=item_cols).reset_index(drop=True)
print(f"Shape after removing incomplete responses: {df.shape}")

# Validate that all remaining item values fall within the expected 1-5 range
assert df[item_cols].min().min() >= 1 and df[item_cols].max().max() <= 5, \
    "Item values fall outside the expected 1-5 range"
print("Validation passed: all item values fall within the expected 1-5 range.")

# The IPIP-50 scoring key specifies that a subset of items are negatively
# keyed - i.e. phrased in the opposite direction to the trait they measure
# (e.g. "I don't talk a lot." for Extraversion). These items must be reverse
# scored (6 - x on a 1-5 scale) so that, for every item, a higher value
# consistently indicates a higher standing on the underlying trait. Without
# this step, negatively-keyed items would load in the opposite direction to
# their trait and distort the correlation structure the analysis depends on.
reverse_keyed_items = {
    "E2", "E4", "E6", "E8", "E10",
    "N2", "N4",
    "A1", "A3", "A5", "A7",
    "C2", "C4", "C6", "C8",
    "O2", "O4", "O6",
}
for col in reverse_keyed_items:
    df[col] = 6 - df[col]
print(f"Reverse-scored {len(reverse_keyed_items)} negatively-keyed items per the IPIP-50 scoring key.")

X = df[item_cols]


# ==============================================================================
# SECTION 2: SUITABILITY DIAGNOSTICS
# ==============================================================================
# Before extracting any factors, it is necessary to confirm that the
# correlation structure of the data is actually suitable for Factor Analysis.
# If the items were largely uncorrelated, there would be no shared variance
# for any latent factor to explain. Two standard diagnostic tests are used:
#
#   - Bartlett's Test of Sphericity tests the null hypothesis that the
#     correlation matrix is an identity matrix (i.e. no correlation between
#     any items). Rejecting this null hypothesis is a precondition for FA.
#
#   - The Kaiser-Meyer-Olkin (KMO) measure of sampling adequacy quantifies
#     the proportion of variance among the items that might be common
#     variance, on a 0-1 scale. Values above 0.8 are considered excellent by
#     conventional benchmarks; values below 0.5 indicate FA is not suitable.

print("\n" + "=" * 80)
print("SECTION 2: SUITABILITY DIAGNOSTICS")
print("=" * 80)

chi_square_value, p_value = calculate_bartlett_sphericity(X)
print(f"Bartlett's Test of Sphericity: chi-squared = {chi_square_value:.1f}, p = {p_value:.4g}")

kmo_per_item, kmo_model = calculate_kmo(X)
print(f"Overall KMO measure of sampling adequacy: {kmo_model:.4f}")
print(f"KMO per item - minimum: {kmo_per_item.min():.3f}, maximum: {kmo_per_item.max():.3f}")

kmo_series = pd.Series(kmo_per_item, index=item_cols).sort_values()
print("Lowest-KMO items (weakest individual contribution to sampling adequacy):")
print(kmo_series.head(3).round(3))


# ==============================================================================
# SECTION 3: EXPLORATORY DATA ANALYSIS
# ==============================================================================
# A correlation heatmap of all 50 items visualises the block structure that
# motivates the entire analysis: if the instrument is working as intended,
# items belonging to the same trait should correlate more strongly with each
# other than with items from other traits, forming five visible blocks along
# the diagonal.

print("\n" + "=" * 80)
print("SECTION 3: EXPLORATORY DATA ANALYSIS")
print("=" * 80)

corr_matrix = X.corr()

fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(corr_matrix, cmap="RdBu_r", center=0, vmin=-0.6, vmax=0.6,
            square=True, linewidths=0.2, cbar_kws={"shrink": 0.7}, ax=ax)
ax.set_title("Correlation Matrix of the 50 IPIP Personality Items\n"
              "(items grouped by theorised trait: E, N, A, C, O)", fontsize=13)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}plot_01_correlation_heatmap.png", dpi=FIG_DPI)
plt.close()
print("Saved: plot_01_correlation_heatmap.png")

# Mean response and standard deviation for each trait's item block, to check
# for any obviously degenerate items (e.g. near-zero variance)
item_stats = X.agg(["mean", "std"]).T
item_stats["trait"] = [trait_names[c[0]] for c in item_stats.index]
print("\nItem-level response statistics by trait (mean +/- 1 SD):")
print(item_stats.groupby("trait")[["mean", "std"]].mean().round(3))


# ==============================================================================
# SECTION 4: FACTOR EXTRACTION AND RETENTION
# ==============================================================================
# Two questions are addressed here: (1) how many factors does the data itself
# suggest, using standard retention heuristics, and (2) how does that compare
# to the five factors the instrument is theorised to measure? Rather than
# simply asserting five factors upfront, both the empirical and theorised
# answers are examined and compared - a discrepancy is as analytically
# informative as agreement.

print("\n" + "=" * 80)
print("SECTION 4: FACTOR EXTRACTION AND RETENTION")
print("=" * 80)

# Unrotated extraction across all possible factors, purely to obtain
# eigenvalues for the scree plot and Kaiser criterion
fa_unrotated = FactorAnalyzer(n_factors=X.shape[1], rotation=None, method="principal")
fa_unrotated.fit(X)
eigenvalues, _ = fa_unrotated.get_eigenvalues()

n_kaiser = int((eigenvalues > 1).sum())
print(f"Factors with eigenvalue > 1 (Kaiser criterion): {n_kaiser}")


def parallel_analysis(n_obs, n_vars, n_iter=20, percentile=95, random_state=42):
    """
    Horn's (1965) Parallel Analysis. Simulates random, uncorrelated data of
    the same dimensions as the real dataset many times, extracts eigenvalues
    from each simulated dataset, and returns the percentile threshold at each
    factor position. A real eigenvalue is only meaningful if it exceeds what
    random noise alone would be expected to produce - this corrects for the
    Kaiser criterion's well-documented tendency to over-extract factors,
    which is a particularly acute risk with ordinal Likert-scale data.
    """
    rng = np.random.default_rng(random_state)
    simulated_eigenvalues = np.zeros((n_iter, n_vars))
    for i in range(n_iter):
        simulated_data = rng.normal(size=(n_obs, n_vars))
        fa_sim = FactorAnalyzer(n_factors=n_vars, rotation=None, method="principal")
        fa_sim.fit(simulated_data)
        sim_ev, _ = fa_sim.get_eigenvalues()
        simulated_eigenvalues[i, :] = sim_ev
    return np.percentile(simulated_eigenvalues, percentile, axis=0)


# n_iter is kept modest (20) rather than the sometimes-recommended 100+,
# reflecting the CPU-only development environment used for this project. At
# this sample size (n ~ 19,700) the simulated eigenvalue thresholds are
# already highly stable across runs, so this is a pragmatic trade-off rather
# than a precision compromise.
PARALLEL_ANALYSIS_ITERATIONS = 20
parallel_threshold = parallel_analysis(X.shape[0], X.shape[1], n_iter=PARALLEL_ANALYSIS_ITERATIONS)
n_parallel = int((eigenvalues > parallel_threshold).sum())
print(f"Factors retained by Parallel Analysis (95th percentile, "
      f"{PARALLEL_ANALYSIS_ITERATIONS} simulated datasets): {n_parallel}")
print(f"Theorised number of factors (Big Five / OCEAN model): 5")

# Scree plot: real eigenvalues against the parallel analysis threshold
fig, ax = plt.subplots(figsize=(10, 6))
x_range = np.arange(1, 16)
ax.plot(x_range, eigenvalues[:15], marker="o", label="Observed eigenvalues", color="#157878")
ax.plot(x_range, parallel_threshold[:15], marker="s", linestyle="--",
        label="Parallel Analysis threshold (95th percentile of random data)", color="#c0392b")
ax.axhline(1, color="grey", linestyle=":", label="Kaiser criterion (eigenvalue = 1)")
ax.axvline(5, color="#2980b9", linestyle=":", alpha=0.7, label="Theorised solution (5 factors)")
ax.set_xlabel("Factor number")
ax.set_ylabel("Eigenvalue")
ax.set_title("Scree Plot: Observed Eigenvalues vs Parallel Analysis Threshold")
ax.legend()
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}plot_02_scree_parallel_analysis.png", dpi=FIG_DPI)
plt.close()
print("Saved: plot_02_scree_parallel_analysis.png")

# Compare variance explained under Principal Axis Factoring (PAF, the primary
# extraction method for this project) against Maximum Likelihood (ML), both
# fixed at the theorised 5-factor solution, to sense-check the choice of
# extraction method before proceeding.
fa_paf_check = FactorAnalyzer(n_factors=5, rotation=None, method="principal")
fa_paf_check.fit(X)
paf_variance = fa_paf_check.get_factor_variance()

fa_ml_check = FactorAnalyzer(n_factors=5, rotation=None, method="ml")
fa_ml_check.fit(X)
ml_variance = fa_ml_check.get_factor_variance()

extraction_comparison = pd.DataFrame({
    "PAF cumulative variance": np.round(paf_variance[2], 4),
    "ML cumulative variance": np.round(ml_variance[2], 4),
}, index=[f"Factor {i+1}" for i in range(5)])
print("\nExtraction method comparison at the 5-factor solution (unrotated):")
print(extraction_comparison)


# ==============================================================================
# SECTION 5: ROTATION AND LOADINGS INTERPRETATION
# ==============================================================================
# The theorised 5-factor solution is extracted using Principal Axis Factoring
# and rotated using Promax, an oblique rotation. Oblique rotation is the
# methodologically appropriate choice here because the five personality
# traits are theoretically permitted to correlate with one another (for
# example, Conscientiousness and Neuroticism are well-documented in the
# psychological literature as negatively correlated). This is a direct
# contrast with the PCA project, where an orthogonal rotation would typically
# be preferred - or rotation dispensed with entirely - because the goal there
# is a set of mathematically uncorrelated components, not a model of
# correlated real-world constructs.

print("\n" + "=" * 80)
print("SECTION 5: ROTATION AND LOADINGS INTERPRETATION (PAF, Promax)")
print("=" * 80)

fa = FactorAnalyzer(n_factors=5, rotation="promax", method="principal")
fa.fit(X)

factor_labels = [f"Factor{i+1}" for i in range(5)]
loadings = pd.DataFrame(fa.loadings_, index=item_cols, columns=factor_labels)
loadings["trait"] = [c[0] for c in loadings.index]

# Assign each item to the factor on which it loads most strongly in absolute
# terms, then cross-tabulate against the item's theorised trait to check
# whether the empirical factor structure recovers the theorised one.
loadings["dominant_factor"] = loadings[factor_labels].abs().idxmax(axis=1)

alignment_table = pd.crosstab(loadings["trait"], loadings["dominant_factor"])
print("Theorised trait vs empirically-dominant rotated factor (item counts):")
print(alignment_table)

# For each theorised trait, its "home" factor is the one most of its items
# dominantly load on; any item loading most strongly on a different factor
# is misaligned. This gives a single, interpretable count of how many of the
# 50 items (out of 50) failed to land on their expected factor.
n_misaligned = int(len(loadings) - alignment_table.values.max(axis=1).sum())
print(f"Items misaligned with their theorised trait's dominant factor: {n_misaligned} of {len(loadings)}")

# Cross-loadings: items with a loading above the conventional 0.32 threshold
# (Tabachnick & Fidell) on more than one factor indicate the item does not
# discriminate cleanly between constructs - a specific, actionable quality
# flag for the assessment's item bank.
CROSS_LOAD_THRESHOLD = 0.32
cross_loaded_items = []
for item in loadings.index:
    row = loadings.loc[item, factor_labels]
    n_strong = (row.abs() > CROSS_LOAD_THRESHOLD).sum()
    if n_strong > 1:
        cross_loaded_items.append(item)
print(f"\nItems cross-loading (> {CROSS_LOAD_THRESHOLD}) on more than one factor: {len(cross_loaded_items)}")
if cross_loaded_items:
    print(cross_loaded_items)

# Heatmap of the rotated loading matrix, ordered by theorised trait, to
# visualise how cleanly (or not) the empirical structure recovers the five
# theorised blocks
loadings_sorted = loadings.sort_values("trait")
fig, ax = plt.subplots(figsize=(8, 12))
sns.heatmap(loadings_sorted[factor_labels].astype(float), cmap="RdBu_r", center=0,
            vmin=-0.85, vmax=0.85, annot=False, linewidths=0.3,
            yticklabels=[f"{i} ({t})" for i, t in zip(loadings_sorted.index, loadings_sorted["trait"])],
            cbar_kws={"label": "Rotated loading"}, ax=ax)
ax.set_title("Promax-Rotated Factor Loadings\n(items ordered by theorised trait)", fontsize=12)
ax.set_xlabel("Rotated factor")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}plot_03_rotated_loadings_heatmap.png", dpi=FIG_DPI)
plt.close()
print("Saved: plot_03_rotated_loadings_heatmap.png")


# ==============================================================================
# SECTION 6: COMMUNALITIES AND MODEL FIT
# ==============================================================================
# Communality is the proportion of an item's variance that is explained by
# the common factors, as distinct from variance unique to that item
# (specific variance plus measurement error). This diagnostic has no
# equivalent in the PCA project: PCA components are constructed precisely so
# that, taken together, they explain 100% of every variable's variance -
# there is no "unexplained, item-specific" remainder by construction. In
# Factor Analysis, low communality is a meaningful and expected finding: it
# identifies items that are not well explained by the underlying constructs
# the instrument claims to measure.

print("\n" + "=" * 80)
print("SECTION 6: COMMUNALITIES AND MODEL FIT")
print("=" * 80)

communalities = pd.Series(fa.get_communalities(), index=item_cols, name="communality")
comm_df = pd.DataFrame({"communality": communalities, "trait": [c[0] for c in item_cols]})

print("Communalities summary across all 50 items:")
print(communalities.describe().round(3))
print("\nMean communality by trait:")
print(comm_df.groupby("trait")["communality"].mean().round(3))
print("\nFive lowest-communality items:")
print(communalities.sort_values().head())

fig, ax = plt.subplots(figsize=(9, 12))
comm_sorted = comm_df.sort_values(["trait", "communality"])
colors = comm_sorted["trait"].map({"E": "#157878", "N": "#c0392b", "A": "#8e44ad",
                                     "C": "#d68910", "O": "#2980b9"})
trait_colors = {"E": "#157878", "N": "#c0392b", "A": "#8e44ad", "C": "#d68910", "O": "#2980b9"}
legend_handles = [mpatches.Patch(color=trait_colors[t], label=trait_names[t]) for t in "ENACO"]
ax.barh(range(len(comm_sorted)), comm_sorted["communality"], color=colors)
ax.set_yticks(range(len(comm_sorted)))
ax.set_yticklabels(comm_sorted.index)
ax.set_xlabel("Communality")
ax.legend(handles=legend_handles, title="Theorised trait", loc="lower right", frameon=True)
ax.set_title("Item Communalities (variance explained by the 5 common factors)\n"
              "coloured by theorised trait", fontsize=12)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}plot_04_communalities.png", dpi=FIG_DPI)
plt.close()
print("Saved: plot_04_communalities.png")

# Factor correlation matrix - only produced because an oblique rotation was
# used. This is itself a point of contrast with PCA: an orthogonal PCA
# solution has, by construction, a component correlation matrix that is the
# identity matrix and therefore uninformative.
factor_corr = pd.DataFrame(fa.phi_, columns=factor_labels, index=factor_labels)
print("\nFactor correlation matrix (Promax oblique rotation):")
print(factor_corr.round(3))

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(factor_corr, cmap="RdBu_r", center=0, vmin=-0.4, vmax=0.4,
            annot=True, fmt=".2f", square=True, ax=ax)
ax.set_title("Inter-Factor Correlation Matrix (Promax)", fontsize=12)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}plot_05_factor_correlation_matrix.png", dpi=FIG_DPI)
plt.close()
print("Saved: plot_05_factor_correlation_matrix.png")


# ==============================================================================
# SECTION 7: BUSINESS INSIGHT AND FACTOR SCORING
# ==============================================================================
# The rotated factors are mapped back to their theorised trait names using
# the dominant-loading assignment established in Section 5, and factor
# scores are computed for every respondent - the deliverable a recruitment
# product would actually consume downstream, in place of the raw 50 item
# responses.

print("\n" + "=" * 80)
print("SECTION 7: BUSINESS INSIGHT AND FACTOR SCORING")
print("=" * 80)

# Establish the Factor -> Trait name mapping from the dominant-loading table
factor_to_trait = (
    loadings.groupby("dominant_factor")["trait"]
    .agg(lambda s: s.value_counts().idxmax())
    .to_dict()
)
factor_to_trait_name = {f: trait_names[t] for f, t in factor_to_trait.items()}
print("Empirical factor -> theorised trait mapping (by dominant item loading):")
for f in factor_labels:
    print(f"  {f} -> {factor_to_trait_name.get(f, 'unmapped')}")

# Validate that the mapping is one-to-one (i.e. no two factors map to the
# same trait, which would indicate the five-factor structure did not recover
# cleanly)
is_one_to_one = len(set(factor_to_trait_name.values())) == 5
print(f"\nMapping is one-to-one across all five traits: {is_one_to_one}")

factor_scores = fa.transform(X)
factor_scores_df = pd.DataFrame(
    factor_scores,
    columns=[factor_to_trait_name.get(f, f) for f in factor_labels],
)
print("\nFactor score summary statistics (standardised, mean ~0, SD ~1):")
print(factor_scores_df.describe().round(3))

# Save the item-to-trait alignment table and factor scores as artefacts a
# downstream recruitment scoring pipeline could consume directly
loadings.drop(columns=["dominant_factor"]).to_csv(f"{PLOT_DIR}../item_loadings.csv")
factor_scores_df.head(1000).to_csv(f"{PLOT_DIR}../factor_scores_sample.csv", index=False)

print("\nAnalysis complete.")