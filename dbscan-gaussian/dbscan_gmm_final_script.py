# ============================================================================
# DBSCAN AND GAUSSIAN MIXTURE MODELS: HOTSPOT DETECTION IN ROAD COLLISION DATA
# ============================================================================
# Business scenario
# -----------------
# A county highway authority has a fixed annual budget for road-safety engineering
# and must decide which locations to investigate first. Collision locations are
# geographic points with very uneven density (dense towns, sparse rural roads, long
# linear corridors), which is exactly where K-Means' assumptions (compact, spherical
# clusters, every point assigned to a cluster, K chosen in advance) are weakest.
#
# Analytical question
# -------------------
# Which clustering method, fitted to one year of injury collisions, identifies hotspots
# that still concentrate collisions the following year, for a given amount of ground
# to investigate? K-Means, Gaussian Mixture Models (GMM) and DBSCAN are compared.
#
# Structure (the script runs top to bottom)
# -----------------------------------------
#   Part A  Synthetic benchmark: where each method's assumptions hold and fail
#           (ground truth known), then how their parameters are chosen without labels
#   Part B  Wheat Seeds bridge: GMM and DBSCAN on the K-Means project's own dataset
#   Part C  Kent road collisions: data audit and fitting the methods to the fit year
#   Part D  Out-of-time validation on the test year, plus a replication on an
#           earlier year pair to check the result is not specific to one pair
#
# Data
# ----
#   Synthetic data   generated in the script (scikit-learn)
#   Wheat Seeds      UCI Seeds dataset, 210 kernels, 7 measurements, 3 varieties
#                    https://archive.ics.uci.edu/ml/datasets/seeds
#                    (expects 'Seed_Data.csv', as in the K-Means Clustering project)
#   Road collisions  DfT road safety data, STATS19 collisions (Open Government Licence v3.0)
#                    https://www.data.gov.uk/dataset/cb7ae6f0-4be6-4935-9277-47e5ce24a11f/road-accidents-safety-data
#                    The yearly files are downloaded automatically into ./data
#
# Charts (saved in the working directory, numbered in the order produced)
# -----------------------------------------------------------------------
#   01-03  synthetic benchmark                 04-07  label-free parameter selection
#   08-11  Wheat Seeds bridge                  12-18  Kent fit-year hotspots
#   19-23  Kent out-of-time validation
#
# Runtime: several minutes on a CPU-only laptop (the Gaussian mixtures with 200
# components dominate). Everything is seeded, so results reproduce exactly.
# ============================================================================

import os
import urllib.request
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import make_blobs, make_moons
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import adjusted_rand_score, silhouette_score, v_measure_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# GMM convergence is checked explicitly through .converged_ where it matters, so the
# library's per-fit warnings would only add noise to the console
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Dataframe presentation configuration
pd.set_option("display.width", 320)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_rows", 80)
pd.set_option("display.precision", 2)
sns.set_style("whitegrid")

# ============================================================================
# 0. CONFIGURATION (every tunable setting is here, with the reason for its value)
# ============================================================================
SHOW_CHARTS = True            # False saves the PNGs without opening a window for each one
RANDOM_STATE = 42             # one seed everywhere, so every run reproduces exactly
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# --- Part A: synthetic benchmark and parameter selection ---------------------
N_SAMPLES = 1500              # large enough for clear shapes, small enough to run in seconds
SYN_DBSCAN_MIN_SAMPLES = 10   # 10 rather than the default 5, so sparse noise is not mistaken for a cluster
KMEANS_K_RANGE = range(2, 9)  # silhouette is undefined for K = 1
GMM_K_RANGE = range(1, 9)     # K = 1 is allowed: BIC can legitimately prefer a single component
COVARIANCE_TYPES = ["full", "tied", "diag", "spherical"]
EPS_FACTORS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]   # multiples of the knee eps in the sensitivity grids
SYN_MIN_SAMPLES_GRID = [5, 10, 15, 20]

# --- Part B: Wheat Seeds ------------------------------------------------------
SEED_FILE = "Seed_Data.csv"   # same file as the K-Means Clustering project
WHEAT_TRUE_K = 3              # three wheat varieties
# Figures from the K-Means Clustering page, used as a reproduction check
KMEANS_PAGE_ARI, KMEANS_PAGE_SILHOUETTE, KMEANS_PAGE_AGREEMENT = 0.773, 0.40, 193 / 210

# --- Part C: Kent collisions --------------------------------------------------
FORCE_CODE = 46               # Kent in the DfT police force lookup (chosen in Section 5)
FIT_YEAR, TEST_YEAR = 2023, 2024          # primary year pair
REPLICATION_PAIRS = [(2022, 2023)]        # earlier pairs that repeat the whole procedure
BASE_URL = "https://data.dft.gov.uk/road-accidents-safety-data/"
FILE_TEMPLATE = "dft-road-casualty-statistics-collision-{year}.csv"
EASTING_RANGE = (0, 700_000)  # valid Ordnance Survey National Grid limits, in metres
NORTHING_RANGE = (0, 1_300_000)
EPS_GRID_M = [100, 150, 200, 250, 300, 400, 500, 750, 1000]   # candidate DBSCAN eps values (metres)
MIN_SAMPLES_GRID = [5, 8, 10, 15, 20]
REFERENCE_MIN_SAMPLES = 10    # used only for the k-distance reference curve
MIN_COLLISIONS_PER_YEAR = 3000    # study-area screen: DBSCAN needs enough points per year for hotspots to mean anything
MAX_MISSING_COORD_PCT = 5.0       # study-area screen: a high share of missing coordinates would undermine the spatial analysis
# Size guard: no hotspot may have a radius (centroid to furthest member) above this. A share-based
# guard was tried first, but a 243-collision, 2.7 km conurbation passed it; a physical size is the
# better test because the decision-maker investigates sites and road sections, not whole towns
MAX_HOTSPOT_RADIUS_M = 750
SITE_OVERRIDE = None          # set (eps_m, min_samples) to replace the automatic Site choice (primary pair only)
STRICT_TIER = (200, 10)       # fewer, larger hotspots: a conservative reading of 'actionable site'
AREA_TIER = (500, 10)         # area-scale reading, kept as a sensitivity (hotspots are town-sized)
K_LARGE = 200                 # second K-Means/GMM size, to test whether a finer partition helps
GMM_BIC_K_GRID = [2, 5, 10, 15, 20, 30, 40, 60, 80, 120, 200]
# GMM covariance floor in km^2: 1e-4 km^2 is a 10 m standard deviation, close to the practical
# precision of recorded collision locations. It stops components collapsing onto a few near-identical
# points, which would give infinite density and meaningless BIC values
GMM_REG_COVAR = 1e-4

# --- Part D: out-of-time validation -------------------------------------------
R_FLAG_M = 150                # a collision within this distance of a flagged one counts as 'at the same place'
                              # (the Site-tier eps: roughly a junction approach or a short road section)
CELL_M = 25                   # raster resolution for measuring areas (a 150 m disc is ~110 cells)
GRID_M = 500                  # side of the practitioner-benchmark grid squares
EXTRA_BUDGET_SHARES = [0.05, 0.10]   # area budgets besides the Site zone, as shares of the affected area
N_BOOT = 1000                 # bootstrap resamples
EPS_STABILITY_M = [100, 150, 200, 250, 300, 400, 500]   # eps values in the stability chart
STABILITY_MIN_SAMPLES = [5, 10]
PLOT_MAX_AREA_PCT = 25        # capture curves are drawn up to this share of the affected area


# ============================================================================
# 1. SHARED HELPERS
# ============================================================================
def finish_chart(filename):
    """Save the current figure as a numbered PNG, optionally show it, then free the memory."""
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    if SHOW_CHARTS:
        plt.show()
    plt.close("all")


def k_distance_curve(X, min_samples):
    """Sorted distance of every point to its min_samples-th nearest neighbour, plus the knee index.

    The knee is the point furthest below the straight line joining the curve's ends, after
    rescaling both axes to 0-1. It marks where the curve bends upward, i.e. where points
    inside clusters (small distances) give way to isolated points (large distances).
    """
    # kneighbors on the training data counts each point as its own first neighbour,
    # which matches DBSCAN's definition (min_samples includes the point itself)
    distances, _ = NearestNeighbors(n_neighbors=min_samples).fit(X).kneighbors(X)
    k_dist = np.sort(distances[:, -1])
    x_norm = np.linspace(0, 1, len(k_dist))
    y_norm = (k_dist - k_dist.min()) / (k_dist.max() - k_dist.min())
    return k_dist, int(np.argmax(x_norm - y_norm))


def matched_agreement(y_true, labels):
    """Hungarian-matched agreement between true classes and cluster labels.

    Cluster IDs are arbitrary, so each class is matched one-to-one to its best-fitting cluster
    before counting agreement. DBSCAN's noise label (-1) can never be matched, so noise points
    stay in the denominator and count as misassigned. Returns (agreement, aligned_table), where
    the table has matched clusters in the diagonal positions (then unmatched clusters, then noise).
    """
    table = pd.crosstab(y_true, labels)
    matchable = table.drop(columns=-1, errors="ignore")
    if matchable.shape[1] == 0:                     # everything was labelled noise
        return 0.0, table
    row_ind, col_ind = linear_sum_assignment(-matchable.values)   # negate to maximise agreement
    n_correct = matchable.values[row_ind, col_ind].sum()
    unmatched = [i for i in range(matchable.shape[1]) if i not in col_ind]
    aligned = matchable.iloc[:, list(col_ind) + unmatched]
    if -1 in table.columns:
        aligned = aligned.assign(Noise=table[-1])
    return n_correct / len(y_true), aligned


def score_labels(X, y_true, labels, with_agreement=False):
    """ARI, V-measure and silhouette (noise excluded), plus clusters found and noise share.

    DBSCAN's noise label (-1) is treated as one more group in the label-based scores, so a method
    that dumps genuine clusters into 'noise' is penalised, and one that finds true noise is rewarded.
    Silhouette excludes noise (it is undefined there) and needs at least two clusters.
    """
    mask = labels != -1
    n_clusters = len(np.unique(labels[mask]))
    scores = {
        "clusters": n_clusters, "noise_pct": 100 * (~mask).mean(),
        "ARI": adjusted_rand_score(y_true, labels), "V-measure": v_measure_score(y_true, labels),
        "silhouette": silhouette_score(X[mask], labels[mask]) if n_clusters >= 2 else np.nan,
    }
    aligned = None
    if with_agreement:
        agreement, aligned = matched_agreement(y_true, labels)
        scores["matched_agreement_pct"] = 100 * agreement
    return scores, aligned


def scatter_labels(ax, X, labels, title, size=8):
    """2D scatter coloured by label; noise (-1) as small grey crosses."""
    palette = sns.color_palette("tab10")
    for label in np.unique(labels):
        mask = labels == label
        if label == -1:
            ax.scatter(X[mask, 0], X[mask, 1], s=10, c="grey", marker="x", linewidths=0.6)
        else:
            ax.scatter(X[mask, 0], X[mask, 1], s=size, color=palette[label % 10])
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


# ============================================================================
# 2. PART A (1/2): SYNTHETIC BENCHMARK
# ============================================================================
# Four small 2D datasets with known structure, each breaking a different assumption. K-Means and
# GMM are GIVEN the correct number of clusters, which is deliberately generous to them: any failure
# is then down to the method's assumptions (shape, density, noise), not a poor choice of K
def make_datasets():
    """Return {name: (X, true_labels, true_k)}. True noise points carry the label -1."""
    datasets = {}

    # Moons: two interleaved crescents. Non-convex shapes break K-Means (compact, spherical
    # clusters) and GMM (elliptical clusters)
    X, y = make_moons(n_samples=N_SAMPLES, noise=0.08, random_state=RANDOM_STATE)
    datasets["Moons"] = (X, y, 2)

    # Anisotropic blobs: three Gaussian blobs stretched by one linear transform. Still elliptical,
    # so GMM's full covariance can model them, whereas K-Means (equal spherical variance) cannot
    X, y = make_blobs(n_samples=N_SAMPLES, centers=3, cluster_std=1.0, random_state=170)
    datasets["Anisotropic blobs"] = (X @ np.array([[0.6, -0.64], [-0.4, 0.8]]), y, 3)

    # Varied-density blobs: very different spreads. A single DBSCAN eps cannot suit both the tight
    # and the diffuse cluster, the textbook weakness of DBSCAN
    X, y = make_blobs(n_samples=N_SAMPLES, centers=3, cluster_std=[1.0, 2.5, 0.5], random_state=170)
    datasets["Varied-density blobs"] = (X, y, 3)

    # Blobs with background noise: three compact blobs plus 10% uniform noise, mirroring isolated
    # collisions on quiet roads. K-Means and GMM must place every point in a cluster; DBSCAN need not
    rng = np.random.RandomState(RANDOM_STATE)
    n_noise = N_SAMPLES // 10
    X_core, y_core = make_blobs(n_samples=N_SAMPLES - n_noise, centers=3, cluster_std=0.6, random_state=7)
    X_noise = rng.uniform(X_core.min(axis=0) - 2, X_core.max(axis=0) + 2, size=(n_noise, 2))
    datasets["Blobs with noise"] = (np.vstack([X_core, X_noise]),
                                    np.concatenate([y_core, np.full(n_noise, -1)]), 3)
    return datasets


print("PART A: Synthetic benchmark")
syn_datasets = make_datasets()
syn_names = list(syn_datasets.keys())
# Standardise so eps is on a comparable scale across datasets and the anisotropic stretch does not
# dominate distance calculations
syn_scaled = {name: StandardScaler().fit_transform(X_raw) for name, (X_raw, _, _) in syn_datasets.items()}

syn_rows, syn_labels, syn_knee = [], {}, {}
for name in syn_names:
    X, (_, y_true, true_k) = syn_scaled[name], syn_datasets[name]
    km_labels = KMeans(n_clusters=true_k, n_init=10, random_state=RANDOM_STATE).fit_predict(X)
    # 'full' covariance lets every component have its own elliptical shape and orientation, which
    # is the whole advantage of GMM over K-Means
    gmm_labels = GaussianMixture(n_components=true_k, covariance_type="full", n_init=5,
                                 random_state=RANDOM_STATE).fit_predict(X)
    # DBSCAN needs no K; eps comes from the k-distance knee (no labels used)
    k_dist, knee_idx = k_distance_curve(X, SYN_DBSCAN_MIN_SAMPLES)
    syn_knee[name] = (k_dist, knee_idx)
    eps = k_dist[knee_idx]
    db_labels = DBSCAN(eps=eps, min_samples=SYN_DBSCAN_MIN_SAMPLES).fit_predict(X)
    for method, labels in [("K-Means", km_labels), ("GMM", gmm_labels), ("DBSCAN", db_labels)]:
        scores, _ = score_labels(X, y_true, labels)
        syn_rows.append({"dataset": name, "method": method, **scores, "true_clusters": true_k,
                         "eps": eps if method == "DBSCAN" else np.nan})
        syn_labels[(name, method)] = labels
syn_results = pd.DataFrame(syn_rows)
print("\nSynthetic benchmark (ARI and V-measure use true labels; silhouette does not):")
print(syn_results.round(3).to_string(index=False))

# Chart 01: the four datasets by true label
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, name in zip(axes, syn_names):
    scatter_labels(ax, syn_scaled[name], syn_datasets[name][1], name)
fig.suptitle("Synthetic Datasets: True Structure (grey crosses = true noise)", fontsize=14, fontweight="bold")
plt.tight_layout()
finish_chart("01_synthetic_ground_truth.png")

# Chart 02: each method on each dataset, with ARI in the title
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
for col, name in enumerate(syn_names):
    for row, method in enumerate(["K-Means", "GMM", "DBSCAN"]):
        res = syn_results[(syn_results["dataset"] == name) & (syn_results["method"] == method)].iloc[0]
        title = f"{method}: ARI {res['ARI']:.2f}"
        if method == "DBSCAN":
            title += f"\neps {res['eps']:.2f}, noise {res['noise_pct']:.1f}%"
        scatter_labels(axes[row, col], syn_scaled[name], syn_labels[(name, method)], title)
    axes[0, col].annotate(name, (0.5, 1.28), xycoords="axes fraction", ha="center", fontsize=12, fontweight="bold")
fig.suptitle("K-Means vs GMM vs DBSCAN on Synthetic Data", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
finish_chart("02_synthetic_method_comparison.png")

# Chart 03: ARI beside silhouette. Where they disagree, the intrinsic score rewards compact shapes
# rather than recovering the true structure
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
sns.barplot(data=syn_results, x="dataset", y="ARI", hue="method", ax=axes[0])
axes[0].set_title("Adjusted Rand Index (uses true labels)", fontsize=12, fontweight="bold")
sns.barplot(data=syn_results, x="dataset", y="silhouette", hue="method", ax=axes[1])
axes[1].set_title("Silhouette Score (no labels; noise points excluded)", fontsize=12, fontweight="bold")
for ax in axes:
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=15)
plt.tight_layout()
finish_chart("03_synthetic_ari_vs_silhouette.png")

# ============================================================================
# 3. PART A (2/2): PARAMETER SELECTION WITHOUT LABELS
# ============================================================================
# Stage 2 gave K-Means and GMM the true K. Real hotspot data has no true K, so this section shows how
# each method's key parameters are chosen without labels, and audits those rules against the known truth
print("\nPARAMETER SELECTION: K-Means K from the elbow and silhouette curves")
kmeans_rows = []
for name in syn_names:
    for k in KMEANS_K_RANGE:
        model = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(syn_scaled[name])
        kmeans_rows.append({"dataset": name, "k": k, "WSS": model.inertia_,
                            "silhouette": silhouette_score(syn_scaled[name], model.labels_)})
kmeans_df = pd.DataFrame(kmeans_rows)
best_k_silhouette = kmeans_df.loc[kmeans_df.groupby("dataset")["silhouette"].idxmax()].set_index("dataset")["k"]
print("K chosen by maximum silhouette:")
print(best_k_silhouette.to_string())

# Chart 04: top row = elbow (WSS), bottom row = silhouette; dotted line = true K
fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True)
for col, name in enumerate(syn_names):
    sub, true_k = kmeans_df[kmeans_df["dataset"] == name], syn_datasets[name][2]
    axes[0, col].plot(sub["k"], sub["WSS"], marker="o")
    axes[0, col].set_title(name, fontsize=11, fontweight="bold")
    axes[1, col].plot(sub["k"], sub["silhouette"], marker="o", color="tab:green")
    chosen = best_k_silhouette[name]
    axes[1, col].plot(chosen, sub.loc[sub["k"] == chosen, "silhouette"].iloc[0], marker="*", color="red",
                      markersize=16, linestyle="none", label="Max silhouette")
    for ax in axes[:, col]:
        ax.axvline(true_k, color="grey", linestyle=":", label="True K")
    axes[1, col].set_xlabel("K")
axes[0, 0].set_ylabel("Within-cluster sum of squares")
axes[1, 0].set_ylabel("Silhouette score")
axes[1, 0].legend(fontsize=8)
fig.suptitle("K-Means: Elbow and Silhouette Curves (dotted line = true K)", fontsize=14, fontweight="bold")
plt.tight_layout()
finish_chart("04_kmeans_k_selection.png")

print("\nPARAMETER SELECTION: GMM components and covariance type from BIC and AIC")
gmm_rows = []
for name in syn_names:
    for covariance in COVARIANCE_TYPES:
        for k in GMM_K_RANGE:
            # n_init=3 restarts guard against EM settling in a poor local optimum
            gmm = GaussianMixture(n_components=k, covariance_type=covariance, n_init=3,
                                  random_state=RANDOM_STATE).fit(syn_scaled[name])
            gmm_rows.append({"dataset": name, "covariance": covariance, "k": k,
                             "BIC": gmm.bic(syn_scaled[name]), "AIC": gmm.aic(syn_scaled[name])})
gmm_df = pd.DataFrame(gmm_rows)
best_bic = gmm_df.loc[gmm_df.groupby("dataset")["BIC"].idxmin()].set_index("dataset")
best_aic = gmm_df.loc[gmm_df.groupby("dataset")["AIC"].idxmin()].set_index("dataset")
print("Best by BIC:")
print(best_bic[["covariance", "k", "BIC"]].to_string())
print("Best by AIC:")
print(best_aic[["covariance", "k", "AIC"]].to_string())

# Chart 05: BIC curves (lower is better); star = overall minimum; dotted line = true K
fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
for ax, name in zip(axes, syn_names):
    sns.lineplot(data=gmm_df[gmm_df["dataset"] == name], x="k", y="BIC", hue="covariance", marker="o", ax=ax,
                 legend=(name == syn_names[0]))
    ax.plot(best_bic.loc[name, "k"], best_bic.loc[name, "BIC"], marker="*", color="red", markersize=16,
            linestyle="none")
    ax.axvline(syn_datasets[name][2], color="grey", linestyle=":")
    ax.set_title(name, fontsize=11, fontweight="bold")
    ax.set_xlabel("Number of components")
axes[0].set_ylabel("BIC (lower is better)")
fig.suptitle("GMM: BIC by Components and Covariance Type (star = minimum, dotted = true K)",
             fontsize=14, fontweight="bold")
plt.tight_layout()
finish_chart("05_gmm_bic_selection.png")

print("\nPARAMETER SELECTION: DBSCAN eps from the k-distance knee, then a sensitivity grid")
# Chart 06: sorted k-distance curves with the knee marked
fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
for ax, name in zip(axes, syn_names):
    k_dist, knee_idx = syn_knee[name]
    ax.plot(k_dist)
    ax.axhline(k_dist[knee_idx], color="red", linestyle="--", label=f"eps = {k_dist[knee_idx]:.3f}")
    ax.axvline(knee_idx, color="red", linestyle=":")
    ax.set_title(name, fontsize=11, fontweight="bold")
    ax.set_xlabel("Points sorted by distance")
    ax.legend(fontsize=9)
axes[0].set_ylabel(f"Distance to {SYN_DBSCAN_MIN_SAMPLES}th nearest neighbour")
fig.suptitle("DBSCAN: Sorted k-Distance Curves with Knee Marked", fontsize=14, fontweight="bold")
plt.tight_layout()
finish_chart("06_dbscan_k_distance.png")

# Sensitivity grid. Labels are used HERE ONLY to audit how fragile the result is on data where the
# truth is known; they play no part in choosing eps or min_samples
grid_rows = []
for name in syn_names:
    X, y_true = syn_scaled[name], syn_datasets[name][1]
    knee_eps = syn_knee[name][0][syn_knee[name][1]]
    for min_samples in SYN_MIN_SAMPLES_GRID:
        for factor in EPS_FACTORS:
            labels = DBSCAN(eps=knee_eps * factor, min_samples=min_samples).fit_predict(X)
            grid_rows.append({"dataset": name, "min_samples": min_samples, "eps_factor": factor,
                              "ARI": adjusted_rand_score(y_true, labels), "clusters": len(set(labels) - {-1})})
syn_grid = pd.DataFrame(grid_rows)

# Chart 07: top row = ARI, bottom row = number of clusters found
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
for col, name in enumerate(syn_names):
    sub = syn_grid[syn_grid["dataset"] == name]
    knee_eps = syn_knee[name][0][syn_knee[name][1]]
    sns.heatmap(sub.pivot(index="min_samples", columns="eps_factor", values="ARI"), annot=True, fmt=".2f",
                cmap="viridis", vmin=0, vmax=1, cbar=False, ax=axes[0, col])
    sns.heatmap(sub.pivot(index="min_samples", columns="eps_factor", values="clusters"), annot=True, fmt="d",
                cmap="Greys", cbar=False, ax=axes[1, col])
    axes[0, col].set_title(f"{name}\n(knee eps = {knee_eps:.2f})", fontsize=11, fontweight="bold")
    axes[0, col].set_xlabel("")
    axes[1, col].set_xlabel("eps as a multiple of knee eps")
    axes[0, col].set_ylabel("min_samples" if col == 0 else "")
    axes[1, col].set_ylabel("min_samples" if col == 0 else "")
fig.suptitle("DBSCAN Sensitivity: ARI (top) and Number of Clusters Found (bottom)", fontsize=14, fontweight="bold")
plt.tight_layout()
finish_chart("07_dbscan_sensitivity.png")

stability = syn_grid.groupby("dataset")["ARI"].apply(lambda s: 100 * (s >= 0.8).mean())
n_grid_settings = len(SYN_MIN_SAMPLES_GRID) * len(EPS_FACTORS)
print(f"\nShare of the {n_grid_settings} eps x min_samples settings with ARI >= 0.8 (%):")
print(stability.round(1).to_string())
selection_summary = pd.DataFrame({
    "true_k": {name: syn_datasets[name][2] for name in syn_names},
    "kmeans_k_silhouette": best_k_silhouette, "gmm_k_bic": best_bic["k"], "gmm_cov_bic": best_bic["covariance"],
    "gmm_k_aic": best_aic["k"], "gmm_cov_aic": best_aic["covariance"],
    "dbscan_knee_eps": pd.Series({n: syn_knee[n][0][syn_knee[n][1]] for n in syn_names}),
    "dbscan_pct_grid_ari_ge_0.8": stability}).loc[syn_names]
print("\nSummary of label-free parameter selection against the known truth:")
print(selection_summary.round(3).to_string())

# ============================================================================
# 4. PART B: WHEAT SEEDS BRIDGE
# ============================================================================
# GMM and DBSCAN on the SAME data, preprocessing and seed as the K-Means Clustering project, so the three
# methods compare like-for-like against that page. The varieties overlap in feature space with no
# low-density gaps between them: a regime where DBSCAN's assumptions are weak and GMM's flexibility may
# help (or overfit), so the result is informative whichever way it falls
print("\nPART B: Wheat Seeds bridge")
wheat_results = None
if not os.path.exists(SEED_FILE):
    print(f"  '{SEED_FILE}' not found in the working directory; skipping Part B. It is the same file as the "
          f"K-Means Clustering project uses (7 features first, variety 0/1/2 last).")
else:
    wheat_df = pd.read_csv(SEED_FILE)
    wheat_raw = wheat_df.iloc[:, :-1].values          # seven geometric features
    wheat_y = wheat_df.iloc[:, -1].values             # true variety: used ONLY for evaluation
    print(f"Dataset shape: {wheat_df.shape}; classes: {dict(zip(*np.unique(wheat_y, return_counts=True)))}")
    # Standardise: features sit on very different scales (area ~15, compactness ~0.87) and all three
    # methods rely on distances or covariances that scale would otherwise distort
    wheat_X = StandardScaler().fit_transform(wheat_raw)
    wheat_pca = PCA(n_components=2)
    wheat_pca_xy = wheat_pca.fit_transform(wheat_X)   # plotting only, as in the K-Means project

    wheat_labels, wheat_aligned, wheat_rows = {}, {}, []

    def add_wheat_result(method, setting, labels, uses_labels=False):
        scores, aligned = score_labels(wheat_X, wheat_y, labels, with_agreement=True)
        wheat_rows.append({"method": method, "setting": setting, "uses_labels_to_choose": uses_labels, **scores})
        wheat_labels[method], wheat_aligned[method] = labels, aligned

    km_wheat = KMeans(n_clusters=WHEAT_TRUE_K, random_state=RANDOM_STATE, n_init=10).fit_predict(wheat_X)
    add_wheat_result("K-Means", f"K={WHEAT_TRUE_K}", km_wheat)
    first = wheat_rows[0]
    print(f"Reproduced K-Means page: ARI {first['ARI']:.3f} (page {KMEANS_PAGE_ARI}), "
          f"silhouette {first['silhouette']:.2f} (page {KMEANS_PAGE_SILHOUETTE}), "
          f"matched agreement {first['matched_agreement_pct']:.2f}% (page {100 * KMEANS_PAGE_AGREEMENT:.2f}%)")

    for covariance in COVARIANCE_TYPES:
        # n_init=10 matches K-Means, so a poor EM start is not mistaken for a poor model
        gmm = GaussianMixture(n_components=WHEAT_TRUE_K, covariance_type=covariance, n_init=10,
                              random_state=RANDOM_STATE).fit(wheat_X)
        add_wheat_result(f"GMM ({covariance}, K=3)", f"BIC {gmm.bic(wheat_X):.0f}", gmm.predict(wheat_X))

    # Label-free GMM selection by BIC over components and covariance type (the Part A rule)
    wheat_gmm_grid = []
    for covariance in COVARIANCE_TYPES:
        for k in GMM_K_RANGE:
            gmm = GaussianMixture(n_components=k, covariance_type=covariance, n_init=5,
                                  random_state=RANDOM_STATE).fit(wheat_X)
            wheat_gmm_grid.append({"covariance": covariance, "k": k, "BIC": gmm.bic(wheat_X), "AIC": gmm.aic(wheat_X)})
    wheat_gmm_grid = pd.DataFrame(wheat_gmm_grid)
    wheat_best = wheat_gmm_grid.loc[wheat_gmm_grid["BIC"].idxmin()]
    print(f"BIC selects: {wheat_best['covariance']} covariance with {int(wheat_best['k'])} components")
    bic_model = GaussianMixture(n_components=int(wheat_best["k"]), covariance_type=wheat_best["covariance"],
                                n_init=10, random_state=RANDOM_STATE).fit(wheat_X)
    add_wheat_result("GMM (BIC-selected)", f"{wheat_best['covariance']}, K={int(wheat_best['k'])}",
                     bic_model.predict(wheat_X))

    # DBSCAN. Rule of thumb: min_samples of at least 2 x the number of features (here 14), so that a
    # 'dense' neighbourhood is meaningful in seven dimensions
    wheat_min_samples = 2 * wheat_X.shape[1]
    wheat_kdist, wheat_knee_idx = k_distance_curve(wheat_X, wheat_min_samples)
    wheat_knee_eps = wheat_kdist[wheat_knee_idx]
    print(f"DBSCAN: min_samples = {wheat_min_samples}, knee eps = {wheat_knee_eps:.3f}")
    add_wheat_result("DBSCAN (knee eps)", f"eps {wheat_knee_eps:.2f}, min_samples {wheat_min_samples}",
                     DBSCAN(eps=wheat_knee_eps, min_samples=wheat_min_samples).fit_predict(wheat_X))
    # Sensitivity grid. Labels are used here ONLY to find the best case DBSCAN could achieve, an upper
    # bound: if even this is poor, no choice of settings rescues DBSCAN on this data
    wheat_grid = []
    for min_samples in [5, 10, wheat_min_samples, 20]:
        for factor in EPS_FACTORS:
            labels = DBSCAN(eps=wheat_knee_eps * factor, min_samples=min_samples).fit_predict(wheat_X)
            scores, _ = score_labels(wheat_X, wheat_y, labels)
            wheat_grid.append({"min_samples": min_samples, "eps_factor": factor, "eps": wheat_knee_eps * factor,
                               **scores})
    wheat_grid = pd.DataFrame(wheat_grid)
    best_row = wheat_grid.loc[wheat_grid["ARI"].idxmax()]
    add_wheat_result("DBSCAN (best grid, oracle)", f"eps {best_row['eps']:.2f}, min_samples {int(best_row['min_samples'])}",
                     DBSCAN(eps=best_row["eps"], min_samples=int(best_row["min_samples"])).fit_predict(wheat_X),
                     uses_labels=True)

    wheat_results = pd.DataFrame(wheat_rows)
    print("\nWheat Seeds: method comparison (ARI, V-measure and agreement use the true varieties):")
    print(wheat_results.round(3).to_string(index=False))

    # Chart 08: PCA projection coloured by the true varieties and by each method
    palette = sns.color_palette("tab10")
    panels = [("True varieties", wheat_y), ("K-Means", wheat_labels["K-Means"]),
              ("GMM (full, K=3)", wheat_labels["GMM (full, K=3)"]), ("GMM (tied, K=3)", wheat_labels["GMM (tied, K=3)"]),
              ("GMM (BIC-selected)", wheat_labels["GMM (BIC-selected)"]), ("DBSCAN (knee eps)", wheat_labels["DBSCAN (knee eps)"])]
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    for ax, (title, labels) in zip(axes.ravel(), panels):
        for label in np.unique(labels):
            mask = labels == label
            if label == -1:
                ax.scatter(wheat_pca_xy[mask, 0], wheat_pca_xy[mask, 1], s=25, c="grey", marker="x")
            else:
                ax.scatter(wheat_pca_xy[mask, 0], wheat_pca_xy[mask, 1], s=25, color=palette[label % 10],
                           edgecolor="white", linewidth=0.3)
        if title != "True varieties":
            res = wheat_results[wheat_results["method"] == title].iloc[0]
            title = f"{title}\nARI {res['ARI']:.2f}, agreement {res['matched_agreement_pct']:.1f}%"
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel(f"PC1 ({100 * wheat_pca.explained_variance_ratio_[0]:.1f}% of variance)")
        ax.set_ylabel(f"PC2 ({100 * wheat_pca.explained_variance_ratio_[1]:.1f}% of variance)")
    fig.suptitle("Wheat Seeds in PCA Space: True Varieties vs Each Method", fontsize=15, fontweight="bold")
    plt.tight_layout()
    finish_chart("08_wheat_pca_comparison.png")

    # Chart 09: matched contingency tables (matches on the diagonal; noise at the right)
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.8))
    for ax, key in zip(axes, ["K-Means", "GMM (full, K=3)", "GMM (tied, K=3)", "DBSCAN (knee eps)"]):
        sns.heatmap(wheat_aligned[key], annot=True, fmt="d", cmap="Blues", cbar=False, linewidths=0.5, ax=ax)
        res = wheat_results[wheat_results["method"] == key].iloc[0]
        ax.set_title(f"{key}\nagreement {res['matched_agreement_pct']:.2f}%", fontsize=11, fontweight="bold")
        ax.set_xlabel("Cluster (reordered to match varieties)")
        ax.set_ylabel("True variety" if ax is axes[0] else "")
    fig.suptitle("Contingency Tables: True Varieties vs Matched Clusters", fontsize=14, fontweight="bold")
    plt.tight_layout()
    finish_chart("09_wheat_contingency_tables.png")

    # Chart 10: GMM BIC and AIC (lower is better); star = minimum; dotted line = true K
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, criterion in zip(axes, ["BIC", "AIC"]):
        sns.lineplot(data=wheat_gmm_grid, x="k", y=criterion, hue="covariance", marker="o", ax=ax)
        best = wheat_gmm_grid.loc[wheat_gmm_grid[criterion].idxmin()]
        ax.plot(best["k"], best[criterion], marker="*", color="red", markersize=16, linestyle="none")
        ax.axvline(WHEAT_TRUE_K, color="grey", linestyle=":")
        ax.set_title(f"{criterion} by Components and Covariance Type", fontsize=12, fontweight="bold")
        ax.set_xlabel("Number of components")
    plt.tight_layout()
    finish_chart("10_wheat_gmm_bic_aic.png")

    # Chart 11: DBSCAN diagnostics: k-distance curve, then ARI, clusters and noise grids
    fig, axes = plt.subplots(1, 4, figsize=(22, 4.8))
    axes[0].plot(wheat_kdist)
    axes[0].axhline(wheat_knee_eps, color="red", linestyle="--", label=f"eps = {wheat_knee_eps:.2f}")
    axes[0].axvline(wheat_knee_idx, color="red", linestyle=":")
    axes[0].set_title(f"k-distance curve (k = {wheat_min_samples})", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Points sorted by distance")
    axes[0].set_ylabel(f"Distance to {wheat_min_samples}th nearest neighbour")
    axes[0].legend()
    for ax, (column, fmt, cmap, title) in zip(axes[1:], [("ARI", ".2f", "viridis", "ARI"),
                                                         ("clusters", "d", "Greys", "Clusters found"),
                                                         ("noise_pct", ".0f", "Reds", "Noise (%)")]):
        sns.heatmap(wheat_grid.pivot(index="min_samples", columns="eps_factor", values=column), annot=True, fmt=fmt,
                    cmap=cmap, cbar=False, ax=ax, **({"vmin": 0, "vmax": 1} if column == "ARI" else {}))
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("eps as a multiple of knee eps")
    fig.suptitle("DBSCAN on Wheat Seeds: Diagnostics and Sensitivity", fontsize=14, fontweight="bold")
    plt.tight_layout()
    finish_chart("11_wheat_dbscan_diagnostics.png")

# ============================================================================
# 5. PART C (1/3): KENT DATA LOADING, STUDY AREA AND AUDIT
# ============================================================================
def download_if_missing(year):
    """Download the collision CSV for a year, caching it in DATA_DIR."""
    file_name = FILE_TEMPLATE.format(year=year)
    local_path = os.path.join(DATA_DIR, file_name)
    if os.path.exists(local_path):
        print(f"Using cached file: {local_path}")
        return local_path
    url = BASE_URL + file_name
    print(f"Downloading {url}")
    # A browser-style User-Agent avoids some servers rejecting default Python requests
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(request, timeout=180) as response, open(local_path, "wb") as out_file:
            out_file.write(response.read())
    except Exception as exc:
        # Fail with a clear manual fallback rather than a long traceback
        raise SystemExit(f"Download failed ({exc}). Save '{file_name}' manually as {local_path} and re-run.")
    return local_path


def find_column(df, candidates, label, required=True):
    """Return the first matching column name. DfT field names change between releases, so each
    field is looked up from a list of candidates rather than hard-coded."""
    for name in candidates:
        if name in df.columns:
            return name
    if required:
        raise SystemExit(f"Could not find a column for '{label}'. Tried {candidates}.\n"
                         f"Columns available: {list(df.columns)}")
    print(f"  WARNING: no column found for '{label}'; it will be left empty.")
    return None


def load_year(year):
    """Load one year of national collisions as a minimal, standardised dataframe."""
    raw = pd.read_csv(download_if_missing(year), na_values=["NULL"], low_memory=False)   # DfT codes missing as 'NULL'
    raw.columns = [c.strip().lower() for c in raw.columns]
    force_col = find_column(raw, ["police_force"], "police force")
    east_col = find_column(raw, ["location_easting_osgr", "easting"], "OSGR easting")
    north_col = find_column(raw, ["location_northing_osgr", "northing"], "OSGR northing")
    sev_col = find_column(raw, ["collision_severity", "accident_severity", "legacy_collision_severity"], "severity")
    urban_col = find_column(raw, ["urban_or_rural_area"], "urban or rural", required=False)

    def numeric(col):
        return pd.Series(np.nan, index=raw.index) if col is None else pd.to_numeric(raw[col], errors="coerce")

    out = pd.DataFrame({"year": year, "police_force": numeric(force_col), "easting": numeric(east_col),
                        "northing": numeric(north_col), "severity": numeric(sev_col),   # 1 fatal, 2 serious, 3 slight
                        "urban_rural": numeric(urban_col)})                              # 1 urban, 2 rural
    # A coordinate is valid only if both values are present and inside the grid limits
    out["coords_valid"] = out["easting"].between(*EASTING_RANGE) & out["northing"].between(*NORTHING_RANGE)
    return out


def extent_km2(valid_rows):
    """Approximate area a force covers: the 1st-99th percentile bounding box, in km2 (percentiles stop a
    few stray locations inflating it)."""
    if len(valid_rows) < 10:
        return np.nan
    width = valid_rows["easting"].quantile(0.99) - valid_rows["easting"].quantile(0.01)
    height = valid_rows["northing"].quantile(0.99) - valid_rows["northing"].quantile(0.01)
    return width * height / 1e6


def force_table(national_fit, national_test):
    """One row per police force: volume in both years, coordinate quality, urban share, KSI share, extent.
    This is the evidence used to choose the study area; Kent was chosen from it (see below)."""
    rows = []
    for force, group in national_fit.groupby("police_force"):
        urban_known, sev_known = group["urban_rural"].isin([1, 2]), group["severity"].isin([1, 2, 3])
        rows.append({
            "police_force": int(force), "collisions_fit": len(group),
            "missing_coord_pct": 100 * (1 - group["coords_valid"].mean()),
            "urban_share_pct": 100 * (group["urban_rural"] == 1).sum() / urban_known.sum() if urban_known.any() else np.nan,
            "ksi_pct": 100 * group["severity"].isin([1, 2]).sum() / sev_known.sum() if sev_known.any() else np.nan,
            "extent_km2": extent_km2(group[group["coords_valid"]]),
        })
    table = pd.DataFrame(rows).set_index("police_force")
    counts_test = national_test.groupby("police_force").size().rename("collisions_test")
    counts_test.index = counts_test.index.astype(int)
    table = table.join(counts_test)
    # These thresholds only FLAG forces for review; the final choice is a judgement (see below)
    table["candidate"] = ((table["collisions_fit"] >= MIN_COLLISIONS_PER_YEAR)
                          & (table["collisions_test"] >= MIN_COLLISIONS_PER_YEAR)
                          & (table["missing_coord_pct"] <= MAX_MISSING_COORD_PCT))
    return table.sort_values("collisions_fit", ascending=False)


def kent_frames(national_fit, national_test, fit_year, test_year, verbose=True):
    """Kent fit and test frames (valid locations only), with the data audit printed."""
    frames = {fit_year: national_fit, test_year: national_test}
    kent_all = {y: d[d["police_force"] == FORCE_CODE].reset_index(drop=True) for y, d in frames.items()}
    both = pd.concat(kent_all.values(), ignore_index=True)    # ignore_index avoids duplicate row labels
    if verbose:
        print(f"\nKent (force {FORCE_CODE}) collisions:")
        for year, frame in kent_all.items():
            print(f"  {year}: {len(frame):,} collisions, {(~frame['coords_valid']).sum()} without a valid location")
        # Is the change in volume spread evenly across severities and area types?
        for column, title in [("severity", "severity (1 fatal, 2 serious, 3 slight)"),
                              ("urban_rural", "area type (1 urban, 2 rural, 3 unallocated)")]:
            table = pd.crosstab(both["year"], both[column]).T
            table["change_pct"] = 100 * (table[test_year] / table[fit_year] - 1)
            print(f"\nKent collisions by {title} and year:")
            print(table.round(1))
        ksi = both.groupby("year")["severity"].apply(lambda s: 100 * s.isin([1, 2]).mean())
        print(f"\nFatal-or-serious (KSI) share: {fit_year} {ksi[fit_year]:.1f}%, {test_year} {ksi[test_year]:.1f}%. "
              f"Severity enrichment is therefore measured against each year's own baseline")
    fit = kent_all[fit_year][kent_all[fit_year]["coords_valid"]].reset_index(drop=True)
    test = kent_all[test_year][kent_all[test_year]["coords_valid"]].reset_index(drop=True)
    if verbose:
        dup = 100 * fit.duplicated(subset=["easting", "northing"], keep=False).mean()
        print(f"Fit-year collisions used: {len(fit):,}; {dup:.1f}% share an exact location with another collision")
    return fit, test


print("\nPART C: Kent road collisions")
national = {year: load_year(year) for year in (FIT_YEAR, TEST_YEAR)}
forces = force_table(national[FIT_YEAR], national[TEST_YEAR])
print(f"\nStudy area: police forces passing the volume and coordinate-quality flags in both years "
      f"({FIT_YEAR} and {TEST_YEAR}):")
print(forces[forces["candidate"]].round(1).to_string())
def force_fact(code, column):
    return forces.loc[code, column] if code in forces.index else float("nan")


print(f"Kent ({FORCE_CODE}) is chosen: it has a balanced urban/rural mix ({force_fact(FORCE_CODE, 'urban_share_pct'):.0f}% "
      f"urban), a county-scale extent ({force_fact(FORCE_CODE, 'extent_km2'):,.0f} km2) and enough collisions "
      f"({force_fact(FORCE_CODE, 'collisions_fit'):,.0f} in {FIT_YEAR}) for DBSCAN to find hotspots. The Metropolitan "
      f"Police (1) is {force_fact(1, 'urban_share_pct'):.0f}% urban, so there is no meaningful rural noise; West "
      f"Midlands (20) and West Yorkshire (13) are {force_fact(20, 'urban_share_pct'):.0f}% and "
      f"{force_fact(13, 'urban_share_pct'):.0f}% urban; Police Scotland (99) spans "
      f"{force_fact(99, 'extent_km2'):,.0f} km2 with highly varied density, which a single DBSCAN eps handles badly.")
kent_fit_df, kent_test_df = kent_frames(national[FIT_YEAR], national[TEST_YEAR], FIT_YEAR, TEST_YEAR)


# ============================================================================
# 6. PART C (2/3): FITTING THE METHODS TO THE FIT YEAR
# ============================================================================
# Key decisions (each explained where it is made):
#   1. Cluster on OSGR easting/northing in METRES, unscaled, so eps is a real distance
#   2. The headline 'Site' DBSCAN setting is chosen by a physical-size guard over the whole
#      eps x min_samples grid, not from the k-distance knee alone
#   3. DBSCAN is reported at three scales (Site, Strict site, Area) so conclusions are shown to
#      depend on scale instead of hiding one choice
#   4. K-Means and GMM get as many clusters as the Site tier found, plus a larger K as a sensitivity
def cluster_radii(xy_m, labels):
    """Radius of each cluster: distance from its centroid to its furthest member, in metres."""
    radii = []
    for label in set(labels) - {-1}:
        points = xy_m[labels == label]
        radii.append(np.hypot(*(points - points.mean(axis=0)).T).max())
    return np.array(radii)


def run_dbscan(xy_m, eps_m, min_samples):
    """Fit DBSCAN in metres and return the fitted model with a one-row summary."""
    model = DBSCAN(eps=eps_m, min_samples=min_samples).fit(xy_m)
    labels = model.labels_
    sizes = pd.Series(labels[labels != -1]).value_counts()
    radii = cluster_radii(xy_m, labels)
    return model, {
        "eps_m": eps_m, "min_samples": min_samples, "clusters": len(sizes),
        "noise_pct": 100 * (labels == -1).mean(), "pct_in_hotspots": 100 * (labels != -1).mean(),
        "median_cluster_size": sizes.median() if len(sizes) else 0.0,
        "median_radius_m": np.median(radii) if len(radii) else 0.0,
        "max_radius_m": radii.max() if len(radii) else 0.0,
    }


def fit_kent_models(fit_df, site_override=None, with_bic=False, verbose=True):
    """Fit DBSCAN (three tiers), K-Means and GMM to one year of Kent collisions."""
    xy_m = fit_df[["easting", "northing"]].to_numpy(dtype=float)   # metres, deliberately NOT standardised:
                                                                   # standardising stretches the axes unequally
                                                                   # and makes eps meaningless as a distance
    xy_km = xy_m / 1000.0                                          # GMM and K-Means work in km for numerical comfort
    n = len(xy_m)

    grid_df = pd.DataFrame([run_dbscan(xy_m, eps, ms)[1] for ms in MIN_SAMPLES_GRID for eps in EPS_GRID_M])
    # The k-distance knee is shown for reference only. Kent mixes dense towns with sparse rural roads, so a
    # single global knee is expected to sit far above a site-scale distance (the varied-density weakness
    # of DBSCAN seen on the synthetic data in Part A)
    k_dist, knee_idx = k_distance_curve(xy_m, REFERENCE_MIN_SAMPLES)

    # Site tier = the grid setting with the most collisions inside hotspots among those whose hotspots all
    # respect the size guard. Coverage is maximised because a method that flags almost nothing is of little
    # use, and the guard stops that coverage coming from chaining several sites into one town-sized cluster
    candidates = grid_df[(grid_df["clusters"] > 0) & (grid_df["max_radius_m"] <= MAX_HOTSPOT_RADIUS_M)]
    candidates = candidates.sort_values(["pct_in_hotspots", "eps_m"], ascending=[False, True])
    auto_site = (float(candidates.iloc[0]["eps_m"]), int(candidates.iloc[0]["min_samples"]))
    site = site_override if site_override is not None else auto_site
    if verbose:
        print(f"Settings passing the size guard (radius <= {MAX_HOTSPOT_RADIUS_M} m), best coverage first:")
        print(candidates.head(6)[["eps_m", "min_samples", "clusters", "pct_in_hotspots", "median_radius_m",
                                  "max_radius_m"]].round(1).to_string(index=False))
        print(f"k-distance knee eps at min_samples {REFERENCE_MIN_SAMPLES} (reference only): {k_dist[knee_idx]:.0f} m")
        print(f"Site tier: eps {site[0]:.0f} m, min_samples {site[1]}"
              + (" (OVERRIDE)" if site_override is not None else " (selected by the size guard)"))

    tiers = {"Site": site, "Strict site": STRICT_TIER, "Area": AREA_TIER}
    tier_models, tier_rows = {}, []
    for tier, (eps_m, min_samples) in tiers.items():
        tier_models[tier], row = run_dbscan(xy_m, eps_m, min_samples)
        tier_rows.append({"tier": tier, **row})
    tier_df = pd.DataFrame(tier_rows)
    if verbose:
        print("DBSCAN tiers:")
        print(tier_df[["tier", "eps_m", "min_samples", "clusters", "pct_in_hotspots", "median_cluster_size",
                       "median_radius_m", "max_radius_m"]].round(1).to_string(index=False))

    # K-Means and GMM get as many clusters as the Site tier found (matched granularity), plus a larger K
    # as a sensitivity. DBSCAN needs no K, so giving the others its cluster count is generous to them
    k_match = max(int(tier_df.loc[tier_df["tier"] == "Site", "clusters"].iloc[0]), 2)
    k_values = [k_match, max(K_LARGE, 2 * k_match)]
    km_labels, gmm_models, gmm_labels = {}, {}, {}
    for k in k_values:
        km_labels[k] = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(xy_km).labels_
        gmm_models[k] = GaussianMixture(n_components=k, covariance_type="full", n_init=2, max_iter=200,
                                        reg_covar=GMM_REG_COVAR, random_state=RANDOM_STATE).fit(xy_km)
        gmm_labels[k] = gmm_models[k].predict(xy_km)
        if verbose:
            print(f"K = {k}: GMM converged = {gmm_models[k].converged_}")

    result = {"xy_m": xy_m, "xy_km": xy_km, "grid_df": grid_df, "k_dist": k_dist, "knee_eps_m": k_dist[knee_idx],
              "tiers": tiers, "tier_models": tier_models, "tier_df": tier_df, "k_values": k_values,
              "km_labels": km_labels, "gmm_models": gmm_models, "gmm_labels": gmm_labels}
    if with_bic:
        # GMM's own label-free choice, for reference (BIC against number of components)
        bic_rows = []
        for k in [k for k in GMM_BIC_K_GRID if k < n // 10]:
            candidate = GaussianMixture(n_components=k, covariance_type="full", n_init=2, max_iter=200,
                                        reg_covar=GMM_REG_COVAR, random_state=RANDOM_STATE).fit(xy_km)
            bic_rows.append({"k": k, "BIC": candidate.bic(xy_km)})
        result["bic_df"] = pd.DataFrame(bic_rows)
    return result


def grid_ref(easting, northing):
    """Four-figure Ordnance Survey grid reference (e.g. 'TQ 7655') so hotspots can be looked up on a map.
    Only the two 100 km squares covering Kent are handled."""
    square = {(5, 1): "TQ", (6, 1): "TR"}.get((int(easting // 100_000), int(northing // 100_000)), "")
    return f"{square} {int(easting % 100_000 // 1000):02d}{int(northing % 100_000 // 1000):02d}"


def hotspot_table(fit_df, labels):
    """One row per hotspot: size, severity mix, area type, location and radius."""
    rows = []
    for label in sorted(set(labels) - {-1}):
        members = fit_df[labels == label]
        centre_e, centre_n = members["easting"].mean(), members["northing"].mean()
        rows.append({
            "hotspot": label, "collisions": len(members), "share_of_all_pct": 100 * len(members) / len(fit_df),
            "ksi_pct": 100 * members["severity"].isin([1, 2]).mean(),
            "urban_pct": 100 * (members["urban_rural"] == 1).mean(), "grid_ref": grid_ref(centre_e, centre_n),
            "max_radius_m": np.hypot(members["easting"] - centre_e, members["northing"] - centre_n).max()})
    return pd.DataFrame(rows).sort_values("collisions", ascending=False).reset_index(drop=True)


print("\nFITTING THE METHODS TO THE FIT YEAR")
kent_fit = fit_kent_models(kent_fit_df, site_override=SITE_OVERRIDE, with_bic=True)
k_match, k_large = kent_fit["k_values"]
site_eps_m, site_ms = kent_fit["tiers"]["Site"]
site_labels_fit = kent_fit["tier_models"]["Site"].labels_

hotspots = hotspot_table(kent_fit_df, site_labels_fit)
overall_ksi = 100 * kent_fit_df["severity"].isin([1, 2]).mean()
print(f"\nTop 10 Site hotspots (overall KSI share in the fit year: {overall_ksi:.1f}%):")
print(hotspots.head(10).round(1).to_string(index=False))
hotspots.to_csv(os.path.join(DATA_DIR, "kent_site_hotspots.csv"), index=False)

best_bic_k = int(kent_fit["bic_df"].loc[kent_fit["bic_df"]["BIC"].idxmin(), "k"])
print(f"GMM BIC is lowest at K = {best_bic_k} (grid {GMM_BIC_K_GRID})")
# GMM's soft membership at the matched K: 1 - the highest posterior probability is an uncertainty score
xy_km = kent_fit["xy_km"]
uncertainty = 1 - kent_fit["gmm_models"][k_match].predict_proba(xy_km).max(axis=1)
print(f"GMM with {k_match} components: {100 * (uncertainty > 0.2).mean():.1f}% of collisions have a posterior "
      f"membership probability below 80% (boundary cases)")


def mean_silhouette(xy, labels):
    """Silhouette on assigned collisions only (noise excluded), in km."""
    mask = labels != -1
    return silhouette_score(xy[mask], labels[mask]) if len(np.unique(labels[mask])) >= 2 else np.nan


comparison_rows = []
for k in kent_fit["k_values"]:
    for name, labels in [("K-Means", kent_fit["km_labels"][k]), ("GMM (full)", kent_fit["gmm_labels"][k])]:
        comparison_rows.append({"method": f"{name}, K={k}", "clusters": k, "assigned_pct": 100.0,
                                "median_cluster_size": pd.Series(labels).value_counts().median(),
                                "silhouette": mean_silhouette(xy_km, labels)})
for tier, model in kent_fit["tier_models"].items():
    labels = model.labels_
    eps_m, min_samples = kent_fit["tiers"][tier]
    comparison_rows.append({"method": f"DBSCAN {tier} ({eps_m:.0f} m, {min_samples})",
                            "clusters": len(set(labels) - {-1}), "assigned_pct": 100 * (labels != -1).mean(),
                            "median_cluster_size": pd.Series(labels[labels != -1]).value_counts().median(),
                            "silhouette": mean_silhouette(xy_km, labels)})
print("\nFit-year method comparison (the DBSCAN silhouette covers hotspot members only, so it is not comparable "
      "with the other rows; Part A showed why intrinsic scores mislead):")
print(pd.DataFrame(comparison_rows).round(2).to_string(index=False))

# ============================================================================
# 7. PART C (3/3): FIT-YEAR CHARTS
# ============================================================================
cluster_palette = sns.color_palette("tab20")


def style_map(ax, title):
    """Equal-aspect map axes in km (no basemap, so the script needs no tile downloads)."""
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_aspect("equal")
    ax.set_xlabel("Easting (km)")
    ax.set_ylabel("Northing (km)")


def draw_clusters(ax, xy, labels, title):
    """Scatter coloured by cluster; noise (-1) as small light-grey points."""
    noise = labels == -1
    ax.scatter(xy[noise, 0], xy[noise, 1], s=3, color="lightgrey")
    for label in np.unique(labels[~noise]):
        mask = labels == label
        ax.scatter(xy[mask, 0], xy[mask, 1], s=6, color=cluster_palette[label % 20])
    style_map(ax, title)


# Chart 12: all fit-year collisions, with fatal and serious highlighted
fig, ax = plt.subplots(figsize=(10, 7))
sev = kent_fit_df["severity"].to_numpy()
ax.scatter(xy_km[sev == 3, 0], xy_km[sev == 3, 1], s=3, color="lightgrey", label="Slight")
ax.scatter(xy_km[sev == 2, 0], xy_km[sev == 2, 1], s=8, color="tab:orange", label="Serious")
ax.scatter(xy_km[sev == 1, 0], xy_km[sev == 1, 1], s=18, color="tab:red", label="Fatal")
style_map(ax, f"Kent Injury Collisions {FIT_YEAR} (n = {len(kent_fit_df):,})")
ax.legend(markerscale=2)
plt.tight_layout()
finish_chart("12_kent_collisions_map.png")

# Chart 13: k-distance curve, then hotspots, noise and largest-radius grids over eps x min_samples
fig, axes = plt.subplots(1, 4, figsize=(22, 5))
axes[0].plot(kent_fit["k_dist"])
axes[0].axhline(kent_fit["knee_eps_m"], color="grey", linestyle="--", label=f"knee {kent_fit['knee_eps_m']:.0f} m")
axes[0].axhline(site_eps_m, color="red", linestyle="--", label=f"Site eps {site_eps_m:.0f} m")
axes[0].set_yscale("log")
axes[0].set_title(f"k-distance curve (k = {REFERENCE_MIN_SAMPLES})", fontsize=11, fontweight="bold")
axes[0].set_xlabel("Collisions sorted by distance")
axes[0].set_ylabel(f"Distance to {REFERENCE_MIN_SAMPLES}th nearest collision (m, log scale)")
axes[0].legend()
for ax, (column, fmt, cmap, title) in zip(axes[1:], [("clusters", "d", "Greys", "Hotspots found"),
                                                     ("noise_pct", ".0f", "Reds", "Noise (% of collisions)"),
                                                     ("max_radius_m", ".0f", "Oranges", "Largest hotspot radius (m)")]):
    sns.heatmap(kent_fit["grid_df"].pivot(index="min_samples", columns="eps_m", values=column), annot=True, fmt=fmt,
                cmap=cmap, cbar=False, ax=ax)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("eps (m)")
fig.suptitle("DBSCAN on Kent Collisions: Diagnostics and Sensitivity", fontsize=14, fontweight="bold")
plt.tight_layout()
finish_chart("13_kent_dbscan_diagnostics.png")

# Chart 14: the three methods on the same map (stacked: Kent is about twice as wide as it is tall, so
# side-by-side equal-aspect maps would be tiny)
fig, axes = plt.subplots(3, 1, figsize=(11, 16))
draw_clusters(axes[0], xy_km, kent_fit["km_labels"][k_match], f"K-Means (K = {k_match})")
draw_clusters(axes[1], xy_km, kent_fit["gmm_labels"][k_match], f"GMM, full covariance (K = {k_match})")
draw_clusters(axes[2], xy_km, site_labels_fit, f"DBSCAN Site tier (eps {site_eps_m:.0f} m, min_samples {site_ms}): "
                                               f"{k_match} hotspots, grey = noise")
fig.suptitle(f"Kent {FIT_YEAR}: K-Means vs GMM vs DBSCAN at Matched Granularity", fontsize=15, fontweight="bold")
plt.tight_layout()
finish_chart("14_kent_method_maps.png")

# Chart 15: GMM BIC against number of components
bic_df = kent_fit["bic_df"]
plt.figure(figsize=(9, 5))
plt.plot(bic_df["k"], bic_df["BIC"], marker="o")
plt.plot(best_bic_k, bic_df["BIC"].min(), marker="*", color="red", markersize=16, linestyle="none",
         label=f"Lowest BIC (K = {best_bic_k})")
plt.axvline(k_match, color="grey", linestyle=":", label=f"Matched to DBSCAN Site (K = {k_match})")
plt.axvline(k_large, color="grey", linestyle="--", label=f"Sensitivity (K = {k_large})")
plt.title("GMM: BIC by Number of Components (full covariance)", fontsize=13, fontweight="bold")
plt.xlabel("Number of components")
plt.ylabel("BIC (lower is better)")
plt.legend()
plt.tight_layout()
finish_chart("15_kent_gmm_bic.png")

# Chart 16: the largest Site hotspots with their severity and area-type mix
top = hotspots.head(15).iloc[::-1]
fig, ax = plt.subplots(figsize=(10, 7))
ax.barh([f"#{h} ({g})" for h, g in zip(top["hotspot"], top["grid_ref"])], top["collisions"], color="tab:blue")
for i, (_, r) in enumerate(top.iterrows()):
    ax.text(r["collisions"] + 0.15, i, f"KSI {r['ksi_pct']:.0f}%, urban {r['urban_pct']:.0f}%, "
            f"radius {r['max_radius_m']:.0f} m", va="center", fontsize=8)
ax.set_xlim(0, top["collisions"].max() * 1.5)
ax.set_title(f"Largest DBSCAN Site Hotspots, Kent {FIT_YEAR} (overall KSI share {overall_ksi:.0f}%)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Collisions in the fit year")
ax.set_ylabel("Hotspot ID (grid reference)")
plt.tight_layout()
finish_chart("16_kent_top_hotspots.png")

# Chart 17: GMM membership uncertainty (1 - highest posterior probability) at the matched K
fig, ax = plt.subplots(figsize=(10, 7))
order = np.argsort(uncertainty)              # draw the most uncertain points last, on top
scatter = ax.scatter(xy_km[order, 0], xy_km[order, 1], c=uncertainty[order], s=6, cmap="viridis")
plt.colorbar(scatter, ax=ax, label="1 - highest posterior probability")
style_map(ax, f"GMM Membership Uncertainty (K = {k_match})")
plt.tight_layout()
finish_chart("17_kent_gmm_uncertainty.png")

# Chart 18: the three DBSCAN tiers, stacked, to show hotspots growing from road sections into town-wide
# clusters as eps and min_samples change
fig, axes = plt.subplots(3, 1, figsize=(11, 16))
for ax, tier in zip(axes, kent_fit["tiers"]):
    row = kent_fit["tier_df"][kent_fit["tier_df"]["tier"] == tier].iloc[0]
    eps_m, min_samples = kent_fit["tiers"][tier]
    draw_clusters(ax, xy_km, kent_fit["tier_models"][tier].labels_,
                  f"{tier} (eps {eps_m:.0f} m, min_samples {min_samples}): {int(row['clusters'])} hotspots holding "
                  f"{row['pct_in_hotspots']:.1f}% of collisions, median radius {row['median_radius_m']:.0f} m, "
                  f"largest radius {row['max_radius_m']:.0f} m")
plt.tight_layout()
finish_chart("18_kent_dbscan_tiers.png")

# ============================================================================
# 8. PART D (1/3): THE OUT-OF-TIME VALIDATION MACHINERY
# ============================================================================
# Real hotspots have no ground-truth labels, and intrinsic scores such as silhouette favour compact shapes
# (Part A), so each method is judged on an out-of-time test that matches the decision a highway authority
# actually makes: 'which places do we investigate with a limited budget?' Every method is given the SAME
# amount of ground to flag, measured as an area, and the test asks how many TEST-year collisions fall
# inside what each method flagged using FIT-year data alone.
#
#   1. A flagged zone is everything within R_FLAG_M (150 m) of a collision in the flagged clusters. The
#      same footprint rule applies to every method, so area is comparable.
#   2. Clusters are flagged strongest first, ranked by fit-year collision density (collisions per km2 of
#      footprint) or by fatal-or-serious (KSI) density.
#   3. Capture = share of ALL test-year collisions inside the zone. Lift compares it with random placement
#      of the same area within the 'collision-affected area' (everything within 150 m of any fit-year collision).
#   4. A 500 m grid ranked by fit-year count is added as the standard practitioner benchmark.
#   5. Areas are measured on an exact budget: whole clusters are flagged in rank order, then the next
#      cluster is flagged only as far as the budget allows, densest cells first.
#   6. A paired bootstrap of the test-year collisions gives 95% intervals, including for the DIFFERENCE
#      between methods, so small gaps are not over-read.
R_CELLS = int(np.ceil(R_FLAG_M / CELL_M))
OFFSETS = np.array([(dx, dy) for dx in range(-R_CELLS, R_CELLS + 1) for dy in range(-R_CELLS, R_CELLS + 1)
                    if (dx * CELL_M) ** 2 + (dy * CELL_M) ** 2 <= R_FLAG_M ** 2])
CENTRE_COL = int(np.where((OFFSETS == 0).all(axis=1))[0][0])    # the (0, 0) offset: a collision's own cell
KEY_STRIDE = 1_000_000                                          # encodes a raster cell (x, y) as one integer
CELL_KM2 = (CELL_M / 1000) ** 2
SQUARE_KM2 = (GRID_M / 1000) ** 2


class HotspotValidator:
    """Out-of-time validation machinery for one fit-year / test-year pair.

    Area is measured on a raster of CELL_M squares. A collision's footprint is the set of cells within
    R_FLAG_M of it; a zone is the union of footprints. Areas, and whether a test-year collision falls inside
    a zone, are both read from the same cells, so they are consistent by construction.
    """

    def __init__(self, xy_fit, ksi_fit, xy_test, ksi_test, seed=RANDOM_STATE):
        self.xy_fit, self.ksi_fit, self.xy_test, self.ksi_test = xy_fit, ksi_fit, xy_test, ksi_test
        self.n_fit, self.n_test = len(xy_fit), len(xy_test)
        self.n_ksi_test = int(ksi_test.sum())

        base_fit = np.floor(xy_fit / CELL_M).astype(np.int64)
        fit_keys = self._to_keys(base_fit[:, None, :] + OFFSETS[None, :, :])    # (collisions, cells per footprint)
        self.all_keys = np.unique(fit_keys)                                     # the collision-affected area
        self.fit_idx = np.searchsorted(self.all_keys, fit_keys)                 # footprint cells as indices
        self.fit_centre_idx = self.fit_idx[:, CENTRE_COL]
        self.area_all_km2 = len(self.all_keys) * CELL_KM2

        test_keys = self._to_keys(np.floor(xy_test / CELL_M).astype(np.int64))
        pos = np.minimum(np.searchsorted(self.all_keys, test_keys), len(self.all_keys) - 1)
        self.test_idx = np.where(self.all_keys[pos] == test_keys, pos, -1)      # -1 = outside the affected area
        self.in_universe = self.test_idx >= 0
        self.ceiling_all = self.in_universe.mean()          # best capture any zone inside the area can reach
        self.ceiling_ksi = self.in_universe[ksi_test].mean()

        # One set of bootstrap resamples is reused for every comparison (a paired bootstrap), so differences
        # between methods are not blurred by independent sampling noise
        rng = np.random.default_rng(seed)
        self.boot_all = rng.integers(0, self.n_test, size=(N_BOOT, self.n_test), dtype=np.int32)
        self.boot_ksi = rng.integers(0, self.n_ksi_test, size=(N_BOOT, self.n_ksi_test), dtype=np.int32)
        self.boot_fit = rng.integers(0, self.n_fit, size=(N_BOOT, self.n_fit), dtype=np.int32)

        # Grid benchmark: 500 m squares ranked by fit-year count. A flagged square costs its FULL area
        # (0.25 km2), including any empty ground inside it
        self.fit_sq, self.test_sq = self._square_id(xy_fit), self._square_id(xy_test)
        self.square_counts = (pd.DataFrame({"sq": self.fit_sq, "n": 1, "ksi": ksi_fit.astype(int)})
                              .groupby("sq").agg(n=("n", "sum"), ksi=("ksi", "sum")).reset_index())

    @staticmethod
    def _to_keys(cells):
        return cells[..., 0] * KEY_STRIDE + cells[..., 1]

    @staticmethod
    def _square_id(xy):
        cells = np.floor(xy / GRID_M).astype(np.int64)
        return cells[:, 0] * KEY_STRIDE + cells[:, 1]

    def captured(self, zone_mask):
        """Boolean array over the test-year collisions: True where the collision lies inside the zone."""
        flags = np.zeros(self.n_test, dtype=bool)
        flags[self.in_universe] = zone_mask[self.test_idx[self.in_universe]]
        return flags

    # ---- cluster-based zones ------------------------------------------------
    def cluster_info(self, labels):
        """Footprint cells, members, size and density of every cluster (noise excluded)."""
        rows, cells, members = [], {}, {}
        for label in np.unique(labels[labels != -1]):
            idx = np.where(labels == label)[0]
            cells[label], members[label] = np.unique(self.fit_idx[idx].ravel()), idx
            area = len(cells[label]) * CELL_KM2
            n_ksi = int(self.ksi_fit[idx].sum())
            rows.append({"cluster": label, "n_fit": len(idx), "area_km2": area,
                         "density": len(idx) / area, "ksi_density": n_ksi / area})
        return pd.DataFrame(rows), cells, members

    @staticmethod
    def flag_order(table, ranking):
        """Clusters strongest first. Density, not size, is used because the question is where collisions
        concentrate per unit of ground an engineer must cover."""
        key = "density" if ranking == "all" else "ksi_density"
        return list(table.sort_values([key, "n_fit"], ascending=[False, False])["cluster"])

    def sweep(self, labels, ranking, info=None):
        """Whole-cluster capture curve (used for the charts; lines join whole-cluster steps)."""
        table, cells, _ = info if info is not None else self.cluster_info(labels)
        mask = np.zeros(len(self.all_keys), dtype=bool)
        rows = [{"area_km2": 0.0, "capture_all": 0.0, "capture_ksi": 0.0}]
        for label in self.flag_order(table, ranking):
            mask[cells[label]] = True
            flags = self.captured(mask)
            rows.append({"area_km2": mask.sum() * CELL_KM2, "capture_all": flags.mean(),
                         "capture_ksi": flags[self.ksi_test].mean()})
        curve = pd.DataFrame(rows)
        curve["area_share"] = curve["area_km2"] / self.area_all_km2
        return curve

    def zone_exact(self, info, ranking, budget_km2):
        """Zone for an exact area budget: whole clusters in rank order, then the next cluster only as far
        as the budget allows, densest cells (most overlapping footprints) first. Returns (mask, area_km2,
        units_flagged, capped); capped means the method ran out of clusters before using the budget."""
        table, cells, members = info
        mask = np.zeros(len(self.all_keys), dtype=bool)
        budget_cells = int(np.floor(budget_km2 / CELL_KM2 + 1e-9))     # integer cell counts avoid float edge cases
        used, units = 0, 0
        for label in self.flag_order(table, ranking):
            new = cells[label][~mask[cells[label]]]
            if used + len(new) <= budget_cells:
                mask[new] = True
                used += len(new)
                units += 1
            else:
                room = budget_cells - used
                if room > 0:
                    depth = np.bincount(self.fit_idx[members[label]].ravel(), minlength=len(self.all_keys))[new]
                    mask[new[np.argsort(-depth, kind="stable")[:room]]] = True
                    used += room
                    units += 1
                return mask, used * CELL_KM2, units, False
        return mask, used * CELL_KM2, units, used < budget_cells

    # ---- grid benchmark ---------------------------------------------------------
    def grid_order(self, ranking):
        keys = ["n", "ksi"] if ranking == "all" else ["ksi", "n"]
        return self.square_counts.sort_values(keys, ascending=False)["sq"].to_numpy()

    def grid_curve(self, ranking):
        order = self.grid_order(ranking)
        per_all = pd.Series(self.test_sq).value_counts().reindex(order, fill_value=0).to_numpy()
        per_ksi = pd.Series(self.test_sq[self.ksi_test]).value_counts().reindex(order, fill_value=0).to_numpy()
        curve = pd.DataFrame({"area_km2": np.arange(len(order) + 1) * SQUARE_KM2,
                              "capture_all": np.r_[0, np.cumsum(per_all)] / self.n_test,
                              "capture_ksi": np.r_[0, np.cumsum(per_ksi)] / max(self.n_ksi_test, 1)})
        curve["area_share"] = curve["area_km2"] / self.area_all_km2
        return curve

    def grid_zone(self, ranking, budget_km2):
        n = int(np.floor(budget_km2 / SQUARE_KM2 + 1e-9))
        return np.isin(self.test_sq, self.grid_order(ranking)[:n]), n * SQUARE_KM2, n, False

    # ---- evaluation ---------------------------------------------------------------
    def evaluate(self, flags, area_km2, units):
        """Capture, lift and severity enrichment for one zone, with 95% bootstrap intervals."""
        area_share = area_km2 / self.area_all_km2
        flags_ksi = flags[self.ksi_test]
        cap_all, cap_ksi = flags.mean(), flags_ksi.mean()
        ci_all = np.percentile(flags[self.boot_all].mean(axis=1), [2.5, 97.5])
        ci_ksi = np.percentile(flags_ksi[self.boot_ksi].mean(axis=1), [2.5, 97.5])
        return {
            "units_flagged": units, "area_km2": area_km2, "area_used_pct": 100 * area_share,
            "capture_all_pct": 100 * cap_all, "capture_all_ci": f"{100 * ci_all[0]:.1f}-{100 * ci_all[1]:.1f}",
            # lift: capture divided by what randomly placed zones of the same area would capture
            "lift": cap_all / (area_share * self.ceiling_all) if area_share > 0 else np.nan,
            "capture_ksi_pct": 100 * cap_ksi, "capture_ksi_ci": f"{100 * ci_ksi[0]:.1f}-{100 * ci_ksi[1]:.1f}",
            # enrichment: KSI share among captured collisions relative to the test-year KSI share overall
            "ksi_enrichment": (flags_ksi.sum() / flags.sum()) / self.ksi_test.mean() if flags.sum() > 0 else np.nan,
        }

    def difference(self, flags_a, flags_b, metric):
        """Paired bootstrap difference in capture (A minus B), in percentage points, with a 95% interval."""
        if metric == "all":
            a, b = flags_a[self.boot_all].mean(axis=1), flags_b[self.boot_all].mean(axis=1)
            point = flags_a.mean() - flags_b.mean()
        else:
            fa, fb = flags_a[self.ksi_test], flags_b[self.ksi_test]
            a, b = fa[self.boot_ksi].mean(axis=1), fb[self.boot_ksi].mean(axis=1)
            point = fa.mean() - fb.mean()
        low, high = np.percentile(a - b, [2.5, 97.5])
        return 100 * point, 100 * low, 100 * high

    def retention(self, zone_mask):
        """Share of collisions inside the zone in the test year divided by the share in the fit year.

        The fit-year share is high by construction (the zone was drawn from those collisions), so the
        ratio measures how much of the apparent concentration persisted: regression to the mean.
        Equivalent to (test count / fit count) divided by the change in county-wide volume."""
        fit_flags, test_flags = zone_mask[self.fit_centre_idx], self.captured(zone_mask)
        point = test_flags.mean() / fit_flags.mean()
        boot = test_flags[self.boot_all].mean(axis=1) / fit_flags[self.boot_fit].mean(axis=1)
        low, high = np.percentile(boot, [2.5, 97.5])
        return {"share_fit_pct": 100 * fit_flags.mean(), "share_test_pct": 100 * test_flags.mean(),
                "retention": point, "ci_low": low, "ci_high": high,
                "n_fit_in_zone": int(fit_flags.sum()), "n_test_in_zone": int(test_flags.sum())}


def build_methods(fit_res):
    """Methods to compare: name, role (stable across year pairs), cluster labels and chart style."""
    k_small, k_big = fit_res["k_values"]
    tiers, labels = fit_res["tiers"], {t: m.labels_ for t, m in fit_res["tier_models"].items()}
    return [
        {"name": f"K-Means K={k_small}", "role": "kmeans_matched", "labels": fit_res["km_labels"][k_small],
         "colour": "tab:orange", "style": "-", "width": 1.8},
        {"name": f"K-Means K={k_big}", "role": "kmeans_large", "labels": fit_res["km_labels"][k_big],
         "colour": "tab:orange", "style": "--", "width": 1.8},
        {"name": f"GMM K={k_small}", "role": "gmm_matched", "labels": fit_res["gmm_labels"][k_small],
         "colour": "tab:green", "style": "-", "width": 1.8},
        {"name": f"GMM K={k_big}", "role": "gmm_large", "labels": fit_res["gmm_labels"][k_big],
         "colour": "tab:green", "style": "--", "width": 1.8},
        {"name": f"DBSCAN Site ({tiers['Site'][0]:.0f} m, {tiers['Site'][1]})", "role": "site",
         "labels": labels["Site"], "colour": "tab:blue", "style": "-", "width": 3.0},
        {"name": f"DBSCAN Strict ({tiers['Strict site'][0]:.0f} m, {tiers['Strict site'][1]})", "role": "strict",
         "labels": labels["Strict site"], "colour": "tab:cyan", "style": "-", "width": 1.8},
        {"name": f"DBSCAN Area ({tiers['Area'][0]:.0f} m, {tiers['Area'][1]})", "role": "area",
         "labels": labels["Area"], "colour": "navy", "style": "-", "width": 1.8},
    ]


def validate_pair(fit_df, test_df, fit_res, verbose=True):
    """Run the full out-of-time validation for one year pair and return every result."""
    v = HotspotValidator(fit_df[["easting", "northing"]].to_numpy(float), fit_df["severity"].isin([1, 2]).to_numpy(),
                         test_df[["easting", "northing"]].to_numpy(float), test_df["severity"].isin([1, 2]).to_numpy())
    methods = build_methods(fit_res)
    infos = {m["name"]: v.cluster_info(m["labels"]) for m in methods}
    if verbose:
        print(f"Collision-affected area (all fit-year collisions, {R_FLAG_M} m footprint): {v.area_all_km2:.1f} km2")
        print(f"{100 * v.ceiling_all:.1f}% of test-year collisions ({100 * v.ceiling_ksi:.1f}% of test-year KSI) lie "
              f"inside it: the ceiling any zone drawn from fit-year data inside that area can reach")

    # Curves for the charts (whole-cluster steps) and the Site zone's total size
    curves = {(m["name"], r): v.sweep(m["labels"], r, infos[m["name"]]) for m in methods for r in ("all", "ksi")}
    for r in ("all", "ksi"):
        curves[("Grid 500 m", r)] = v.grid_curve(r)
    site = next(m for m in methods if m["role"] == "site")
    site_zone_km2 = curves[(site["name"], "all")]["area_km2"].iloc[-1]
    if verbose:
        print(f"The whole DBSCAN Site zone covers {site_zone_km2:.1f} km2, "
              f"{100 * site_zone_km2 / v.area_all_km2:.1f}% of the collision-affected area")

    # Equal-area results at exact budgets
    budgets = {"Site area": site_zone_km2}
    budgets.update({f"{int(100 * s)}%": s * v.area_all_km2 for s in EXTRA_BUDGET_SHARES})
    rows, flag_store = [], {}
    for ranking in ("all", "ksi"):
        for budget_label, budget_km2 in budgets.items():
            entries = []
            for m in methods:
                mask, area_km2, units, capped = v.zone_exact(infos[m["name"]], ranking, budget_km2)
                entries.append((m["name"], m["role"], v.captured(mask), area_km2, units, capped))
            flags, area_km2, units, capped = v.grid_zone(ranking, budget_km2)
            entries.append(("Grid 500 m", "grid", flags, area_km2, units, capped))
            for name, role, flags, area_km2, units, capped in entries:
                flag_store[(ranking, budget_label, role)] = flags
                rows.append({"ranking": ranking, "budget": budget_label, "method": name, "role": role,
                             "capped": capped, **v.evaluate(flags, area_km2, units)})
    results = pd.DataFrame(rows)

    # Paired differences: Site minus each other method. Only like-for-like comparisons are made: the Site zone
    # must be able to use the whole budget, and so must the comparator (a method that ran out of clusters
    # flags less ground, so a difference against it would partly be a difference in area)
    diff_rows = []
    for ranking in ("all", "ksi"):
        for budget_label, budget_km2 in budgets.items():
            site_row = results[(results["ranking"] == ranking) & (results["budget"] == budget_label)
                               & (results["role"] == "site")].iloc[0]
            if site_row["area_km2"] < 0.98 * budget_km2:
                continue                                   # Site is capped here, so the comparison is not like-for-like
            for other in results[(results["ranking"] == ranking) & (results["budget"] == budget_label)
                                 & (results["role"] != "site")].itertuples():
                if other.capped:
                    continue
                point, low, high = v.difference(flag_store[(ranking, budget_label, "site")],
                                                flag_store[(ranking, budget_label, other.role)], ranking)
                diff_rows.append({"ranking": ranking, "budget": budget_label, "comparison": f"Site - {other.method}",
                                  "role": other.role, "diff_pp": point, "ci": f"{low:+.1f} to {high:+.1f}",
                                  "excludes_zero": (low > 0) or (high < 0), "other_area_used_pct": other.area_used_pct})
    diffs = pd.DataFrame(diff_rows)

    # Persistence of the Site hotspots (union of footprints, so overlapping footprints are not double-counted)
    table, cells, members = infos[site["name"]]
    union = np.zeros(len(v.all_keys), dtype=bool)
    per_hotspot = []
    for label in table["cluster"]:
        mask = np.zeros(len(v.all_keys), dtype=bool)
        mask[cells[label]] = True
        union |= mask
        per_hotspot.append({"hotspot": label, "grid_ref": grid_ref(*v.xy_fit[members[label]].mean(axis=0)),
                            "n_fit": int(mask[v.fit_centre_idx].sum()), "n_test": int(v.captured(mask).sum()),
                            "ksi_fit": int((mask[v.fit_centre_idx] & v.ksi_fit).sum()),
                            "ksi_test": int((v.captured(mask) & v.ksi_test).sum())})
    persistence = {"union": v.retention(union), "per_hotspot": pd.DataFrame(per_hotspot),
                   "volume_ratio": v.n_test / v.n_fit, "union_flags_test": v.captured(union)}
    for tier, role in [("Strict site", "strict"), ("Area", "area")]:
        m = next(x for x in methods if x["role"] == role)
        t, c, _ = infos[m["name"]]
        mask = np.zeros(len(v.all_keys), dtype=bool)
        for label in t["cluster"]:
            mask[c[label]] = True
        persistence[tier] = v.retention(mask)
    return {"validator": v, "methods": methods, "infos": infos, "curves": curves, "results": results,
            "diffs": diffs, "persistence": persistence, "site_zone_km2": site_zone_km2, "budgets": budgets}


def headline(val, fit_res, fit_year, test_year, fit_df, test_df):
    """The numbers that summarise one year pair (used in the final replication table)."""
    r, d, v, p = val["results"], val["diffs"], val["validator"], val["persistence"]

    def result(ranking, role, column):
        return r[(r["ranking"] == ranking) & (r["budget"] == "Site area") & (r["role"] == role)].iloc[0][column]

    def cell(ranking, role, metric):
        return f"{result(ranking, role, f'capture_{metric}_pct'):.1f} ({result(ranking, role, f'capture_{metric}_ci')})"

    def diff(ranking, role):
        rows = d[(d["ranking"] == ranking) & (d["budget"] == "Site area") & (d["role"] == role)]
        if rows.empty:
            return "n/a"
        row = rows.iloc[0]
        return f"{row['diff_pp']:+.1f} ({row['ci']})" + (" *" if row["excludes_zero"] else "")

    eps_m, ms = fit_res["tiers"]["Site"]
    tier_row = fit_res["tier_df"][fit_res["tier_df"]["tier"] == "Site"].iloc[0]
    out = {
        "Site tier (eps m, min_samples)": f"{eps_m:.0f}, {ms}",
        "Site hotspots": int(tier_row["clusters"]),
        "Fit-year collisions inside hotspots (%)": f"{tier_row['pct_in_hotspots']:.1f}",
        "Site zone: area (km2) / share of affected area (%)": f"{val['site_zone_km2']:.1f} / "
                                                               f"{100 * val['site_zone_km2'] / v.area_all_km2:.1f}",
        "Test collisions within 150 m of a fit-year collision (%)": f"{100 * v.ceiling_all:.1f}",
        "KSI share of collisions, fit year / test year (%)": f"{100 * fit_df['severity'].isin([1, 2]).mean():.1f} / "
                                                             f"{100 * test_df['severity'].isin([1, 2]).mean():.1f}",
        "ALL collisions captured at the Site area (%), ranked by density": "",
        "  DBSCAN Site": cell("all", "site", "all"), "  Grid 500 m": cell("all", "grid", "all"),
        "  GMM, matched K": cell("all", "gmm_matched", "all"), "  GMM, large K": cell("all", "gmm_large", "all"),
        "  K-Means, matched K": cell("all", "kmeans_matched", "all"),
        "  K-Means, large K": cell("all", "kmeans_large", "all"),
        "  Difference, Site minus grid (pp)": diff("all", "grid"),
        "  Difference, Site minus GMM large K (pp)": diff("all", "gmm_large"),
        "  Difference, Site minus K-Means large K (pp)": diff("all", "kmeans_large"),
        "KSI collisions captured at the Site area (%), ranked by KSI density": "",
        "  DBSCAN Site ": cell("ksi", "site", "ksi"), "  Grid 500 m ": cell("ksi", "grid", "ksi"),
        "  Difference, Site minus grid (pp) ": diff("ksi", "grid"),
        "  Difference, Site minus GMM large K (pp) ": diff("ksi", "gmm_large"),
        "Persistence of Site hotspots": "",
        "  Share of collisions inside the zone, fit year / test year (%)":
            f"{p['union']['share_fit_pct']:.1f} / {p['union']['share_test_pct']:.1f}",
        "  Retention (test share / fit share)": f"{p['union']['retention']:.2f} "
                                                 f"({p['union']['ci_low']:.2f}-{p['union']['ci_high']:.2f})",
    }
    return out


def print_validation_tables(val):
    """Console report for the primary pair."""
    r, d = val["results"], val["diffs"]
    columns_all = ["budget", "method", "units_flagged", "area_used_pct", "capped", "capture_all_pct",
                   "capture_all_ci", "lift", "ksi_enrichment"]
    columns_ksi = ["budget", "method", "units_flagged", "area_used_pct", "capped", "capture_ksi_pct",
                   "capture_ksi_ci", "ksi_enrichment", "capture_all_pct"]
    print("\nRanked by collision density: share of ALL test-year collisions captured. area_used_pct is a share of the "
          "collision-affected area; capped = the method ran out of clusters before using the budget; ci = 95% "
          "bootstrap interval; lift = capture relative to random placement of the same area; ksi_enrichment = KSI "
          "share among captured collisions relative to the test year's overall KSI share:")
    print(r[r["ranking"] == "all"][columns_all].round(2).to_string(index=False))
    print("\nRanked by KSI density: share of test-year fatal/serious collisions captured:")
    print(r[r["ranking"] == "ksi"][columns_ksi].round(2).to_string(index=False))
    print("\nPaired bootstrap differences in capture (percentage points; excludes_zero = the 95% interval excludes "
          "zero). Only like-for-like comparisons are made: budgets the Site zone and the comparator can both fully use:")
    print(d[["ranking", "budget", "comparison", "diff_pp", "ci", "excludes_zero", "other_area_used_pct"]]
          .round(2).to_string(index=False))


# ============================================================================
# 9. PART D (2/3): PRIMARY VALIDATION, FIT YEAR -> TEST YEAR
# ============================================================================
print(f"\nPART D: Out-of-time validation, {FIT_YEAR} hotspots against {TEST_YEAR} collisions")
validation = validate_pair(kent_fit_df, kent_test_df, kent_fit, verbose=True)
print_validation_tables(validation)
validation["results"].to_csv(os.path.join(DATA_DIR, "kent_budget_results.csv"), index=False)
validation["diffs"].to_csv(os.path.join(DATA_DIR, "kent_paired_differences.csv"), index=False)

v = validation["validator"]
pers = validation["persistence"]
print(f"\nPersistence of the Site hotspots (county volume changed by a factor of {pers['volume_ratio']:.3f}):")
for tier in ("union", "Strict site", "Area"):
    p = pers[tier]
    label = "Site" if tier == "union" else tier
    print(f"  {label}: {p['share_fit_pct']:.1f}% of {FIT_YEAR} collisions inside the zone "
          f"({p['n_fit_in_zone']:,}), {p['share_test_pct']:.1f}% of {TEST_YEAR} ({p['n_test_in_zone']:,}); "
          f"retention {p['retention']:.2f} (95% interval {p['ci_low']:.2f}-{p['ci_high']:.2f})")
per_hotspot = pers["per_hotspot"].copy()
per_hotspot["expected_test_if_volume_only"] = per_hotspot["n_fit"] * pers["volume_ratio"]
kept = (per_hotspot["n_test"] >= per_hotspot["expected_test_if_volume_only"]).mean()
print(f"  {100 * kept:.0f}% of Site hotspots had at least the {TEST_YEAR} count that county volume alone would "
      f"predict; {int((per_hotspot['n_test'] == 0).sum())} had no {TEST_YEAR} collisions at all. "
      f"(Per-hotspot counts can overlap between neighbouring hotspots, so the zone-level figures above are the "
      f"ones to quote.)")
print("Largest Site hotspots by fit-year count:")
print(per_hotspot.sort_values("n_fit", ascending=False).head(10).round(1).to_string(index=False))
per_hotspot.to_csv(os.path.join(DATA_DIR, "kent_site_hotspot_persistence.csv"), index=False)

# ============================================================================
# 10. PART D (3/3): VALIDATION CHARTS
# ============================================================================
def draw_capture_chart(val, ranking, metric, ylabel, title, filename, ceiling):
    """Capture-against-area curves for every method, plus the grid benchmark and random placement."""
    validator, curves = val["validator"], val["curves"]
    site_share = val["site_zone_km2"] / validator.area_all_km2
    fig, ax = plt.subplots(figsize=(12, 7))
    for m in val["methods"]:
        curve = curves[(m["name"], ranking)]
        ax.plot(100 * curve["area_share"], 100 * curve[metric], color=m["colour"], linestyle=m["style"],
                linewidth=m["width"], label=m["name"])
    grid = curves[("Grid 500 m", ranking)]
    ax.plot(100 * grid["area_share"], 100 * grid[metric], color="dimgrey", linewidth=2.2, label="Grid 500 m ranked by count")
    ax.plot([0, PLOT_MAX_AREA_PCT], [0, ceiling * PLOT_MAX_AREA_PCT], color="black", linestyle=":", linewidth=1.8,
            label="Random placement (same area)")
    ax.axvline(100 * site_share, color="tab:blue", linestyle=":", alpha=0.6)
    ax.annotate("full DBSCAN Site zone", (100 * site_share, 0), textcoords="offset points", xytext=(4, 8),
                color="tab:blue", fontsize=9)
    ax.set_xlim(0, PLOT_MAX_AREA_PCT)
    ax.set_ylim(0, 1.1 * max(100 * c[metric][c["area_share"] <= PLOT_MAX_AREA_PCT / 100].max() for c in curves.values()))
    ax.set_xlabel(f"Area flagged (% of the {validator.area_all_km2:.0f} km2 collision-affected area)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    plt.tight_layout()
    finish_chart(filename)


# Chart 19 and 20: capture against area, for all collisions and for fatal/serious collisions
draw_capture_chart(validation, "all", "capture_all", f"% of all {TEST_YEAR} collisions captured",
                   f"Out-of-Time Capture: {FIT_YEAR} Hotspots Against {TEST_YEAR} Collisions (ranked by density)",
                   "19_capture_all_collisions.png", v.ceiling_all)
draw_capture_chart(validation, "ksi", "capture_ksi", f"% of {TEST_YEAR} fatal/serious collisions captured",
                   "Out-of-Time Capture of Fatal and Serious Collisions (ranked by KSI density)",
                   "20_capture_ksi.png", v.ceiling_ksi)

# Chart 21: how stable is the DBSCAN result across eps? One curve per eps, one panel per min_samples
fig, axes = plt.subplots(1, len(STABILITY_MIN_SAMPLES), figsize=(16, 6), sharey=True)
colours = sns.color_palette("viridis", len(EPS_STABILITY_M))
for ax, min_samples in zip(np.atleast_1d(axes), STABILITY_MIN_SAMPLES):
    for colour, eps_m in zip(colours, EPS_STABILITY_M):
        labels = DBSCAN(eps=eps_m, min_samples=min_samples).fit_predict(v.xy_fit)
        if (labels != -1).sum() == 0:
            continue
        curve = v.sweep(labels, "all")
        ax.plot(100 * curve["area_share"], 100 * curve["capture_all"], color=colour, linewidth=1.8, label=f"eps {eps_m} m")
    grid = validation["curves"][("Grid 500 m", "all")]
    ax.plot(100 * grid["area_share"], 100 * grid["capture_all"], color="dimgrey", linewidth=2.2, label="Grid 500 m")
    ax.plot([0, PLOT_MAX_AREA_PCT], [0, v.ceiling_all * PLOT_MAX_AREA_PCT], color="black", linestyle=":",
            label="Random placement")
    ax.set_xlim(0, PLOT_MAX_AREA_PCT)
    ax.set_ylim(0, 40)
    ax.set_title(f"DBSCAN, min_samples = {min_samples}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Area flagged (% of collision-affected area)")
    ax.legend(fontsize=8)
np.atleast_1d(axes)[0].set_ylabel(f"% of all {TEST_YEAR} collisions captured")
fig.suptitle("DBSCAN Out-of-Time Capture Across the eps Range", fontsize=14, fontweight="bold")
plt.tight_layout()
finish_chart("21_dbscan_eps_stability.png")

# Chart 22: persistence of each Site hotspot (fit-year against test-year counts in the same footprint)
rng_plot = np.random.default_rng(RANDOM_STATE)
fig, ax = plt.subplots(figsize=(8.5, 8))
jitter = rng_plot.uniform(-0.12, 0.12, size=(len(per_hotspot), 2))    # counts are whole numbers; jitter avoids overplotting
ax.scatter(per_hotspot["n_fit"] + jitter[:, 0], per_hotspot["n_test"] + jitter[:, 1], alpha=0.7, color="tab:blue")
limit = max(per_hotspot["n_fit"].max(), per_hotspot["n_test"].max()) * 1.05
ax.plot([0, limit], [0, limit], color="grey", linestyle="--", label="Same count in both years")
ax.plot([0, limit], [0, limit * pers["volume_ratio"]], color="black", linestyle=":",
        label=f"Scaled to county volume (x{pers['volume_ratio']:.2f})")
ax.set_xlim(0, limit)
ax.set_ylim(0, limit)
ax.set_xlabel(f"Collisions in the hotspot footprint, {FIT_YEAR}", fontsize=12)
ax.set_ylabel(f"Collisions in the same footprint, {TEST_YEAR}", fontsize=12)
ax.set_title(f"Hotspot Persistence: retention {pers['union']['retention']:.2f} "
             f"({pers['union']['ci_low']:.2f}-{pers['union']['ci_high']:.2f})", fontsize=13, fontweight="bold")
ax.legend()
plt.tight_layout()
finish_chart("22_hotspot_persistence.png")

# Chart 23: test-year collisions inside and outside the whole Site zone
site_flags = pers["union_flags_test"]
fig, ax = plt.subplots(figsize=(11, 7))
ax.scatter(v.xy_test[~site_flags, 0] / 1000, v.xy_test[~site_flags, 1] / 1000, s=3, color="lightgrey",
           label=f"Outside the Site zone ({(~site_flags).sum():,})")
ax.scatter(v.xy_test[site_flags, 0] / 1000, v.xy_test[site_flags, 1] / 1000, s=14, color="tab:red",
           label=f"Inside the Site zone ({site_flags.sum():,})")
ax.set_aspect("equal")
ax.set_xlabel("Easting (km)")
ax.set_ylabel("Northing (km)")
ax.set_title(f"{TEST_YEAR} Collisions Against the {FIT_YEAR} DBSCAN Site Zone "
             f"({100 * site_flags.mean():.1f}% captured in {validation['site_zone_km2']:.1f} km2)",
             fontsize=12, fontweight="bold")
ax.legend(markerscale=2)
plt.tight_layout()
finish_chart("23_site_zone_capture_map.png")

# ============================================================================
# 11. REPLICATION ON EARLIER YEAR PAIRS
# ============================================================================
# The whole procedure (Site tier re-selected by the same size guard, K matched to the new hotspot count,
# equal-area test, paired bootstrap) is repeated on an earlier year pair. If the result held only for
# one pair it could be an accident of that year, particularly given the change in severity mix seen
# between the primary years. Only the headline table is reported for replications
headlines = {f"{FIT_YEAR} -> {TEST_YEAR} (primary)": headline(validation, kent_fit, FIT_YEAR, TEST_YEAR,
                                                               kent_fit_df, kent_test_df)}
for rep_fit, rep_test in REPLICATION_PAIRS:
    print(f"\nREPLICATION: {rep_fit} hotspots against {rep_test} collisions")
    try:
        rep_national = {year: (national[year] if year in national else load_year(year)) for year in (rep_fit, rep_test)}
    except SystemExit as exc:
        print(f"  Skipping this replication: {exc}")
        continue
    rep_fit_df, rep_test_df = kent_frames(rep_national[rep_fit], rep_national[rep_test], rep_fit, rep_test)
    rep_models = fit_kent_models(rep_fit_df, site_override=None, with_bic=False)    # Site tier re-selected by the guard
    rep_validation = validate_pair(rep_fit_df, rep_test_df, rep_models, verbose=True)
    headlines[f"{rep_fit} -> {rep_test} (replication)"] = headline(rep_validation, rep_models, rep_fit, rep_test,
                                                                    rep_fit_df, rep_test_df)

# ============================================================================
# 12. HEADLINE SUMMARY
# ============================================================================
summary = pd.DataFrame(headlines)
summary.index.name = "metric"
print("\n" + "=" * 100)
print("HEADLINE RESULTS (capture figures are % of test-year collisions, with 95% bootstrap intervals; "
      "pp = percentage points; * = the interval for the difference excludes zero)")
print("=" * 100)
print(summary.to_string())
summary.to_csv(os.path.join(DATA_DIR, "headline_summary.csv"))
print("\nDone. Charts 01-23 are saved in the working directory and tables in ./data.")
