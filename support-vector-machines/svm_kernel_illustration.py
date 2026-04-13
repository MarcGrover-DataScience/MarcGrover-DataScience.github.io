# Support Vector Machine — Kernel Trick Illustration
#
# Purpose:
#   A self-contained illustration of the kernel trick using synthetically generated concentric circular data (make_circles).
#   Demonstrates that a linear SVM cannot separate classes that are not linearly separable in their original feature space,
#   and that an RBF kernel SVM resolves this by implicitly mapping observations into a higher-dimensional space.
#
#   This script is a companion to svm_breast_cancer.py and is intended to be read first.
#   It establishes the conceptual foundation for kernel selection before the main analysis begins.
#
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

from sklearn.datasets import make_circles
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# =============================================================================
# 0. GLOBAL PLOT SETTINGS
# =============================================================================

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

COLOUR_CLASS_0  = "red"   # red  — outer ring
COLOUR_CLASS_1  = "cornflowerblue"   # blue — inner ring
COLOUR_BOUNDARY = "darkgrey"   # dark grey boundary line
COLOUR_BAR_LIN  = "mediumseagreen"   # bar chart — linear kernel
COLOUR_BAR_RBF  = "seagreen"   # bar chart — RBF kernel

RANDOM_STATE    = 42
FIGURE_DPI      = 150

# =============================================================================
# 1. DATA GENERATION
# =============================================================================

print("\nSECTION 1: DATA GENERATION")

X, y = make_circles(
    n_samples   = 500,
    noise       = 0.12,
    factor      = 0.45,
    random_state= RANDOM_STATE
)

# Wrap into a DataFrame for validation and EDA
df = pd.DataFrame(X, columns=["Feature_1", "Feature_2"])
df["Class"] = y
df["Class_Label"] = df["Class"].map({0: "Outer (Class 0)", 1: "Inner (Class 1)"})

# =============================================================================
# 2. DATA VALIDATION
# =============================================================================

print("\n--- Dataset Dimensions ---")
print(f"Observations : {df.shape[0]}")
print(f"Features     : {X.shape[1]}  (Feature_1, Feature_2)")

print("\n--- Class Distribution ---")
class_counts = df["Class_Label"].value_counts().sort_index()
class_pct    = (class_counts / len(df) * 100).round(1)
dist_summary = pd.DataFrame({"Count": class_counts, "Percentage (%)": class_pct})
print(dist_summary.to_string())

print("\n--- Descriptive Statistics (Features) ---")
print(df[["Feature_1", "Feature_2"]].describe().round(4).to_string())

print("\n--- Linear Separability Check ---")
print(
    "The two classes form concentric rings in 2D feature space.\n"
    "No linear boundary can correctly separate them — this is confirmed\n"
    "visually in the decision boundary plots and quantitatively by the\n"
    "linear SVM accuracy reported in Section 4."
)

# =============================================================================
# 3. PLOT 01 — CLASS DISTRIBUTION (SCATTER)
# =============================================================================

fig, ax = plt.subplots(figsize=(7, 6))

palette = {"Outer (Class 0)": COLOUR_CLASS_0, "Inner (Class 1)": COLOUR_CLASS_1}

sns.scatterplot(
    data      = df,
    x         = "Feature_1",
    y         = "Feature_2",
    hue       = "Class_Label",
    palette   = palette,
    alpha     = 0.75,
    s         = 45,
    edgecolor = "white",
    linewidth = 0.4,
    ax        = ax
)

ax.set_title("Generated Data — Concentric Circles", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Feature 1", fontsize=11)
ax.set_ylabel("Feature 2", fontsize=11)
ax.legend(title="Class", title_fontsize=10, fontsize=10, loc="upper right")

plt.tight_layout()
plt.savefig("plot_01_data_distribution.png", dpi=FIGURE_DPI, bbox_inches="tight")
plt.show()

# =============================================================================
# 4. TRAIN / TEST SPLIT AND SCALING
# =============================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size    = 0.20,
    stratify     = y,
    random_state = RANDOM_STATE
)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

print("\n--- Train / Test Split ---")
print(f"Training observations : {X_train.shape[0]}")
print(f"Test observations     : {X_test.shape[0]}")

# =============================================================================
# 5. MODEL FITTING — LINEAR AND RBF KERNELS
# =============================================================================

print("\nSECTION 2: MODEL FITTING")

svm_linear = SVC(kernel="linear", C=1.0, random_state=RANDOM_STATE)
svm_linear.fit(X_train, y_train)
y_pred_linear  = svm_linear.predict(X_test)
acc_linear     = accuracy_score(y_test, y_pred_linear)

svm_rbf = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=RANDOM_STATE)
svm_rbf.fit(X_train, y_train)
y_pred_rbf = svm_rbf.predict(X_test)
acc_rbf    = accuracy_score(y_test, y_pred_rbf)

print(f"\nLinear SVM — Test Accuracy : {acc_linear:.4f}  ({acc_linear * 100:.2f}%)")
print(f"RBF SVM    — Test Accuracy : {acc_rbf:.4f}  ({acc_rbf * 100:.2f}%)")

print("\n--- Linear SVM Classification Report ---")
print(classification_report(y_test, y_pred_linear, target_names=["Outer (0)", "Inner (1)"]))

print("--- RBF SVM Classification Report ---")
print(classification_report(y_test, y_pred_rbf, target_names=["Outer (0)", "Inner (1)"]))

print(f"\nLinear SVM support vectors : {svm_linear.n_support_}")
print(f"RBF SVM    support vectors : {svm_rbf.n_support_}")

# =============================================================================
# 6. HELPER — DECISION BOUNDARY PLOT FUNCTION
# =============================================================================

def plot_decision_boundary(model, X_scaled, y_true, scaler_obj,
                           title, filename, acc):
    """
    Plots the SVM decision boundary and margin on the original
    (unscaled) feature space so axis labels remain interpretable.
    Observations are colour-coded by true class label.
    """

    # Inverse-transform scaled data back to original space for plotting
    X_orig = scaler_obj.inverse_transform(X_scaled)

    # Build mesh in original space
    x_min, x_max = X_orig[:, 0].min() - 0.3, X_orig[:, 0].max() + 0.3
    y_min, y_max = X_orig[:, 1].min() - 0.3, X_orig[:, 1].max() + 0.3

    h  = 0.01
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h)
    )

    # Scale mesh points before prediction
    mesh_scaled = scaler_obj.transform(np.c_[xx.ravel(), yy.ravel()])
    Z = model.predict(mesh_scaled).reshape(xx.shape)

    # ---- Build plot ----
    fig, ax = plt.subplots(figsize=(7, 6))

    # Filled region background (light, desaturated)
    cmap_bg = mcolors.ListedColormap(["#F5C6C6", "#C6D9F0"])
    ax.contourf(xx, yy, Z, alpha=0.35, cmap=cmap_bg)

    # Decision boundary line
    ax.contour(xx, yy, Z, levels=[0.5], colors=COLOUR_BOUNDARY,
               linewidths=1.8, linestyles="--")

    # Scatter observations coloured by true label
    colours = np.where(y_true == 0, COLOUR_CLASS_0, COLOUR_CLASS_1)

    ax.scatter(
        X_orig[:, 0], X_orig[:, 1],
        c         = colours,
        edgecolors= "white",
        linewidths= 0.4,
        s         = 45,
        alpha     = 0.80,
        zorder    = 3
    )

    # Legend proxy
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOUR_CLASS_0,
               markersize=9, label="Outer (Class 0)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOUR_CLASS_1,
               markersize=9, label="Inner (Class 1)"),
        Line2D([0], [0], color=COLOUR_BOUNDARY, linewidth=1.8,
               linestyle="--", label="Decision Boundary"),
    ]
    ax.legend(handles=legend_elements, fontsize=10, loc="upper right")

    ax.set_title(f"{title}\nTest Accuracy: {acc * 100:.0f}%",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Feature 1", fontsize=11)
    ax.set_ylabel("Feature 2", fontsize=11)

    plt.tight_layout()
    plt.savefig(filename, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.show()


# =============================================================================
# 7. PLOT 02 — LINEAR SVM DECISION BOUNDARY
# =============================================================================

plot_decision_boundary(
    model      = svm_linear,
    X_scaled   = X_train,
    y_true     = y_train,
    scaler_obj = scaler,
    title      = "Linear SVM — Decision Boundary (Training Data)",
    filename   = "plot_02_linear_svm_boundary.png",
    acc        = acc_linear
)

# =============================================================================
# 8. PLOT 03 — RBF KERNEL SVM DECISION BOUNDARY
# =============================================================================

plot_decision_boundary(
    model      = svm_rbf,
    X_scaled   = X_train,
    y_true     = y_train,
    scaler_obj = scaler,
    title      = "RBF Kernel SVM — Decision Boundary (Training Data)",
    filename   = "plot_03_rbf_svm_boundary.png",
    acc        = acc_rbf
)

# =============================================================================
# 9. PLOT 04 — KERNEL ACCURACY COMPARISON (BAR CHART)
# =============================================================================

accuracy_df = pd.DataFrame({
    "Kernel"  : ["Linear SVM", "RBF Kernel SVM"],
    "Accuracy": [acc_linear,    acc_rbf]
})

fig, ax = plt.subplots(figsize=(7, 5))

bar_colours = [COLOUR_BAR_LIN, COLOUR_BAR_RBF]

sns.barplot(
    data    = accuracy_df,
    x       = "Kernel",
    y       = "Accuracy",
    palette = bar_colours,
    errorbar= None,
    ax      = ax,
    hue     = "Kernel"
)

# Annotate bars with accuracy values
for idx, row in accuracy_df.iterrows():
    ax.text(
        idx, row["Accuracy"] + 0.005,
        f"{row['Accuracy'] * 100:.2f}%",
        ha="center", va="bottom", fontsize=12, fontweight="bold"
    )

ax.set_ylim(0, 1.10)
ax.set_title("Test Accuracy by Kernel Type — make_circles Data",
             fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Kernel", fontsize=11)
ax.set_ylabel("Test Accuracy", fontsize=11)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

plt.tight_layout()
plt.savefig("plot_04_kernel_accuracy_comparison.png", dpi=FIGURE_DPI, bbox_inches="tight")
plt.show()

# =============================================================================
# 10. SUMMARY
# =============================================================================

print("\nSUMMARY")
print(f"\nDataset           : make_circles  (n=500, noise=0.12, factor=0.45)")
print(f"Train / Test split: 80% / 20%  ({X_train.shape[0]} / {X_test.shape[0]} observations)")
print(f"\nLinear SVM")
print(f"  Test Accuracy   : {acc_linear * 100:.2f}%")
print(f"  Support Vectors : {svm_linear.n_support_}  (class 0, class 1)")
print(f"\nRBF Kernel SVM")
print(f"  Test Accuracy   : {acc_rbf * 100:.2f}%")
print(f"  Support Vectors : {svm_rbf.n_support_}  (class 0, class 1)")
print(f"\nAccuracy improvement (RBF over Linear) : "
      f"{(acc_rbf - acc_linear) * 100:.2f} percentage points")