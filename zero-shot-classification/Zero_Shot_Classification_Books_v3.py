# ============================================================
# Zero-Shot Book Genre Classification
# Main Analysis Pipeline — Proof of Concept
# ============================================================
# Dataset:   Goodreads Best Books Ever (books_1.Best_Books_Ever.csv)
#            Source: kaggle.com/datasets/arnabchaki/goodreads-best-books-ever
#
# Models:    facebook/bart-large-mnli  (primary)
#            roberta-large-mnli        (comparison)
#
# Note: Scoped as a CPU-compatible Proof of Concept.
#       Inference on SAMPLE_SIZE records takes approximately 25–35minutes on a mid-range CPU without GPU support.
#       When SAMPLE_SIZE = 200
#       Full inventory classification at production scale would require GPU compute.
# ============================================================

import ast
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from transformers import pipeline

warnings.filterwarnings('ignore')

# Display settings
pd.set_option('display.width', 320)
np.set_printoptions(linewidth=320)
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 20)
pd.set_option('display.min_rows', 10)
pd.set_option('display.precision', 3)
pd.set_option('display.max_colwidth', 60)

sns.set_theme(style='whitegrid', font_scale=1.1)


# ============================================================
# CONFIGURATION
# ============================================================

SAMPLE_SIZE = 200                           # PoC sample size — CPU constraint
RANDOM_STATE = 42                           # Fixed seed — reproducible random sampling

DATA_FILE   = 'books_1.Best_Books_Ever.csv'

# TARGET_LABELS = [
#     'Science Fiction, Fantasy and Adventure', 'Romance',
#     'Mystery', 'Historical', 'Biography'
# ]

TARGET_LABELS = [
    'Science Fiction and Fantasy', 'Romance',
    'Mystery', 'Historical', 'Biography'
]

FICTION_NONFICTION_LABELS = ['Fiction', 'Non-Fiction']

MODEL_BART    = 'facebook/bart-large-mnli'
MODEL_ROBERTA = 'roberta-large-mnli'

# Plot colours
COL_BART    = '#2e86ab'   # Blue  — bart model
COL_ROBERTA = '#e84855'   # Red   — roberta model
COL_NEUTRAL = '#4a4e69'   # Grey  — single-model elements


# ============================================================
# GENRE MAPPING
# ============================================================

# Maps raw Goodreads genre tags (lowercase) to one of the 5 target categories.
# Conservative: only tags that unambiguously belong to a single target category.
# Tags that could plausibly belong to more than one category are excluded;
# books with multiple conflicting category matches are excluded from evaluation.
GENRE_MAP = {

    # Science Fiction and Fantasy — merged into a single category.
    'science fiction':       'Science Fiction and Fantasy',
    'sci-fi':                'Science Fiction and Fantasy',
    'sci fi':                'Science Fiction and Fantasy',
    'space opera':           'Science Fiction and Fantasy',
    'cyberpunk':             'Science Fiction and Fantasy',
    'steampunk':             'Science Fiction and Fantasy',
    'adventure':             'Science Fiction and Fantasy',
    'fantasy':               'Science Fiction and Fantasy',
    'fairy tale':            'Science Fiction and Fantasy',
    'fairy tales':           'Science Fiction and Fantasy',
    'mythology':             'Science Fiction and Fantasy',
    'urban fantasy':         'Science Fiction and Fantasy',
    'epic fantasy':          'Science Fiction and Fantasy',
    'high fantasy':          'Science Fiction and Fantasy',
    'dark fantasy':          'Science Fiction and Fantasy',

    # Romance
    'romance':               'Romance',
    'chick lit':             'Romance',
    'paranormal romance':    'Romance',
    'contemporary romance':  'Romance',

    # Mystery
    'mystery':               'Mystery',
    'thriller':              'Mystery',
    'crime':                 'Mystery',
    'detective':             'Mystery',
    'suspense':              'Mystery',
    'noir':                  'Mystery',
    'cozy mystery':          'Mystery',
    'murder mystery':        'Mystery',

    # Historical
    'historical fiction':    'Historical',
    'historical':            'Historical',

    # Biography
    'biography':             'Biography',
    'autobiography':         'Biography',
    'memoir':                'Biography',
    'biographies':           'Biography',
    'true story':            'Biography',
}

# Genre tags used to determine Fiction / Non-Fiction ground truth for the negative finding test (Section 6).
FICTION_INDICATORS = {
    'fiction', 'novel', 'fantasy', 'science fiction', 'sci-fi',
    'romance', 'mystery', 'thriller', 'crime', 'adventure',
    'historical fiction', 'short stories', 'graphic novels', 'comics',
    'literary fiction', 'contemporary', 'horror', 'paranormal',
    'urban fantasy', 'dystopia', "children's", 'juvenile fiction',
}

NONFICTION_INDICATORS = {
    'nonfiction', 'non-fiction', 'biography', 'autobiography', 'memoir',
    'true story', 'self-help', 'science', 'history', 'politics',
    'philosophy', 'essays', 'true crime', 'religion', 'travel',
    'cooking', 'health', 'business', 'economics', 'psychology',
    'popular science', 'journalism',
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def parse_genres(genre_string):
    """Parse a string-encoded genre list (as stored in the dataset) into a Python list."""
    if pd.isna(genre_string):
        return []
    try:
        parsed = ast.literal_eval(genre_string)
        if isinstance(parsed, list):
            return [str(g).strip() for g in parsed]
    except (ValueError, SyntaxError):
        pass
    return []


def map_to_category(genre_string):
    """
    Map a book's genre tags to one of the 7 target categories.

    Returns the matched category if exactly one target category is found
    across all genre tags; otherwise returns None.

    None indicates either:
      - No match: none of the genre tags map to a target category.
      - Ambiguous: genre tags match more than one target category.
    Both cases are excluded from quantitative evaluation.
    """
    genres  = parse_genres(genre_string)
    matched = set()
    for genre in genres:
        key = genre.lower().strip()
        if key in GENRE_MAP:
            matched.add(GENRE_MAP[key])
    return matched.pop() if len(matched) == 1 else None


def map_to_fiction_nonfiction(genre_string):
    """
    Map genre tags to 'Fiction' or 'Non-Fiction' for the negative finding test.
    Returns None if ambiguous or no usable signal is found.
    """
    tags          = [g.lower().strip() for g in parse_genres(genre_string)]
    is_fiction    = any(t in FICTION_INDICATORS    for t in tags)
    is_nonfiction = any(t in NONFICTION_INDICATORS for t in tags)
    if is_fiction and not is_nonfiction:
        return 'Fiction'
    if is_nonfiction and not is_fiction:
        return 'Non-Fiction'
    return None


def run_classifier(descriptions, clf, labels, model_name):
    """
    Run zero-shot classification on all descriptions.
    Prints progress every 10 records and total elapsed time.

    Returns a DataFrame (aligned to descriptions.index) containing:
      predicted_label  — category with the highest confidence score
      top1_score       — confidence score of the predicted category
      top2_score       — confidence score of the second-ranked category
      conf_margin      — top1_score minus top2_score (classification certainty)
    """
    print(f'\n  Running: {model_name}')
    print(f'  {len(descriptions)} records to classify — estimated 15–30 min on CPU...')

    records   = []
    t_start   = time.time()
    desc_list = descriptions.tolist()

    for i, text in enumerate(desc_list, 1):
        result = clf(str(text), labels)
        records.append({
            'predicted_label': result['labels'][0],
            'top1_score':      result['scores'][0],
            'top2_score':      result['scores'][1],
            'conf_margin':     result['scores'][0] - result['scores'][1],
        })
        if i % 10 == 0 or i == len(desc_list):
            elapsed = time.time() - t_start
            print(f'  {i}/{len(desc_list)} classified  ({elapsed:.0f}s elapsed)')

    print(f'  Completed in {time.time() - t_start:.1f}s')
    return pd.DataFrame(records, index=descriptions.index)


def print_evaluation(model_label, y_true, y_pred, categories):
    """Print accuracy and full classification report for one model."""
    acc = accuracy_score(y_true, y_pred)
    print(f'\n{"=" * 58}')
    print(f'  Evaluation — {model_label}')
    print(f'{"=" * 58}')
    print(f'  Validation set n : {len(y_true)}')
    print(f'  Accuracy         : {acc:.4f}  ({acc * 100:.2f}%)')
    print()
    print(classification_report(y_true, y_pred, labels=categories, zero_division=0))


# ============================================================
# PLOTTING FUNCTIONS
# ============================================================

def plot_01_genre_distribution(df_val, save_path='plot_01_genre_distribution.png'):
    """
    Bar chart of ground truth category counts in the validation subset.
    All 7 target categories are shown; zero-count categories indicate
    those absent from the current sample.
    """
    counts = df_val['ground_truth'].value_counts().reindex(TARGET_LABELS, fill_value=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(counts.index, counts.values, color=COL_NEUTRAL)
    ax.set_title('Ground Truth Category Distribution — Validation Subset', fontsize=14, pad=12)
    ax.set_xlabel('Category', fontsize=12)
    ax.set_ylabel('Number of Books', fontsize=12)
    ax.tick_params(axis='x', rotation=30)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.1,
                str(int(h)), ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'    Saved: {save_path}')


def plot_02_model_accuracy(acc_bart, acc_roberta, save_path='plot_02_model_accuracy.png'):
    """
    Horizontal bar chart comparing overall accuracy of both models on the
    validation subset. Horizontal layout accommodates the full model names.
    """
    models     = ['facebook/bart-large-mnli', 'roberta-large-mnli']
    accuracies = [acc_bart * 100, acc_roberta * 100]
    colours    = [COL_BART, COL_ROBERTA]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.barh(models, accuracies, color=colours, height=0.4)
    ax.set_title('Model Accuracy Comparison', fontsize=14, pad=12)
    ax.set_xlabel('Accuracy (%)', fontsize=12)
    ax.set_xlim(0, 115)
    for bar, val in zip(bars, accuracies):
        ax.text(val + 1.0,
                bar.get_y() + bar.get_height() / 2,
                f'{val:.1f}%', ha='left', va='center', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'    Saved: {save_path}')


def plot_03_f1_comparison(report_bart, report_roberta, categories,
                           save_path='plot_03_f1_comparison.png'):
    """
    Grouped bar chart of per-category F1 scores for both models.
    Only categories present in the validation ground truth are shown.
    """
    f1_bart    = [report_bart.get(c,    {}).get('f1-score', 0) for c in categories]
    f1_roberta = [report_roberta.get(c, {}).get('f1-score', 0) for c in categories]

    x     = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars_b = ax.bar(x - width / 2, f1_bart,    width, label='facebook/bart-large-mnli',
                    color=COL_BART)
    bars_r = ax.bar(x + width / 2, f1_roberta, width, label='roberta-large-mnli',
                    color=COL_ROBERTA)

    ax.set_title('Per-Category F1 Score — Model Comparison', fontsize=14, pad=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=30, ha='right', fontsize=11)
    ax.set_ylim(0, 1.2)
    ax.legend(fontsize=10)

    for bar in [*bars_b, *bars_r]:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02,
                    f'{h:.2f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'    Saved: {save_path}')


def plot_04_confusion_matrix(y_true, y_pred, categories, model_name,
                              save_path='plot_04_confusion_matrix.png'):
    """Confusion matrix heatmap for the best-performing model."""
    cm = confusion_matrix(y_true, y_pred, labels=categories)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=categories, yticklabels=categories,
                linewidths=0.5, ax=ax)
    short_name = model_name.split('/')[-1]
    ax.set_title(f'Confusion Matrix — {short_name}', fontsize=13, pad=12)
    ax.set_xlabel('Predicted Category', fontsize=11)
    ax.set_ylabel('True Category', fontsize=11)
    ax.tick_params(axis='x', rotation=30)
    ax.tick_params(axis='y', rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'    Saved: {save_path}')


def plot_05_confidence_distribution(df, save_path='plot_05_confidence_distribution.png'):
    """
    Overlaid histogram of top-1 confidence scores for both models across all
    classified records. Dashed vertical lines mark the mean for each model.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(df['bart_top1_score'],    bins=15, alpha=0.6, color=COL_BART,
            label='facebook/bart-large-mnli', edgecolor='white')
    ax.hist(df['roberta_top1_score'], bins=15, alpha=0.6, color=COL_ROBERTA,
            label='roberta-large-mnli', edgecolor='white')

    bart_mean    = df['bart_top1_score'].mean()
    roberta_mean = df['roberta_top1_score'].mean()
    ax.axvline(bart_mean,    color=COL_BART,    linestyle='--', linewidth=1.5,
               label=f'bart mean: {bart_mean:.2f}')
    ax.axvline(roberta_mean, color=COL_ROBERTA, linestyle='--', linewidth=1.5,
               label=f'roberta mean: {roberta_mean:.2f}')

    ax.set_title('Top-1 Confidence Score Distribution', fontsize=14, pad=12)
    ax.set_xlabel('Confidence Score', fontsize=12)
    ax.set_ylabel('Number of Books', fontsize=12)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'    Saved: {save_path}')


def plot_06_confidence_margin(df, save_path='plot_06_confidence_margin.png'):
    """
    Overlaid histogram of confidence margin (top-1 minus top-2 score) for both
    models. A dotted reference line at 0.10 marks the low-certainty threshold:
    books below this line were near-tied between two categories.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(df['bart_conf_margin'],    bins=15, alpha=0.6, color=COL_BART,
            label='facebook/bart-large-mnli', edgecolor='white')
    ax.hist(df['roberta_conf_margin'], bins=15, alpha=0.6, color=COL_ROBERTA,
            label='roberta-large-mnli', edgecolor='white')

    ax.axvline(0.10, color='grey', linestyle=':', linewidth=1.5,
               label='Margin = 0.10 (low certainty threshold)')

    ax.set_title('Confidence Margin Distribution (Top-1 minus Top-2 Score)',
                 fontsize=14, pad=12)
    ax.set_xlabel('Confidence Margin', fontsize=12)
    ax.set_ylabel('Number of Books', fontsize=12)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'    Saved: {save_path}')


# ============================================================
# MAIN
# ============================================================

def main():
    t_total_start = time.time()

    # ----------------------------------------------------------
    # 1. LOAD AND CLEAN DATA
    # ----------------------------------------------------------
    print('\n[1] Loading data...')
    print('  Reading full file and cleaning before sampling...')

    # engine='python' tolerates malformed quoting in the raw scraped CSV (the C engine aborts entirely on the first
    # unescaped quote it can't resolve).  on_bad_lines='skip' drops any row that still can't be parsed.
    df_full = pd.read_csv(
        DATA_FILE,
        usecols=['title', 'author', 'description', 'genres'],
        engine='python',
        on_bad_lines='skip'
    )
    df_full = df_full.dropna(subset=['description', 'genres'])

    # Remove catalogue / metadata entries that are not book descriptions
    mask = ~df_full['description'].str.contains('librarian|isbn', case=False, regex=True, na=False)
    df_full = df_full[mask].copy()
    df_full['description'] = df_full['description'].str.strip()

    print(f'  Usable records in full dataset: {len(df_full)}')

    # Random sample — NOT the first N rows. The source file is rank-ordered by popularity, so taking the head of the
    # file systematically over-represents bestseller-skewed genres (Fantasy/YA/Romance) and can exclude others
    # (e.g. Biography) almost entirely. A reproducible random sample avoids this.
    n_sample = min(SAMPLE_SIZE, len(df_full))
    if n_sample < SAMPLE_SIZE:
        print(f'  WARNING: SAMPLE_SIZE ({SAMPLE_SIZE}) exceeds usable records '
              f'({len(df_full)}). Sampling {n_sample} instead.')

    df = df_full.sample(n=n_sample, random_state=RANDOM_STATE).reset_index(drop=True)

    print(f'  Random sample drawn: {len(df)}  (random_state={RANDOM_STATE})')
    print(df[['title', 'author', 'description']].head())

    # ----------------------------------------------------------
    # 2. GENRE MAPPING — BUILD GROUND TRUTH
    # ----------------------------------------------------------
    print('\n[2] Building ground truth via genre mapping...')
    df['ground_truth']          = df['genres'].apply(map_to_category)
    df['fiction_nonfiction_gt'] = df['genres'].apply(map_to_fiction_nonfiction)

    n_valid    = int(df['ground_truth'].notna().sum())
    n_excluded = len(df) - n_valid

    print(f'  Total records:             {len(df)}')
    print(f'  Clean validation subset:   {n_valid}  (single unambiguous category match)')
    print(f'  Excluded from evaluation:  {n_excluded}  (ambiguous or no category match)')
    print(f'\n  Ground truth distribution:')
    print(df['ground_truth'].value_counts().to_string())

    if n_valid < 10:
        print(f'\n  WARNING: Validation subset is small ({n_valid} records).')
        print('  Consider increasing SAMPLE_SIZE for more reliable evaluation metrics.')

    # ----------------------------------------------------------
    # 3. LOAD CLASSIFIERS
    # ----------------------------------------------------------
    print('\n[3] Loading classifiers')
    print(f'  Loading {MODEL_BART}')
    clf_bart    = pipeline('zero-shot-classification', model=MODEL_BART,    device=-1)
    print(f'  Loading {MODEL_ROBERTA}')
    clf_roberta = pipeline('zero-shot-classification', model=MODEL_ROBERTA, device=-1)
    print('  Both classifiers loaded.')

    # ----------------------------------------------------------
    # 4. RUN CLASSIFICATION — BOTH MODELS ON ALL RECORDS
    # ----------------------------------------------------------
    print('\n[4] Running zero-shot classification...')
    results_bart    = run_classifier(df['description'], clf_bart,    TARGET_LABELS, MODEL_BART)
    results_roberta = run_classifier(df['description'], clf_roberta, TARGET_LABELS, MODEL_ROBERTA)

    # Attach predictions and scores to main dataframe, aligned by index
    df['bart_prediction']    = results_bart['predicted_label']
    df['bart_top1_score']    = results_bart['top1_score']
    df['bart_top2_score']    = results_bart['top2_score']
    df['bart_conf_margin']   = results_bart['conf_margin']

    df['roberta_prediction']  = results_roberta['predicted_label']
    df['roberta_top1_score']  = results_roberta['top1_score']
    df['roberta_top2_score']  = results_roberta['top2_score']
    df['roberta_conf_margin'] = results_roberta['conf_margin']

    # ----------------------------------------------------------
    # 5. EVALUATION — VALIDATION SUBSET ONLY
    # ----------------------------------------------------------
    print('\n[5] Evaluating models on validation subset...')
    df_val = df[df['ground_truth'].notna()].copy()

    # Only evaluate against categories actually present in the ground truth;
    # prevents misleading zero-support rows in the classification report
    valid_categories = [c for c in TARGET_LABELS if c in df_val['ground_truth'].values]

    acc_bart    = accuracy_score(df_val['ground_truth'], df_val['bart_prediction'])
    acc_roberta = accuracy_score(df_val['ground_truth'], df_val['roberta_prediction'])

    report_bart    = classification_report(
        df_val['ground_truth'], df_val['bart_prediction'],
        labels=valid_categories, zero_division=0, output_dict=True
    )
    report_roberta = classification_report(
        df_val['ground_truth'], df_val['roberta_prediction'],
        labels=valid_categories, zero_division=0, output_dict=True
    )

    print_evaluation(MODEL_BART,    df_val['ground_truth'],
                     df_val['bart_prediction'],    valid_categories)
    print_evaluation(MODEL_ROBERTA, df_val['ground_truth'],
                     df_val['roberta_prediction'], valid_categories)

    # Identify the better-performing model
    best_model    = MODEL_BART    if acc_bart >= acc_roberta else MODEL_ROBERTA
    best_pred_col = 'bart_prediction' if acc_bart >= acc_roberta else 'roberta_prediction'
    best_acc      = max(acc_bart, acc_roberta)
    print(f'\n  Best model: {best_model}  ({best_acc * 100:.2f}% accuracy)')

    # ----------------------------------------------------------
    # 6. FICTION / NON-FICTION — NEGATIVE FINDING TEST
    # ----------------------------------------------------------
    print('\n[6] Fiction / Non-Fiction negative finding test...')
    df_fn = df[df['fiction_nonfiction_gt'].notna()].copy()

    if len(df_fn) == 0:
        print('  No unambiguous Fiction/Non-Fiction records found in this sample.')
        print('  Consider increasing SAMPLE_SIZE or reviewing FICTION/NONFICTION_INDICATORS.')
    else:
        # Run the best model on the Fiction/Non-Fiction classification task
        clf_best = clf_bart if acc_bart >= acc_roberta else clf_roberta
        print(f'  Using: {best_model}')
        print(f'  Classifying {len(df_fn)} records...')

        fn_preds = []
        t_fn     = time.time()
        fn_texts = df_fn['description'].tolist()

        for i, text in enumerate(fn_texts, 1):
            result = clf_best(str(text), FICTION_NONFICTION_LABELS)
            fn_preds.append(result['labels'][0])
            if i % 10 == 0 or i == len(fn_texts):
                print(f'  {i}/{len(fn_texts)} classified  ({time.time() - t_fn:.0f}s elapsed)')

        df_fn['fn_prediction'] = fn_preds
        fn_acc = accuracy_score(df_fn['fiction_nonfiction_gt'], df_fn['fn_prediction'])

        print(f'\n  Fiction / Non-Fiction Results:')
        print(f'  Records in test:    {len(df_fn)}')
        print(f'  Accuracy:           {fn_acc:.4f}  ({fn_acc * 100:.2f}%)')
        print(f'  Random baseline:    50.00%')
        print()
        print(classification_report(df_fn['fiction_nonfiction_gt'],
                                    df_fn['fn_prediction'], zero_division=0))

    # ----------------------------------------------------------
    # 7. CONFIDENCE SCORE ANALYSIS
    # ----------------------------------------------------------
    print('\n[7] Confidence score analysis (all classified records)...')

    for col_score, col_margin, label in [
        ('bart_top1_score',    'bart_conf_margin',    MODEL_BART),
        ('roberta_top1_score', 'roberta_conf_margin', MODEL_ROBERTA),
    ]:
        scores  = df[col_score]
        margins = df[col_margin]
        print(f'\n  {label}:')
        print(f'    Mean top-1 confidence:            {scores.mean():.3f}')
        print(f'    Median top-1 confidence:          {scores.median():.3f}')
        print(f'    % with confidence > 0.80:         {(scores > 0.80).mean() * 100:.1f}%')
        print(f'    % with confidence margin < 0.10:  {(margins < 0.10).mean() * 100:.1f}%'
              f'  (low certainty — near-tied categories)')

    # ----------------------------------------------------------
    # 8. VISUALISATIONS
    # ----------------------------------------------------------
    print('\n[8] Generating plots...')
    plot_01_genre_distribution(df_val)
    plot_02_model_accuracy(acc_bart, acc_roberta)
    plot_03_f1_comparison(report_bart, report_roberta, valid_categories)
    plot_04_confusion_matrix(
        df_val['ground_truth'], df_val[best_pred_col],
        valid_categories, best_model
    )
    plot_05_confidence_distribution(df)
    plot_06_confidence_margin(df)
    print('  All plots saved.')

    # ----------------------------------------------------------
    # 9. SAVE OUTPUT
    # ----------------------------------------------------------
    print('\n[9] Saving output...')
    output_cols = [
        'author', 'title', 'description', 'ground_truth',
        'bart_prediction',    'bart_top1_score',    'bart_conf_margin',
        'roberta_prediction', 'roberta_top1_score', 'roberta_conf_margin',
    ]
    output_file = 'Zero_Shot_Classification_output.xlsx'
    df[output_cols].to_excel(output_file, index=False)
    print(f'  Saved: {output_file}')

    # ----------------------------------------------------------
    # 10. RUN SUMMARY
    # ----------------------------------------------------------
    t_total = time.time() - t_total_start
    print(f'\n{"=" * 58}')
    print('  RUN SUMMARY')
    print(f'{"=" * 58}')
    print(f'  Sample size (SAMPLE_SIZE):       {SAMPLE_SIZE}')
    print(f'  Records after cleaning:          {len(df)}')
    print(f'  Clean validation subset:         {n_valid}')
    print(f'  Excluded from evaluation:        {n_excluded}')
    print(f'  {MODEL_BART:<36} Accuracy: {acc_bart * 100:.2f}%')
    print(f'  {MODEL_ROBERTA:<36} Accuracy: {acc_roberta * 100:.2f}%')
    print(f'  Best model:                      {best_model}')
    print(f'  Total runtime:                   {t_total:.1f}s')
    print(f'{"=" * 58}')


if __name__ == '__main__':
    main()