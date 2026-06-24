# ============================================================
# Image Classification — google/vit-base-patch16-224 vs
#                         microsoft/resnet-50
# Main Analysis Pipeline — Proof of Concept
# ============================================================
# Dataset:   Imagenette (160px), a clean 10-class subset of
#            ImageNet-1k. Auto-downloaded on first run from:
#            https://github.com/fastai/imagenette
#            (no login or API key required)
#
# Models:    google/vit-base-patch16-224  (Vision Transformer)
#            microsoft/resnet-50          (Convolutional Neural Network)
#
# Note: Scoped as a CPU-compatible Proof of Concept. Unlike the
#       Zero-Shot Classification project, image classification
#       requires only a single forward pass per image regardless
#       of the number of output classes, so this script runs in
#       well under 5 minutes on CPU for the default sample size
#       (excluding the one-time dataset/model download).
#
# Required packages (pip install <package>):
#   torch transformers Pillow pandas numpy matplotlib seaborn
#   scikit-learn openpyxl
# ============================================================

import os
import random
import tarfile
import time
import urllib.request
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import AutoImageProcessor, AutoModelForImageClassification
from transformers import logging as hf_logging

warnings.filterwarnings('ignore')
hf_logging.set_verbosity_error()  # Suppress routine model-loading warnings

# Display settings
pd.set_option('display.width', 320)
pd.set_option('display.max_columns', 12)
pd.set_option('display.precision', 3)

sns.set_theme(style='whitegrid', font_scale=1.1)


# ============================================================
# CONFIGURATION
# ============================================================

IMAGENETTE_URL = 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz'
DATA_ROOT      = 'imagenette_data'
DATA_DIR       = os.path.join(DATA_ROOT, 'imagenette2-160')
ARCHIVE_PATH   = os.path.join(DATA_ROOT, 'imagenette2-160.tgz')

SAMPLES_PER_CLASS = 20          # PoC sample size -- 20 x 10 classes = 200 images
RANDOM_STATE      = 42          # Fixed seed for reproducible sampling

MODEL_VIT    = 'google/vit-base-patch16-224'
MODEL_RESNET = 'microsoft/resnet-50'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Plot colours (consistent with the Zero-Shot Classification project)
COL_VIT     = '#2e86ab'   # Blue  -- ViT model
COL_RESNET  = '#e84855'   # Red   -- ResNet model
COL_NEUTRAL = '#4a4e69'   # Grey  -- single-model elements


# ============================================================
# IMAGENETTE CLASS MAPPING
# ============================================================

# Imagenette class folders are named by their original ImageNet synset ID.
# Each maps to one specific, fixed index in the standard 1000-class
# ImageNet-1k label space used by both models below -- this gives an exact,
# unambiguous ground truth with no fuzzy keyword-mapping required (unlike
# the genre-tag mapping needed for the Zero-Shot Classification project).
SYNSET_TO_IMAGENET_INDEX = {
    'n01440764': 0,    # tench
    'n02102040': 217,  # English springer
    'n02979186': 482,  # cassette player
    'n03000684': 491,  # chain saw
    'n03028079': 497,  # church
    'n03394916': 566,  # French horn
    'n03417042': 569,  # garbage truck
    'n03425413': 571,  # gas pump
    'n03445777': 574,  # golf ball
    'n03888257': 701,  # parachute
}

SYNSET_TO_NAME = {
    'n01440764': 'tench',
    'n02102040': 'English springer',
    'n02979186': 'cassette player',
    'n03000684': 'chain saw',
    'n03028079': 'church',
    'n03394916': 'French horn',
    'n03417042': 'garbage truck',
    'n03425413': 'gas pump',
    'n03445777': 'golf ball',
    'n03888257': 'parachute',
}

TARGET_CLASS_NAMES = sorted(SYNSET_TO_NAME.values(), key=str.lower)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def download_and_extract_imagenette():
    """
    Downloads and extracts Imagenette if not already present locally.
    Safe to re-run: skips the download/extraction if the data already
    exists, so this only costs time on the very first run.
    """
    if os.path.isdir(DATA_DIR):
        print(f'  Imagenette already present at: {DATA_DIR}')
        return

    os.makedirs(DATA_ROOT, exist_ok=True)
    print(f'  Downloading Imagenette (160px, ~99MB) from:\n  {IMAGENETTE_URL}')

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        pct = min(100, downloaded * 100 / total_size) if total_size > 0 else 0
        print(f'\r  Download progress: {pct:5.1f}%', end='', flush=True)

    try:
        urllib.request.urlretrieve(IMAGENETTE_URL, ARCHIVE_PATH, _progress)
    except Exception as e:
        print(f'\n  ERROR: Could not download Imagenette automatically: {e}')
        print(f'  You can download it manually from:\n    {IMAGENETTE_URL}')
        print(f'  Then extract it so that this path exists:\n    {DATA_DIR}')
        raise

    print('\n  Download complete. Extracting...')
    with tarfile.open(ARCHIVE_PATH) as tar:
        tar.extractall(DATA_ROOT)
    os.remove(ARCHIVE_PATH)
    print(f'  Extracted to: {DATA_DIR}')


def build_sample(data_dir, samples_per_class, random_state):
    """
    Build a balanced random sample from the Imagenette validation split:
    `samples_per_class` images per class, drawn from `val/<synset_id>/`.
    Sampling from the validation split (not training) follows standard
    ML evaluation convention -- these images were held out, not used to
    train the original ImageNet models.
    """
    val_dir = os.path.join(data_dir, 'val')
    rng = random.Random(random_state)
    records = []

    for synset_id in sorted(SYNSET_TO_IMAGENET_INDEX.keys()):
        class_dir = os.path.join(val_dir, synset_id)
        all_images = [f for f in os.listdir(class_dir)
                      if f.lower().endswith(('.jpeg', '.jpg', '.png'))]
        chosen = rng.sample(all_images, min(samples_per_class, len(all_images)))

        for fname in chosen:
            records.append({
                'image_path':       os.path.join(class_dir, fname),
                'synset_id':        synset_id,
                'true_class_name':  SYNSET_TO_NAME[synset_id],
                'true_class_index': SYNSET_TO_IMAGENET_INDEX[synset_id],
            })

    return pd.DataFrame(records)


def get_id2label(model, index):
    """
    Robustly looks up a class label by index. Some Hugging Face model
    configs store id2label keys as integers, others as strings -- this
    handles both without raising a KeyError.
    """
    id2label = model.config.id2label
    return id2label.get(index, id2label.get(str(index), f'class_{index}'))


def map_prediction_to_known_class(pred_index):
    """
    Maps a predicted ImageNet-1k index back to its Imagenette class name
    if it matches one of the 10 known target classes; otherwise returns
    'Other'. Unlike the Zero-Shot Classification project, these models
    are not restricted to a fixed candidate label set -- they classify
    freely across all 1000 ImageNet-1k classes, so an incorrect
    prediction can land on any of the other 990 classes, not just the
    9 other Imagenette categories.
    """
    for synset_id, idx in SYNSET_TO_IMAGENET_INDEX.items():
        if idx == pred_index:
            return SYNSET_TO_NAME[synset_id]
    return 'Other'


def classify_images(df, processor, model, model_label):
    """
    Run image classification on every image in df using the given model.
    Returns a DataFrame (aligned to df's index) containing:
      pred_index    -- predicted top-1 ImageNet-1k class index
      pred_name     -- the model's own label text for that index
      top1_score    -- softmax confidence of the top-1 prediction
      top2_score    -- softmax confidence of the runner-up prediction
      conf_margin   -- top1_score minus top2_score
      top5_indices  -- the 5 highest-confidence class indices
    """
    print(f'\n  Running: {model_label}')
    print(f'  {len(df)} images to classify...')

    records = []
    t_start = time.time()

    for i, image_path in enumerate(df['image_path'], 1):
        image = Image.open(image_path).convert('RGB')
        inputs = processor(images=image, return_tensors='pt').to(DEVICE)

        with torch.no_grad():
            logits = model(**inputs).logits

        probs = torch.softmax(logits, dim=-1)[0]
        top5_probs, top5_indices = torch.topk(probs, 5)

        pred_index = int(top5_indices[0])
        records.append({
            'pred_index':   pred_index,
            'pred_name':    get_id2label(model, pred_index),
            'top1_score':   float(top5_probs[0]),
            'top2_score':   float(top5_probs[1]),
            'conf_margin':  float(top5_probs[0] - top5_probs[1]),
            'top5_indices': top5_indices.cpu().tolist(),
        })

        if i % 25 == 0 or i == len(df):
            print(f'  {i}/{len(df)} classified  ({time.time() - t_start:.0f}s elapsed)')

    print(f'  Completed in {time.time() - t_start:.1f}s')
    return pd.DataFrame(records, index=df.index)


def print_evaluation(model_label, acc_top1, acc_top5, other_rate, y_true, y_pred, categories):
    """Print top-1/top-5 accuracy and full classification report for one model."""
    print(f'\n{"=" * 58}')
    print(f'  Evaluation -- {model_label}')
    print(f'{"=" * 58}')
    print(f'  Top-1 Accuracy : {acc_top1:.4f}  ({acc_top1 * 100:.2f}%)')
    print(f'  Top-5 Accuracy : {acc_top5:.4f}  ({acc_top5 * 100:.2f}%)')
    print(f'  Predictions outside the 10 known classes: {other_rate * 100:.1f}%')
    print()
    print(classification_report(y_true, y_pred, labels=categories, zero_division=0))


def analyze_other_predictions(df, prefix, model_label, top_n=5):
    """
    For images where the model's top-1 prediction fell outside the 10 known
    Imagenette classes ('Other'), report which specific ImageNet labels the
    model predicted instead -- using the verbatim pred_name already captured
    during classification. Also isolates the single weakest-recall true
    class and shows what it was most often predicted as, since this is
    usually the most informative slice of the 'Other' bucket.
    """
    other_mask  = df[f'{prefix}_pred_class_name'] == 'Other'
    other_preds = df.loc[other_mask, f'{prefix}_pred_name']

    print(f'\n  {model_label} -- most common "Other" predictions ({other_mask.sum()} total):')
    if other_mask.sum() == 0:
        print('    None.')
    else:
        for label, count in other_preds.value_counts().head(top_n).items():
            print(f'    {label}: {count}')

    recall_by_class = df.groupby('true_class_name').apply(
        lambda g: (g[f'{prefix}_pred_index'] == g['true_class_index']).mean()
    )
    weakest_class = recall_by_class.idxmin()
    weak_mask = (df['true_class_name'] == weakest_class) & other_mask
    weak_other_preds = df.loc[weak_mask, f'{prefix}_pred_name']

    print(f'  Weakest class for {model_label}: "{weakest_class}" '
          f'(recall {recall_by_class.min():.2f})')
    if weak_mask.sum() > 0:
        print(f'  "{weakest_class}" images predicted as ("Other"):')
        for label, count in weak_other_preds.value_counts().items():
            print(f'    {label}: {count}')


# ============================================================
# PLOTTING FUNCTIONS
# ============================================================

def plot_01_class_distribution(df, save_path='plot_01_class_distribution.png'):
    """Bar chart confirming the balanced per-class sample size."""
    counts = df['true_class_name'].value_counts().reindex(TARGET_CLASS_NAMES, fill_value=0)

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(counts.index, counts.values, color=COL_NEUTRAL)
    ax.set_title('Validation Sample -- Images per Class', fontsize=14, pad=12)
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Number of Images', fontsize=12)
    ax.tick_params(axis='x', rotation=30)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.1, str(int(h)),
                ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'    Saved: {save_path}')


def plot_02_model_accuracy(acc_vit, acc_resnet, save_path='plot_02_model_accuracy.png'):
    """Horizontal bar chart comparing top-1 accuracy of both models."""
    models  = [MODEL_VIT, MODEL_RESNET]
    accs    = [acc_vit * 100, acc_resnet * 100]
    colours = [COL_VIT, COL_RESNET]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.barh(models, accs, color=colours, height=0.4)
    ax.set_title('Model Top-1 Accuracy Comparison', fontsize=14, pad=12)
    ax.set_xlabel('Accuracy (%)', fontsize=12)
    ax.set_xlim(0, 115)
    for bar, val in zip(bars, accs):
        ax.text(val + 1.0, bar.get_y() + bar.get_height() / 2, f'{val:.1f}%',
                ha='left', va='center', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'    Saved: {save_path}')


def plot_03_f1_comparison(report_vit, report_resnet, categories,
                           save_path='plot_03_f1_comparison.png'):
    """Grouped bar chart of per-class F1 scores for both models."""
    f1_vit    = [report_vit.get(c, {}).get('f1-score', 0) for c in categories]
    f1_resnet = [report_resnet.get(c, {}).get('f1-score', 0) for c in categories]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(13, 6))
    bars_v = ax.bar(x - width / 2, f1_vit,    width, label=MODEL_VIT,    color=COL_VIT)
    bars_r = ax.bar(x + width / 2, f1_resnet, width, label=MODEL_RESNET, color=COL_RESNET)

    ax.set_title('Per-Class F1 Score -- Model Comparison', fontsize=14, pad=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=30, ha='right', fontsize=10)
    ax.set_ylim(0, 1.2)
    ax.legend(fontsize=10)

    for bar in [*bars_v, *bars_r]:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02, f'{h:.2f}',
                    ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'    Saved: {save_path}')


def plot_04_confusion_matrix(y_true, y_pred, labels, model_name,
                              save_path='plot_04_confusion_matrix.png'):
    """
    Confusion matrix heatmap, including an 'Other' column for predictions
    that fell outside the 10 known Imagenette classes entirely.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, linewidths=0.5, ax=ax)
    ax.set_title(f'Confusion Matrix -- {model_name}', fontsize=13, pad=12)
    ax.set_xlabel('Predicted Class', fontsize=11)
    ax.set_ylabel('True Class', fontsize=11)
    ax.tick_params(axis='x', rotation=30)
    ax.tick_params(axis='y', rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'    Saved: {save_path}')


def plot_05_confidence_distribution(df, save_path='plot_05_confidence_distribution.png'):
    """Overlaid histogram of top-1 confidence scores for both models."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df['vit_top1_score'],    bins=15, alpha=0.6, color=COL_VIT,
            label=MODEL_VIT, edgecolor='white')
    ax.hist(df['resnet_top1_score'], bins=15, alpha=0.6, color=COL_RESNET,
            label=MODEL_RESNET, edgecolor='white')

    vit_mean    = df['vit_top1_score'].mean()
    resnet_mean = df['resnet_top1_score'].mean()
    ax.axvline(vit_mean,    color=COL_VIT,    linestyle='--', linewidth=1.5,
               label=f'ViT mean: {vit_mean:.2f}')
    ax.axvline(resnet_mean, color=COL_RESNET, linestyle='--', linewidth=1.5,
               label=f'ResNet mean: {resnet_mean:.2f}')

    ax.set_title('Top-1 Confidence Score Distribution', fontsize=14, pad=12)
    ax.set_xlabel('Confidence Score', fontsize=12)
    ax.set_ylabel('Number of Images', fontsize=12)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'    Saved: {save_path}')


def plot_06_confidence_margin(df, save_path='plot_06_confidence_margin.png'):
    """Overlaid histogram of confidence margin (top-1 minus top-2 score)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df['vit_conf_margin'],    bins=15, alpha=0.6, color=COL_VIT,
            label=MODEL_VIT, edgecolor='white')
    ax.hist(df['resnet_conf_margin'], bins=15, alpha=0.6, color=COL_RESNET,
            label=MODEL_RESNET, edgecolor='white')
    ax.axvline(0.10, color='grey', linestyle=':', linewidth=1.5,
               label='Margin = 0.10 (low certainty threshold)')
    ax.set_title('Confidence Margin Distribution (Top-1 minus Top-2 Score)',
                 fontsize=14, pad=12)
    ax.set_xlabel('Confidence Margin', fontsize=12)
    ax.set_ylabel('Number of Images', fontsize=12)
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
    print(f'Device: {DEVICE}')

    # ----------------------------------------------------------
    # 1. DOWNLOAD / LOCATE DATASET
    # ----------------------------------------------------------
    print('\n[1] Preparing Imagenette dataset...')
    download_and_extract_imagenette()

    # ----------------------------------------------------------
    # 2. BUILD BALANCED SAMPLE
    # ----------------------------------------------------------
    print('\n[2] Building balanced validation sample...')
    df = build_sample(DATA_DIR, SAMPLES_PER_CLASS, RANDOM_STATE)
    n_classes = len(SYNSET_TO_IMAGENET_INDEX)
    print(f'  Total images sampled: {len(df)}  ({SAMPLES_PER_CLASS} per class x {n_classes} classes)')
    print(df['true_class_name'].value_counts().to_string())

    # ----------------------------------------------------------
    # 3. LOAD MODELS
    # ----------------------------------------------------------
    print('\n[3] Loading classifiers...')
    print(f'  Loading {MODEL_VIT}...')
    processor_vit = AutoImageProcessor.from_pretrained(MODEL_VIT)
    model_vit     = AutoModelForImageClassification.from_pretrained(MODEL_VIT).to(DEVICE)

    print(f'  Loading {MODEL_RESNET}...')
    processor_resnet = AutoImageProcessor.from_pretrained(MODEL_RESNET)
    model_resnet     = AutoModelForImageClassification.from_pretrained(MODEL_RESNET).to(DEVICE)

    # Sanity check: confirm both models share the same ImageNet-1k index
    # ordering before relying on it for evaluation.
    check_idx = 0  # known to correspond to 'tench'
    print(f'  Sanity check -- index {check_idx} should be "tench" on both models:')
    print(f'    ViT label:    {get_id2label(model_vit, check_idx)}')
    print(f'    ResNet label: {get_id2label(model_resnet, check_idx)}')
    print('  Both classifiers loaded.')

    # ----------------------------------------------------------
    # 4. RUN CLASSIFICATION
    # ----------------------------------------------------------
    print('\n[4] Running image classification (estimated under 5 minutes on CPU)...')
    results_vit    = classify_images(df, processor_vit, model_vit, MODEL_VIT)
    results_resnet = classify_images(df, processor_resnet, model_resnet, MODEL_RESNET)

    for prefix, results in [('vit', results_vit), ('resnet', results_resnet)]:
        df[f'{prefix}_pred_index']   = results['pred_index']
        df[f'{prefix}_pred_name']    = results['pred_name']
        df[f'{prefix}_top1_score']   = results['top1_score']
        df[f'{prefix}_top2_score']   = results['top2_score']
        df[f'{prefix}_conf_margin']  = results['conf_margin']
        df[f'{prefix}_top5_indices'] = results['top5_indices']

    # ----------------------------------------------------------
    # 5. EVALUATION
    # ----------------------------------------------------------
    print('\n[5] Evaluating models...')

    for prefix in ['vit', 'resnet']:
        df[f'{prefix}_pred_class_name'] = df[f'{prefix}_pred_index'].apply(map_prediction_to_known_class)
        df[f'{prefix}_top1_correct'] = df[f'{prefix}_pred_index'] == df['true_class_index']
        df[f'{prefix}_top5_correct'] = df.apply(
            lambda row, p=prefix: row['true_class_index'] in row[f'{p}_top5_indices'], axis=1
        )

    acc_vit_top1    = df['vit_top1_correct'].mean()
    acc_resnet_top1 = df['resnet_top1_correct'].mean()
    acc_vit_top5    = df['vit_top5_correct'].mean()
    acc_resnet_top5 = df['resnet_top5_correct'].mean()

    other_rate_vit    = (df['vit_pred_class_name'] == 'Other').mean()
    other_rate_resnet = (df['resnet_pred_class_name'] == 'Other').mean()

    report_vit = classification_report(
        df['true_class_name'], df['vit_pred_class_name'],
        labels=TARGET_CLASS_NAMES, zero_division=0, output_dict=True
    )
    report_resnet = classification_report(
        df['true_class_name'], df['resnet_pred_class_name'],
        labels=TARGET_CLASS_NAMES, zero_division=0, output_dict=True
    )

    print_evaluation(MODEL_VIT, acc_vit_top1, acc_vit_top5, other_rate_vit,
                      df['true_class_name'], df['vit_pred_class_name'], TARGET_CLASS_NAMES)
    print_evaluation(MODEL_RESNET, acc_resnet_top1, acc_resnet_top5, other_rate_resnet,
                      df['true_class_name'], df['resnet_pred_class_name'], TARGET_CLASS_NAMES)

    best_model    = MODEL_VIT if acc_vit_top1 >= acc_resnet_top1 else MODEL_RESNET
    best_pred_col = 'vit_pred_class_name' if acc_vit_top1 >= acc_resnet_top1 else 'resnet_pred_class_name'
    best_acc      = max(acc_vit_top1, acc_resnet_top1)
    print(f'\n  Best model (top-1): {best_model}  ({best_acc * 100:.2f}%)')

    # ----------------------------------------------------------
    # 6. CONFIDENCE AND ERROR ANALYSIS
    # ----------------------------------------------------------
    print('\n[6] Confidence and error analysis...')
    for prefix, label in [('vit', MODEL_VIT), ('resnet', MODEL_RESNET)]:
        scores  = df[f'{prefix}_top1_score']
        margins = df[f'{prefix}_conf_margin']
        print(f'\n  {label}:')
        print(f'    Mean top-1 confidence:            {scores.mean():.3f}')
        print(f'    Median top-1 confidence:          {scores.median():.3f}')
        print(f'    % with confidence > 0.80:         {(scores > 0.80).mean() * 100:.1f}%')
        print(f'    % with confidence margin < 0.10:  {(margins < 0.10).mean() * 100:.1f}%')

        incorrect       = df[f'{prefix}_pred_class_name'] != df['true_class_name']
        cross_confusion  = incorrect & (df[f'{prefix}_pred_class_name'] != 'Other')
        out_of_set       = incorrect & (df[f'{prefix}_pred_class_name'] == 'Other')
        n_incorrect      = incorrect.sum()

        print(f'    Incorrect predictions:            {n_incorrect}')
        if n_incorrect > 0:
            print(f'      Cross-confusion among the 10 known classes: '
                  f'{cross_confusion.sum()} ({cross_confusion.sum() / n_incorrect * 100:.1f}%)')
            print(f'      Fell outside the known class set ("Other"):  '
                  f'{out_of_set.sum()} ({out_of_set.sum() / n_incorrect * 100:.1f}%)')

        analyze_other_predictions(df, prefix, label)

    # ----------------------------------------------------------
    # 7. VISUALISATIONS
    # ----------------------------------------------------------
    print('\n[7] Generating plots...')
    plot_01_class_distribution(df)
    plot_02_model_accuracy(acc_vit_top1, acc_resnet_top1)
    plot_03_f1_comparison(report_vit, report_resnet, TARGET_CLASS_NAMES)
    plot_04_confusion_matrix(
        df['true_class_name'], df[best_pred_col],
        TARGET_CLASS_NAMES + ['Other'], best_model
    )
    plot_05_confidence_distribution(df)
    plot_06_confidence_margin(df)
    print('  All plots saved.')

    # ----------------------------------------------------------
    # 8. SAVE OUTPUT
    # ----------------------------------------------------------
    print('\n[8] Saving output...')
    output_cols = [
        'image_path', 'true_class_name',
        'vit_pred_class_name',    'vit_pred_name',    'vit_top1_score',    'vit_conf_margin',
        'resnet_pred_class_name', 'resnet_pred_name', 'resnet_top1_score', 'resnet_conf_margin',
    ]
    output_file = 'Image_Classification_output.xlsx'
    df[output_cols].to_excel(output_file, index=False)
    print(f'  Saved: {output_file}')

    # ----------------------------------------------------------
    # 9. RUN SUMMARY
    # ----------------------------------------------------------
    t_total = time.time() - t_total_start
    print(f'\n{"=" * 58}')
    print('  RUN SUMMARY')
    print(f'{"=" * 58}')
    print(f'  Samples per class:               {SAMPLES_PER_CLASS}')
    print(f'  Total images classified:         {len(df)}')
    print(f'  {MODEL_VIT:<32} Top-1: {acc_vit_top1 * 100:.2f}%   Top-5: {acc_vit_top5 * 100:.2f}%')
    print(f'  {MODEL_RESNET:<32} Top-1: {acc_resnet_top1 * 100:.2f}%   Top-5: {acc_resnet_top5 * 100:.2f}%')
    print(f'  Best model (top-1):              {best_model}')
    print(f'  Total runtime:                   {t_total:.1f}s')
    print(f'{"=" * 58}')


if __name__ == '__main__':
    main()