"""
Evaluates PhotoTune's mood classification accuracy against a test set
organized as one folder per mood (ImageFolder convention).

Produces:
  - Top-1 and Top-3 accuracy
  - Per-mood precision, recall, F1
  - Confusion matrix heatmap
  - Per-mood metrics bar chart
  - Detailed per-photo CSV
"""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from dotenv import load_dotenv

from moods import MOODS
from phototune import Deezer, LastFM, PhotoTune, load_credentials

load_dotenv()

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".avif", ".bmp"}

def discover_labeled_photos(photos_dir: Path) -> list[tuple[Path, str]]:
    """Walk the photo folders and pair each image with its mood (folder name).
    Returns a list of (image_path, true_mood) tuples.
    """
    if not photos_dir.exists():
        sys.exit(f"Photos folder not found: {photos_dir}")

    pairs = []
    found_folders = []
    skipped_folders = []

    for subfolder in sorted(photos_dir.iterdir()):
        if not subfolder.is_dir():
            continue

        mood = subfolder.name
        if mood not in MOODS:
            skipped_folders.append(mood)
            continue

        found_folders.append(mood)
        images_in_folder = 0
        for image_file in sorted(subfolder.iterdir()):
            if image_file.suffix.lower() in VALID_EXTENSIONS:
                pairs.append((image_file, mood))
                images_in_folder += 1

        if images_in_folder == 0:
            print(f"  Note: '{mood}/' folder exists but contains no images.")

    if skipped_folders:
        print(f"Skipped unknown folders (not in MOODS): {skipped_folders}")

    if not found_folders:
        sys.exit(
            f"No valid mood folders found in {photos_dir}.\n"
            f"Expected subfolders named: {list(MOODS.keys())}"
        )

    print(f"Found {len(pairs)} photos across {len(found_folders)} mood folders.")
    return pairs


def evaluate(tune: PhotoTune, labeled_photos: list[tuple[Path, str]]):
    """Run PhotoTune on each labeled photo and collect predictions."""
    from PIL import Image

    results = []
    print(f"\nEvaluating {len(labeled_photos)} photos...")
    print("-" * 80)

    for i, (image_path, true_mood) in enumerate(labeled_photos, 1):
        # We only need score_moods, not the full recommend() pipeline.
        # This skips the Last.fm/Deezer calls and makes evaluation fast.
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"  [{i}/{len(labeled_photos)}] SKIP: {image_path.name} ({e})")
            continue

        ranked_moods = tune.score_moods(image)

        top1 = ranked_moods[0][0]
        top3_moods = [m for m, _ in ranked_moods[:3]]
        top1_confidence = ranked_moods[0][1]

        correct_top1 = top1 == true_mood
        correct_top3 = true_mood in top3_moods

        results.append({
            "filename": image_path.name,
            "folder": true_mood,
            "true_mood": true_mood,
            "predicted_top1": top1,
            "predicted_top3": ",".join(top3_moods),
            "top1_confidence": round(top1_confidence, 4),
            "correct_top1": correct_top1,
            "correct_top3": correct_top3,
        })

        marker = "✓" if correct_top1 else "✗"
        # Truncate long filenames for readable output.
        display_name = image_path.name[:28]
        print(f"  [{i}/{len(labeled_photos)}] {marker} {display_name:<30s} "
              f"true={true_mood:<18s} pred={top1:<18s} ({top1_confidence:.0%})")

    return results


def compute_metrics(results: list[dict]) -> dict:
    """Compute overall and per-mood metrics from the raw results."""
    n = len(results)
    if n == 0:
        return {}

    top1_correct = sum(1 for r in results if r["correct_top1"])
    top3_correct = sum(1 for r in results if r["correct_top3"])

    per_mood_tp = Counter()
    per_mood_fp = Counter()
    per_mood_fn = Counter()
    per_mood_support = Counter()

    for r in results:
        true, pred = r["true_mood"], r["predicted_top1"]
        per_mood_support[true] += 1
        if true == pred:
            per_mood_tp[true] += 1
        else:
            per_mood_fp[pred] += 1
            per_mood_fn[true] += 1

    per_mood = {}
    for mood in MOODS:
        tp = per_mood_tp[mood]
        fp = per_mood_fp[mood]
        fn = per_mood_fn[mood]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        per_mood[mood] = {
            "support": per_mood_support[mood],
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
        }

    return {
        "n_photos": n,
        "top1_accuracy": round(top1_correct / n, 3),
        "top3_accuracy": round(top3_correct / n, 3),
        "per_mood": per_mood,
    }


def save_results_csv(results: list[dict], output_path: Path):
    if not results:
        return
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nPer-photo results saved to: {output_path}")


def plot_confusion_matrix(results: list[dict], output_path: Path):
    """Build and save a confusion matrix heatmap.

    Rows are true moods, columns are predicted moods. The diagonal shows
    correct predictions; off-diagonal cells reveal which moods get confused
    with which.
    """
    mood_list = list(MOODS.keys())
    mood_to_idx = {m: i for i, m in enumerate(mood_list)}
    n_moods = len(mood_list)
    matrix = np.zeros((n_moods, n_moods), dtype=int)

    for r in results:
        i = mood_to_idx[r["true_mood"]]
        j = mood_to_idx[r["predicted_top1"]]
        matrix[i, j] += 1

    pretty = [m.replace("_", " ").title() for m in mood_list]

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        matrix,
        annot=True, fmt="d", cmap="Purples",
        xticklabels=pretty, yticklabels=pretty,
        cbar_kws={"label": "Count"},
        square=True, linewidths=0.5,
    )
    plt.xlabel("Predicted Mood", fontsize=12)
    plt.ylabel("True Mood", fontsize=12)
    plt.title("PhotoTune Mood Classification - Confusion Matrix",
              fontsize=14, pad=20)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved to: {output_path}")


def plot_per_mood_metrics(metrics: dict, output_path: Path):
    """Bar chart showing precision, recall, F1 for each mood."""
    moods = list(metrics["per_mood"].keys())
    pretty = [m.replace("_", " ").title() for m in moods]
    precision = [metrics["per_mood"][m]["precision"] for m in moods]
    recall = [metrics["per_mood"][m]["recall"] for m in moods]
    f1 = [metrics["per_mood"][m]["f1"] for m in moods]

    x = np.arange(len(moods))
    width = 0.27

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width, precision, width, label="Precision", color="#a78bfa")
    ax.bar(x, recall, width, label="Recall", color="#ec4899")
    ax.bar(x + width, f1, width, label="F1", color="#6366f1")

    ax.set_xlabel("Mood", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Per-Mood Classification Metrics", fontsize=14, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(pretty, rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Per-mood metrics chart saved to: {output_path}")


def print_summary(metrics: dict):
    print("\n" + "=" * 64)
    print("EVALUATION RESULTS")
    print("=" * 64)
    print(f"  Photos evaluated:    {metrics['n_photos']}")
    print(f"  Top-1 accuracy:      {metrics['top1_accuracy']:.1%}")
    print(f"  Top-3 accuracy:      {metrics['top3_accuracy']:.1%}")
    print()
    print("Per-mood breakdown (precision / recall / F1):")
    print(f"  {'Mood':<20s} {'Support':>8s} {'Precision':>10s} {'Recall':>8s} {'F1':>6s}")
    print(f"  {'-' * 20:<20s} {'-' * 8:>8s} {'-' * 10:>10s} {'-' * 8:>8s} {'-' * 6:>6s}")
    for mood, m in metrics["per_mood"].items():
        print(f"  {mood:<20s} {m['support']:>8d} {m['precision']:>10.3f} "
              f"{m['recall']:>8.3f} {m['f1']:>6.3f}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Evaluate PhotoTune mood classification.")
    parser.add_argument(
        "--photos-dir",
        type=Path,
        default=Path("evaluation/eval_set/photos"),
        help="Folder containing mood subfolders with images",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("evaluation/results"),
        help="Where to save outputs",
    )
    args = parser.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)
    
    lastfm_key = load_credentials()
    tune = PhotoTune(LastFM(lastfm_key), Deezer())

    labeled_photos = discover_labeled_photos(args.photos_dir)
    results = evaluate(tune, labeled_photos)

    if not results:
        sys.exit("No photos were successfully evaluated.")

    metrics = compute_metrics(results)
    print_summary(metrics)

    save_results_csv(results, args.results_dir / "per_photo_results.csv")
    plot_confusion_matrix(results, args.results_dir / "confusion_matrix.png")
    plot_per_mood_metrics(metrics, args.results_dir / "per_mood_metrics.png")

    with open(args.results_dir / "summary.txt", "w") as f:
        f.write(f"Photos evaluated: {metrics['n_photos']}\n")
        f.write(f"Top-1 accuracy: {metrics['top1_accuracy']:.1%}\n")
        f.write(f"Top-3 accuracy: {metrics['top3_accuracy']:.1%}\n\n")
        f.write("Per-mood F1 scores:\n")
        for mood, m in sorted(metrics["per_mood"].items(),
                              key=lambda kv: kv[1]["f1"], reverse=True):
            f.write(f"  {mood:<20s} F1={m['f1']:.3f}  (n={m['support']})\n")
    print(f"Summary saved to: {args.results_dir / 'summary.txt'}")


if __name__ == "__main__":
    main()