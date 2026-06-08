"""
Multilingual CoT Visualization

Generates cross-language comparison plots from analyze_rollouts output.
Loads chunks_labeled.json from problem directories and aggregates metrics.

Usage:
    python -m multicot.plots --dataset mgsm --languages en,fr,zh --model Llama-3.2-3B-Instruct-Turbo
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm


IMPORTANCE_METRICS = [
    "counterfactual_importance_accuracy",
    "counterfactual_importance_kl",
    "resampling_importance_accuracy",
    "resampling_importance_kl",
    "forced_importance_accuracy",
    "forced_importance_kl",
]

FUNCTION_TAGS = [
    "problem_setup",
    "plan_generation",
    "fact_retrieval",
    "active_computation",
    "result_consolidation",
    "uncertainty_management",
    "final_answer_emission",
    "self_checking",
    "unknown",
]

LANGUAGE_NAMES = {
    "en": "English", "fr": "French", "zh": "Chinese",
    "ar": "Arabic", "de": "German", "es": "Spanish",
    "hi": "Hindi", "ja": "Japanese", "ko": "Korean",
    "pt": "Portuguese", "it": "Italian",
}

# Colorblind-friendly palette
LANG_COLORS = {
    "en": "#0072B2",  # blue
    "fr": "#D55E00",  # red-orange
    "zh": "#009E73",  # green
    "ar": "#CC79A7",  # pink
    "de": "#E69F00",  # orange
    "es": "#56B4E9",  # sky blue
    "hi": "#F0E442",  # yellow
    "ja": "#000000",  # black
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_language_chunks(rollouts_dir: Path, language: str) -> list[dict]:
    """
    Load all chunks_labeled.json entries for a language.

    Returns a flat list of chunk dicts (each has chunk_idx, importance metrics, etc.)
    with added 'problem_id' and 'language' fields.
    """
    lang_dir = rollouts_dir / language / "correct_base_solution"
    if not lang_dir.exists():
        print(f"  [warn] No data at {lang_dir}")
        return []

    all_chunks = []
    for problem_dir in sorted(lang_dir.iterdir()):
        if not problem_dir.is_dir() or not problem_dir.name.startswith("problem_"):
            continue
        labeled_file = problem_dir / "chunks_labeled.json"
        if not labeled_file.exists():
            continue
        try:
            with open(labeled_file, "r", encoding="utf-8") as f:
                chunks = json.load(f)
            for c in chunks:
                c["problem_id"] = problem_dir.name
                c["language"] = language
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"  [warn] Could not load {labeled_file}: {e}")

    return all_chunks


# ---------------------------------------------------------------------------
# Plot 1: Mean importance curve by normalized chunk position
# ---------------------------------------------------------------------------

def plot_importance_curves(
    lang_chunks: dict[str, list[dict]],
    metric: str,
    output_path: Path,
    n_bins: int = 10,
):
    """
    Plot mean importance vs. normalized chunk position (0→1) for each language.

    Chunks are binned by their relative position within the problem CoT.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for language, chunks in lang_chunks.items():
        if not chunks:
            continue

        # Group by problem to get per-problem normalized positions
        by_problem = defaultdict(list)
        for c in chunks:
            if metric in c and c[metric] is not None:
                by_problem[c.get("problem_id", "default")].append(c)

        # Bin chunks by normalized position across all problems
        bin_values = defaultdict(list)
        for pid, prob_chunks in by_problem.items():
            n = len(prob_chunks)
            if n == 0:
                continue
            for i, c in enumerate(sorted(prob_chunks, key=lambda x: x.get("chunk_idx", 0))):
                norm_pos = i / max(n - 1, 1)
                bin_idx = min(int(norm_pos * n_bins), n_bins - 1)
                val = c.get(metric)
                if val is not None and not np.isnan(val):
                    bin_values[bin_idx].append(val)

        if not bin_values:
            continue

        xs = sorted(bin_values.keys())
        ys = [np.mean(bin_values[b]) for b in xs]
        yerr = [np.std(bin_values[b]) / np.sqrt(len(bin_values[b])) for b in xs]
        xs_norm = [x / (n_bins - 1) for x in xs]

        color = LANG_COLORS.get(language, None)
        label = LANGUAGE_NAMES.get(language, language)
        ax.plot(xs_norm, ys, marker="o", label=label, color=color, linewidth=2, markersize=5)
        ax.fill_between(
            xs_norm,
            [y - e for y, e in zip(ys, yerr)],
            [y + e for y, e in zip(ys, yerr)],
            alpha=0.15,
            color=color,
        )

    ax.set_xlabel("Normalized Chunk Position (0=start → 1=end)", fontsize=11)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=11)
    ax.set_title(f"Importance Curve by Position\n({metric})", fontsize=12)
    ax.legend(fontsize=10)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Plot 2: Mean importance per language (bar chart)
# ---------------------------------------------------------------------------

def plot_mean_importance_by_language(
    lang_chunks: dict[str, list[dict]],
    metrics: list[str],
    output_path: Path,
):
    """Bar chart: mean importance (multiple metrics) across languages."""
    languages = [l for l in lang_chunks if lang_chunks[l]]
    if not languages:
        print("  [skip] No data for mean importance bar chart")
        return

    x = np.arange(len(languages))
    width = 0.8 / max(len(metrics), 1)

    fig, ax = plt.subplots(figsize=(max(6, len(languages) * 1.5), 5))

    for mi, metric in enumerate(metrics):
        means = []
        errs = []
        for lang in languages:
            vals = [
                c[metric] for c in lang_chunks[lang]
                if metric in c and c[metric] is not None and not np.isnan(c[metric])
            ]
            means.append(np.mean(vals) if vals else 0.0)
            errs.append(np.std(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0)

        offset = (mi - len(metrics) / 2 + 0.5) * width
        ax.bar(x + offset, means, width, label=metric.replace("_", " "), yerr=errs, capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels([LANGUAGE_NAMES.get(l, l) for l in languages], fontsize=11)
    ax.set_ylabel("Mean Importance", fontsize=11)
    ax.set_title("Mean Importance by Language and Metric", fontsize=12)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Plot 3: Importance by function tag (if DAG labels exist)
# ---------------------------------------------------------------------------

def plot_importance_by_function_tag(
    lang_chunks: dict[str, list[dict]],
    metric: str,
    output_path: Path,
):
    """
    Grouped bar chart: mean importance per function tag, for each language.

    Only uses chunks that have function_tags set (from GPT-4o labeling).
    """
    # Collect tag-level importance per language
    tag_lang_vals: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    has_labels = False
    for language, chunks in lang_chunks.items():
        for c in chunks:
            tags = c.get("function_tags")
            val = c.get(metric)
            if tags and val is not None and not np.isnan(val) and tags != ["unknown"]:
                has_labels = True
                for tag in tags:
                    tag_lang_vals[tag][language].append(val)

    if not has_labels:
        print(f"  [skip] No function_tags found — run analyze_rollouts.py first (GPT-4o labeling)")
        return

    # Only show tags with data
    active_tags = [t for t in FUNCTION_TAGS if t in tag_lang_vals]
    languages = [l for l in lang_chunks if lang_chunks[l]]
    if not active_tags or not languages:
        return

    x = np.arange(len(active_tags))
    width = 0.8 / max(len(languages), 1)

    fig, ax = plt.subplots(figsize=(max(8, len(active_tags) * 1.2), 5))

    for li, language in enumerate(languages):
        means = []
        errs = []
        for tag in active_tags:
            vals = tag_lang_vals[tag].get(language, [])
            means.append(np.mean(vals) if vals else 0.0)
            errs.append(np.std(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0)

        offset = (li - len(languages) / 2 + 0.5) * width
        color = LANG_COLORS.get(language, None)
        ax.bar(
            x + offset, means, width,
            label=LANGUAGE_NAMES.get(language, language),
            yerr=errs, capsize=3, color=color,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("_", "\n") for t in active_tags], fontsize=9)
    ax.set_ylabel(f"Mean {metric.replace('_', ' ').title()}", fontsize=11)
    ax.set_title(f"Importance by Function Tag ({metric})", fontsize=12)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Plot 4: Language switch frequency vs chunk position
# ---------------------------------------------------------------------------

def plot_language_switch_frequency(
    lang_chunks: dict[str, list[dict]],
    output_path: Path,
    n_bins: int = 10,
):
    """
    For non-English languages, plot fraction of rollouts with language switch
    vs. normalized chunk position. A spike indicates where the model tends to
    switch back to English.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    has_switch_data = False

    for language, chunks in lang_chunks.items():
        if language == "en":
            continue

        # Look for language_switch_info in base solution files (from problem dirs)
        # Since chunks_labeled.json doesn't directly have this, we approximate
        # using 'different_trajectories_fraction' as a proxy for divergence.
        by_problem = defaultdict(list)
        for c in chunks:
            dtf = c.get("different_trajectories_fraction")
            if dtf is not None:
                by_problem[c.get("problem_id", "default")].append(c)

        if not by_problem:
            continue

        bin_values = defaultdict(list)
        for pid, prob_chunks in by_problem.items():
            n = len(prob_chunks)
            if n == 0:
                continue
            for i, c in enumerate(sorted(prob_chunks, key=lambda x: x.get("chunk_idx", 0))):
                norm_pos = i / max(n - 1, 1)
                bin_idx = min(int(norm_pos * n_bins), n_bins - 1)
                dtf = c.get("different_trajectories_fraction")
                if dtf is not None:
                    bin_values[bin_idx].append(dtf)

        if not bin_values:
            continue

        has_switch_data = True
        xs = sorted(bin_values.keys())
        ys = [np.mean(bin_values[b]) for b in xs]
        xs_norm = [x / (n_bins - 1) for x in xs]

        color = LANG_COLORS.get(language, None)
        label = LANGUAGE_NAMES.get(language, language)
        ax.plot(xs_norm, ys, marker="o", label=label, color=color, linewidth=2)

    if not has_switch_data:
        print("  [skip] No different_trajectories_fraction data for language switch plot")
        plt.close(fig)
        return

    ax.set_xlabel("Normalized Chunk Position", fontsize=11)
    ax.set_ylabel("Different Trajectories Fraction", fontsize=11)
    ax.set_title("Trajectory Divergence by Position\n(proxy for reasoning sensitivity)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Plot 5: Accuracy at each chunk position
# ---------------------------------------------------------------------------

def plot_accuracy_by_position(
    lang_chunks: dict[str, list[dict]],
    output_path: Path,
    n_bins: int = 10,
):
    """Plot mean rollout accuracy vs. normalized chunk position per language."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for language, chunks in lang_chunks.items():
        if not chunks:
            continue

        by_problem = defaultdict(list)
        for c in chunks:
            if "accuracy" in c and c["accuracy"] is not None:
                by_problem[c.get("problem_id", "default")].append(c)

        bin_values = defaultdict(list)
        for pid, prob_chunks in by_problem.items():
            n = len(prob_chunks)
            for i, c in enumerate(sorted(prob_chunks, key=lambda x: x.get("chunk_idx", 0))):
                norm_pos = i / max(n - 1, 1)
                bin_idx = min(int(norm_pos * n_bins), n_bins - 1)
                acc = c.get("accuracy")
                if acc is not None:
                    bin_values[bin_idx].append(acc)

        if not bin_values:
            continue

        xs = sorted(bin_values.keys())
        ys = [np.mean(bin_values[b]) for b in xs]
        xs_norm = [x / (n_bins - 1) for x in xs]

        color = LANG_COLORS.get(language, None)
        ax.plot(xs_norm, ys, marker="s", label=LANGUAGE_NAMES.get(language, language),
                color=color, linewidth=2, markersize=5)

    ax.set_xlabel("Normalized Chunk Position", fontsize=11)
    ax.set_ylabel("Rollout Accuracy", fontsize=11)
    ax.set_title("Rollout Accuracy by Chunk Position\n(removing chunk i → how often correct?)", fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Plot 6: Cross-language alignment importance scatter
# ---------------------------------------------------------------------------

TAG_COLORS = {tag: plt.cm.tab10(i / 9) for i, tag in enumerate(FUNCTION_TAGS)}


def load_alignment_files(rollouts_dir: Path, lang1: str, lang2: str) -> List[Path]:
    """
    Find all alignment_{lang1}_{lang2}.json files under the rollouts directory.

    Globs: rollouts_dir/{lang1}/correct_base_solution/*/alignment_{lang1}_{lang2}.json
    """
    pattern = rollouts_dir / lang1 / "correct_base_solution" / "*" / f"alignment_{lang1}_{lang2}.json"
    return sorted(pattern.parent.parent.parent.parent.glob(
        f"{lang1}/correct_base_solution/*/alignment_{lang1}_{lang2}.json"
    ))


def plot_alignment_importance_scatter(
    alignment_files: List[Path],
    lang1: str,
    lang2: str,
    metric: str,
    output_path: Path,
) -> None:
    """
    Scatter plot: x=importance_lang1, y=importance_lang2 per aligned pair.

    Color encodes function_tag_lang1, marker size ∝ cosine similarity.
    Includes a y=x reference line and a legend for tags.
    """
    xs, ys, colors, sizes, labels = [], [], [], [], []

    for afile in alignment_files:
        try:
            with open(afile, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"  [warn] Could not load {afile}: {e}")
            continue

        for pair in data.get("aligned_pairs", []):
            x = pair.get(f"{metric}_lang1")
            y = pair.get(f"{metric}_lang2")
            if x is None or y is None:
                continue
            try:
                x, y = float(x), float(y)
            except (TypeError, ValueError):
                continue

            tags = pair.get("function_tags_lang1") or ["unknown"]
            tag = tags[0] if tags else "unknown"
            sim = float(pair.get("similarity", 0.5))

            xs.append(x)
            ys.append(y)
            colors.append(TAG_COLORS.get(tag, TAG_COLORS["unknown"]))
            sizes.append(20 + 120 * sim)
            labels.append(tag)

    if not xs:
        print(f"  [skip] No alignment data for scatter plot {lang1}/{lang2}/{metric}")
        return

    fig, ax = plt.subplots(figsize=(7, 6))

    # Reference y=x line
    lo = min(min(xs), min(ys))
    hi = max(max(xs), max(ys))
    ax.plot([lo, hi], [lo, hi], "--", color="gray", linewidth=1, alpha=0.6, label="_nolegend_")

    # Plot each unique tag separately for legend
    seen_tags = {}
    for x, y, c, s, tag in zip(xs, ys, colors, sizes, labels):
        sc = ax.scatter(x, y, c=[c], s=s, alpha=0.7, edgecolors="none")
        if tag not in seen_tags:
            seen_tags[tag] = sc

    # Build legend handles from unique tags
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=TAG_COLORS.get(t, TAG_COLORS["unknown"]),
               markersize=8, label=t.replace("_", " "))
        for t in FUNCTION_TAGS if t in set(labels)
    ]
    ax.legend(handles=handles, fontsize=8, loc="upper left", framealpha=0.7)

    l1_name = LANGUAGE_NAMES.get(lang1, lang1)
    l2_name = LANGUAGE_NAMES.get(lang2, lang2)
    ax.set_xlabel(f"{metric.replace('_', ' ').title()} ({l1_name})", fontsize=11)
    ax.set_ylabel(f"{metric.replace('_', ' ').title()} ({l2_name})", fontsize=11)
    ax.set_title(
        f"Cross-Language Importance Alignment\n{l1_name} vs {l2_name} — {metric}",
        fontsize=12,
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate multilingual CoT analysis plots")
    parser.add_argument("--dataset", type=str, required=True, choices=["mgsm", "mmath", "mmmlu"])
    parser.add_argument("--languages", type=str, default="en,fr,zh,ar",
                        help="Comma-separated language codes")
    parser.add_argument("--model", type=str, default="Llama-3.2-3B-Instruct-Turbo",
                        help="Model subdirectory name (e.g. Llama-3.2-3B-Instruct-Turbo)")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--rollouts_base", type=str, default="multicot",
                        help="Base rollouts directory (default: multicot)")
    parser.add_argument("--output_dir", type=str, default="multicot/figures",
                        help="Output directory for figures")
    parser.add_argument("--metric", type=str,
                        default="counterfactual_importance_accuracy",
                        choices=IMPORTANCE_METRICS,
                        help="Primary importance metric for single-metric plots")
    args = parser.parse_args()

    languages = [l.strip() for l in args.languages.split(",")]
    output_dir = Path(args.output_dir) / args.dataset / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build base rollouts path
    rollouts_dir = (
        Path(args.rollouts_base) / args.dataset / args.model
        / f"temperature_{args.temperature}_top_p_{args.top_p}"
    )

    if not rollouts_dir.exists():
        print(f"Rollouts directory not found: {rollouts_dir}")
        print("Run generate_rollouts.py and analyze_rollouts.py first.")
        return

    print(f"\nLoading data from: {rollouts_dir}")
    print(f"Languages: {languages}")

    # Load chunks for each language
    lang_chunks: dict[str, list[dict]] = {}
    for language in languages:
        chunks = load_language_chunks(rollouts_dir, language)
        lang_chunks[language] = chunks
        print(f"  {language}: {len(chunks)} chunks loaded")

    total = sum(len(v) for v in lang_chunks.values())
    if total == 0:
        print("\nNo labeled chunks found. Run analyze_rollouts.py first to compute importance metrics.")
        return

    print(f"\nGenerating plots in: {output_dir}")

    # Plot 1: Importance curve (primary metric)
    plot_importance_curves(
        lang_chunks,
        metric=args.metric,
        output_path=output_dir / f"importance_curve_{args.metric}.png",
    )

    # Plot 2: Mean importance bar chart (key metrics)
    key_metrics = [
        m for m in [
            "counterfactual_importance_accuracy",
            "resampling_importance_accuracy",
            "forced_importance_accuracy",
        ]
        if any(args.metric == m or any(m in c for c in lang_chunks[l]) for l in languages)
    ]
    if not key_metrics:
        key_metrics = [args.metric]
    plot_mean_importance_by_language(
        lang_chunks,
        metrics=key_metrics,
        output_path=output_dir / "mean_importance_by_language.png",
    )

    # Plot 3: Importance by function tag (if labels available)
    plot_importance_by_function_tag(
        lang_chunks,
        metric=args.metric,
        output_path=output_dir / f"importance_by_tag_{args.metric}.png",
    )

    # Plot 4: Trajectory divergence (different_trajectories_fraction)
    plot_language_switch_frequency(
        lang_chunks,
        output_path=output_dir / "trajectory_divergence_by_position.png",
    )

    # Plot 5: Rollout accuracy by position
    plot_accuracy_by_position(
        lang_chunks,
        output_path=output_dir / "accuracy_by_position.png",
    )

    # Plot 6: Cross-language importance scatter (per language pair)
    lang_pairs = [
        (languages[i], languages[j])
        for i in range(len(languages))
        for j in range(i + 1, len(languages))
    ]
    for l1, l2 in lang_pairs:
        afiles = load_alignment_files(rollouts_dir, l1, l2)
        if afiles:
            plot_alignment_importance_scatter(
                afiles, l1, l2, metric=args.metric,
                output_path=output_dir / f"alignment_scatter_{l1}_{l2}_{args.metric}.png",
            )

    print(f"\nDone. All figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
