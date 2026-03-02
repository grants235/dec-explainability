"""
Statistical significance testing for the teleological XAI paper.

Tests performed
---------------
- Wilcoxon signed-rank test: purposive method vs each baseline on each metric
- Bootstrap 95 % confidence intervals for each method/metric
- Effect size (Cohen's d, rank-biserial r)

Usage
-----
    python experiments/analysis/statistical_tests.py \\
        --results-csv  results/image/image_results.csv \\
        --output-dir   results/stats/

Output
------
  - ``results/stats/wilcoxon_tests.csv``   – p-values, effect sizes
  - ``results/stats/bootstrap_ci.csv``     – 95 % CI per method/metric
  - ``results/stats/stats_report.txt``     – human-readable summary
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import wilcoxon, norm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

METRIC_COLUMNS: List[str] = [
    "pbpa",
    "deletion_auc",
    "insertion_auc",
    "purposive_specificity",
]

# For deletion AUC, *lower* is better; mark so effect-size sign is correct
LOWER_IS_BETTER: Dict[str, bool] = {
    "pbpa":                   False,
    "deletion_auc":           True,
    "insertion_auc":          False,
    "purposive_specificity":  False,
}

BASELINE_METHODS: List[str] = ["ig", "gradcam", "shap"]
TARGET_METHOD: str = "purposive"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_per_image_results(csv_path: str) -> Dict[str, Dict[str, List[float]]]:
    """
    Load per-image results from the flat CSV produced by image_eval.py.

    Returns
    -------
    dict
        method -> metric_name -> list of float values (one per image).
    """
    results: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Results CSV not found: {csv_path}")

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = row.get("method", "").strip()
            if not method:
                continue
            for col in METRIC_COLUMNS:
                val_str = row.get(col, "").strip()
                if val_str and val_str.lower() not in ("nan", "none", ""):
                    try:
                        results[method][col].append(float(val_str))
                    except ValueError:
                        pass

    return {m: dict(d) for m, d in results.items()}


# ---------------------------------------------------------------------------
# Wilcoxon signed-rank test
# ---------------------------------------------------------------------------

def wilcoxon_test_purposive_vs_baselines(
    per_image_results: Dict[str, Dict[str, List[float]]],
) -> List[Dict[str, Any]]:
    """
    Run paired Wilcoxon signed-rank tests comparing *purposive* vs each
    baseline on each metric.

    The test requires that both methods have scores for the *same* images.
    We pair on shared image indices (aligned by list position).

    Parameters
    ----------
    per_image_results : dict
        method -> metric_name -> list of per-image floats.

    Returns
    -------
    list of dict, each row:
        baseline, metric, n_pairs, statistic, p_value, effect_size_r,
        cohens_d, purposive_better (bool)
    """
    rows: List[Dict[str, Any]] = []

    purposive_data = per_image_results.get(TARGET_METHOD, {})
    if not purposive_data:
        logger.warning("No data found for method='%s'", TARGET_METHOD)
        return rows

    for baseline in BASELINE_METHODS:
        baseline_data = per_image_results.get(baseline, {})
        if not baseline_data:
            logger.warning("No data for baseline='%s'; skipping.", baseline)
            continue

        for metric in METRIC_COLUMNS:
            p_vals = purposive_data.get(metric, [])
            b_vals = baseline_data.get(metric, [])

            n = min(len(p_vals), len(b_vals))
            if n < 5:
                logger.warning(
                    "Too few paired samples (n=%d) for %s vs %s on %s; skipping.",
                    n, TARGET_METHOD, baseline, metric,
                )
                continue

            p_arr = np.array(p_vals[:n], dtype=np.float64)
            b_arr = np.array(b_vals[:n], dtype=np.float64)

            # Remove NaNs
            valid = np.isfinite(p_arr) & np.isfinite(b_arr)
            p_arr = p_arr[valid]
            b_arr = b_arr[valid]
            n_valid = len(p_arr)

            if n_valid < 5:
                continue

            differences = p_arr - b_arr

            # For metrics where lower is better, flip sign so positive
            # difference means purposive is better.
            if LOWER_IS_BETTER.get(metric, False):
                differences = -differences

            try:
                stat, pval = wilcoxon(differences, alternative="greater", zero_method="wilcox")
            except ValueError:
                # All differences zero
                stat, pval = 0.0, 1.0

            # Effect size: rank-biserial correlation  r = 1 - 2*W / (n*(n+1)/2)
            n_n = n_valid * (n_valid + 1) / 2
            r_effect = float(1.0 - 2.0 * stat / n_n) if n_n > 0 else 0.0

            # Cohen's d
            diff_std = float(np.std(differences, ddof=1))
            cohens_d = float(np.mean(differences) / diff_std) if diff_std > 0 else 0.0

            purposive_better = bool(pval < 0.05)

            rows.append({
                "baseline":          baseline,
                "metric":            metric,
                "n_pairs":           n_valid,
                "statistic":         float(stat),
                "p_value":           float(pval),
                "effect_size_r":     float(r_effect),
                "cohens_d":          float(cohens_d),
                "purposive_better":  purposive_better,
            })

    return rows


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def bootstrap_confidence_intervals(
    per_image_results: Dict[str, Dict[str, List[float]]],
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Compute bootstrap confidence intervals for the mean of each method/metric.

    Parameters
    ----------
    per_image_results : dict
    n_bootstrap : int
        Number of bootstrap resamples (default 2000).
    confidence : float
        CI level (default 0.95 -> 95 %).
    seed : int

    Returns
    -------
    list of dicts with keys:
        method, metric, mean, ci_lower, ci_upper, n
    """
    rng = np.random.default_rng(seed)
    alpha = 1.0 - confidence
    rows: List[Dict[str, Any]] = []

    for method, metrics in per_image_results.items():
        for metric, vals in metrics.items():
            arr = np.array([v for v in vals if np.isfinite(v)], dtype=np.float64)
            n = len(arr)
            if n < 2:
                continue

            observed_mean = float(np.mean(arr))
            boot_means = np.array([
                np.mean(rng.choice(arr, size=n, replace=True))
                for _ in range(n_bootstrap)
            ])

            ci_lo = float(np.percentile(boot_means, 100 * alpha / 2))
            ci_hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))

            rows.append({
                "method":    method,
                "metric":    metric,
                "mean":      observed_mean,
                "ci_lower":  ci_lo,
                "ci_upper":  ci_hi,
                "n":         n,
            })

    return rows


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _write_wilcoxon_csv(rows: List[Dict[str, Any]], out_dir: str) -> None:
    path = os.path.join(out_dir, "wilcoxon_tests.csv")
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Saved Wilcoxon tests to %s", path)


def _write_bootstrap_csv(rows: List[Dict[str, Any]], out_dir: str) -> None:
    path = os.path.join(out_dir, "bootstrap_ci.csv")
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Saved bootstrap CIs to %s", path)


def _write_text_report(
    wilcoxon_rows: List[Dict[str, Any]],
    bootstrap_rows: List[Dict[str, Any]],
    out_dir: str,
) -> None:
    path = os.path.join(out_dir, "stats_report.txt")
    lines: List[str] = [
        "=" * 72,
        "TELEOLOGICAL XAI – STATISTICAL SIGNIFICANCE REPORT",
        "=" * 72,
        "",
        "WILCOXON SIGNED-RANK TESTS  (Purposive vs. Baselines)",
        "-" * 72,
        f"{'Baseline':<20} {'Metric':<28} {'n':>5} {'p-value':>10} {'r':>8} {'d':>8} {'Better':>7}",
        "-" * 72,
    ]
    for r in wilcoxon_rows:
        lines.append(
            f"{r['baseline']:<20} {r['metric']:<28} {r['n_pairs']:>5} "
            f"{r['p_value']:>10.4f} {r['effect_size_r']:>8.3f} "
            f"{r['cohens_d']:>8.3f} {'YES' if r['purposive_better'] else 'NO':>7}"
        )

    lines += [
        "",
        "BOOTSTRAP 95% CONFIDENCE INTERVALS",
        "-" * 72,
        f"{'Method':<20} {'Metric':<28} {'n':>5} {'Mean':>8} {'95% CI':>20}",
        "-" * 72,
    ]
    for r in bootstrap_rows:
        ci_str = f"[{r['ci_lower']:.4f}, {r['ci_upper']:.4f}]"
        lines.append(
            f"{r['method']:<20} {r['metric']:<28} {r['n']:>5} "
            f"{r['mean']:>8.4f} {ci_str:>20}"
        )
    lines += ["", "=" * 72]

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("Saved text report to %s", path)


def _write_latex_stats(
    wilcoxon_rows: List[Dict[str, Any]],
    out_dir: str,
) -> None:
    """Generate a LaTeX table of Wilcoxon p-values."""
    path = os.path.join(out_dir, "wilcoxon_table.tex")

    # Pivot: rows = baselines, columns = metrics
    baselines = BASELINE_METHODS
    metrics    = METRIC_COLUMNS

    # Build lookup: (baseline, metric) -> row
    lookup: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for r in wilcoxon_rows:
        lookup[(r["baseline"], r["metric"])] = r

    metric_labels = {
        "pbpa":                  "PBPA",
        "deletion_auc":          "Del AUC",
        "insertion_auc":         "Ins AUC",
        "purposive_specificity": "PS",
    }
    baseline_labels = {
        "ig":      "IG",
        "gradcam": "GradCAM",
        "shap":    "SHAP",
    }

    col_header = " & ".join(metric_labels.get(m, m) for m in metrics)
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Wilcoxon signed-rank $p$-values: Purposive vs.\ baselines "
        r"(one-sided, purposive $>$ baseline). $^*p<.05$; $^{**}p<.01$; $^{***}p<.001$.}",
        r"\label{tab:wilcoxon}",
        r"\begin{tabular}{l" + "c" * len(metrics) + "}",
        r"\toprule",
        "Baseline & " + col_header + r" \\",
        r"\midrule",
    ]

    for bl in baselines:
        cells = [baseline_labels.get(bl, bl)]
        for m in metrics:
            r = lookup.get((bl, m))
            if r is None:
                cells.append("--")
                continue
            p = r["p_value"]
            stars = ""
            if p < 0.001:
                stars = "$^{***}$"
            elif p < 0.01:
                stars = "$^{**}$"
            elif p < 0.05:
                stars = "$^{*}$"
            cells.append(f"{p:.3f}{stars}")
        lines.append(" & ".join(cells) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("Saved Wilcoxon LaTeX table to %s", path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Statistical significance tests for teleological XAI."
    )
    parser.add_argument(
        "--results-csv",
        default="results/image/image_results.csv",
        help="Per-image results CSV from image_eval.py",
    )
    parser.add_argument(
        "--output-dir",
        default="results/stats/",
        help="Directory to write statistical outputs",
    )
    parser.add_argument(
        "--n-bootstrap", type=int, default=2000,
        help="Number of bootstrap resamples for CIs",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for bootstrap",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Load data --------------------------------------------------------
    try:
        data = load_per_image_results(args.results_csv)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    # ---- Wilcoxon tests ---------------------------------------------------
    logger.info("Running Wilcoxon tests...")
    wilcoxon_rows = wilcoxon_test_purposive_vs_baselines(data)

    # ---- Bootstrap CIs ----------------------------------------------------
    logger.info("Computing bootstrap confidence intervals (n=%d)...", args.n_bootstrap)
    bootstrap_rows = bootstrap_confidence_intervals(
        data, n_bootstrap=args.n_bootstrap, seed=args.seed
    )

    # ---- Save outputs -----------------------------------------------------
    _write_wilcoxon_csv(wilcoxon_rows, args.output_dir)
    _write_bootstrap_csv(bootstrap_rows, args.output_dir)
    _write_text_report(wilcoxon_rows, bootstrap_rows, args.output_dir)
    _write_latex_stats(wilcoxon_rows, args.output_dir)

    # ---- Print summary to console -----------------------------------------
    if wilcoxon_rows:
        print("\nWilcoxon results summary:")
        print(f"{'Baseline':<12} {'Metric':<25} {'p':>8} {'Better':>7}")
        print("-" * 56)
        for r in wilcoxon_rows:
            print(
                f"{r['baseline']:<12} {r['metric']:<25} "
                f"{r['p_value']:>8.4f} {'YES' if r['purposive_better'] else 'NO':>7}"
            )
    else:
        print("No Wilcoxon results computed (insufficient data).")

    logger.info("Statistical analysis complete. Results in %s", args.output_dir)


if __name__ == "__main__":
    main()
