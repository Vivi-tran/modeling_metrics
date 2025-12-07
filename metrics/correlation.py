#!/usr/bin/env python
from __future__ import annotations

import os
import argparse
from typing import List, Literal, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def compute_correlations(
    df: pd.DataFrame,
    metrics: Sequence[str] = ("pearson", "spearman"),
    features: Sequence[str] = ("ptm", "plddt", "iptm"),
) -> pd.DataFrame:
    """
    Compute Pearson/Spearman correlation between each feature and target_col.
    """
    target_col = "dockq"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not in DataFrame.")
    if features is None:
        features = ["ptm", "plddt", "iptm"]
    y = df[target_col].astype(float).values

    rows = []
    for feat in features:
        x = df[feat].astype(float).values
        
        # Check if feature has constant values
        if len(np.unique(x)) <= 1:
            print(f"Warning: Feature '{feat}' has constant values. Correlation undefined.")
            # Add NaN results for constant features
            if "pearson" in metrics:
                rows.append({
                    "feature": feat,
                    "metric": "pearson", 
                    "r": np.nan,
                    "p_value": np.nan,
                })
            if "spearman" in metrics:
                rows.append({
                    "feature": feat,
                    "metric": "spearman",
                    "r": np.nan, 
                    "p_value": np.nan,
                })
            continue

        if "pearson" in metrics:
            try:
                r_p, p_p = pearsonr(x, y)
                rows.append({
                    "feature": feat,
                    "metric": "pearson",
                    "r": round(float(r_p), 3),
                    "p_value": round(float(p_p), 3),
                })
            except Exception as e:
                print(f"Error computing Pearson correlation for {feat}: {e}")
                
        if "spearman" in metrics:
            try:
                r_s, p_s = spearmanr(x, y)
                rows.append({
                    "feature": feat,
                    "metric": "spearman",
                    "r": round(float(r_s), 3),
                    "p_value": round(float(p_s), 3),
                })
            except Exception as e:
                print(f"Error computing Spearman correlation for {feat}: {e}")

    return pd.DataFrame(rows).sort_values(
        ["feature", "metric", "r"], ascending=[True, True, False]
    )

def correlation(
    df: pd.DataFrame,
    methods: Sequence[str] = ("all", "rank1", "best_dockq"),
    metrics: Sequence[str] = ("pearson", "spearman"),
    features: Sequence[str] = ("ptm", "plddt", "iptm"),
) -> pd.DataFrame:

    all_results = []

    for method_name in methods:
        if method_name == "all":
            method_df = df
        elif method_name == "rank1":
            method_df = df[df["rank"] == 1]
        elif method_name == "best_dockq":
            idx_best = df.groupby("id")["dockq"].idxmax()
            method_df = df.loc[idx_best]
        else:
            continue  # Skip unknown methods

        # Check if we have enough data for correlation
        if len(method_df) <= 2:
            print(
                f"Warning: Method '{method_name}' resulted in only {len(method_df)} rows. Skipping correlation computation."
            )
            continue

        # Check if we have the required columns
        missing_features = [f for f in features if f not in method_df.columns]
        if missing_features:
            print(
                f"Warning: Method '{method_name}' missing features: {missing_features}. Skipping."
            )
            continue

        print(
            f"Computing correlations for method '{method_name}' with {len(method_df)} rows"
        )

        # Compute correlations for this method
        try:
            results = compute_correlations(
                df=method_df, metrics=metrics, features=features
            )

            # Add method column to identify which method was used
            results["method"] = method_name

            all_results.append(results)
        except Exception as e:
            print(f"Error computing correlations for method '{method_name}': {e}")
            continue

    # Combine all results
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        cols = ["method"] + [col for col in final_df.columns if col != "method"]
        return final_df[cols]
    else:
        print("No valid correlation results computed.")
        return pd.DataFrame()


def build_correlation_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute Pearson/Spearman correlation vs DockQ."
    )

    parser.add_argument(
        "--metrics.dockq",
        required=True,
        help="Path to model data table (CSV) with features and 'dockq' column.",
    )

    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output path for correlation table (CSV/TSV decided by extension).",
    )

    # Accept comma-separated string
    parser.add_argument(
        "--metrics",
        default="pearson,spearman",
        help="Comma-separated list of correlation metrics to compute (Available: pearson,spearman).",
    )

    parser.add_argument(
        "--features",
        default="ptm,plddt,iptm",
        help="Comma-separated list of feature columns (Available: ptm,plddt,iptm).",
    )

    # Accept comma-separated string
    parser.add_argument(
        "--methods",
        default="all,rank1,best_dockq",
        help="Comma-separated list of correlation methods to use (Available: all,rank1,best_dockq).",
    )

    parser.add_argument(
        "--name",
        default="AFMultimer",
        help="Name of the modeling method (default: AFMultimer).",
    )

    return parser


def main() -> None:
    parser = build_correlation_parser()
    args = parser.parse_args()


    # Load main data
    data_path = getattr(args, 'metrics.dockq')
    output_path = getattr(args, 'output_dir')
    name = getattr(args, 'name')
    df = pd.read_csv(data_path)

    metrics = [metric.strip() for metric in args.metrics.split(",")]
    methods = [method.strip() for method in args.methods.split(",")]
    features = [feat.strip() for feat in args.features.split(",")]

    corr_df = correlation(
        df=df,
        methods=methods,
        metrics=metrics,
        features=features,
    )
    log_path = os.path.join(os.getcwd(), f"{name}_correlation.log")
    with open(log_path, 'w') as f:
        f.write(f"Correlation results saved to {os.path.join(args.output_dir, f'{name}.correlation.csv')}\n")
        f.write(f"Results: {corr_df.shape}\n")

    corr_df.to_csv(os.path.join(args.output_dir, f"{name}.correlation.csv"), index=False)

if __name__ == "__main__":
    main()
