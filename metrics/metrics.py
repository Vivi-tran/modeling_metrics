import sys, os
import tarfile
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
import argparse
import pandas as pd
from typing import List


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute DockQ as ground-truth and correlation between model metrics and ground-truth data."
    )

    subparsers = parser.add_subparsers(
        dest="command", help="Available commands", required=True
    )

    # DockQ subparser - manually define arguments
    dockq_subparser = subparsers.add_parser("dockq", help="Calculate DockQ score")
    dockq_subparser.add_argument(
        "--data.models",
        required=True,
        help="Path to model tar file (e.g., AFm.tar).",
    )
    dockq_subparser.add_argument(
        "--data.natives",
        required=True,
        help="Path to native tar file (e.g., AFm.natives.tar).",
    )
    dockq_subparser.add_argument(
        "--output",
        required=True,
        help="Path to write DockQ values added to the metadata table.",
    )
    dockq_subparser.add_argument(
        "--name",
        default="AFMultimer",
        help="Name of the modeling method (default: AFMultimer).",
    )

    # Correlation subparser - manually define arguments
    corr_subparser = subparsers.add_parser("correlation", help="Compute correlations")
    corr_subparser.add_argument(
        "--data.dockq",
        required=True,
        help="Path to model data table (CSV) with features and 'dockq' column.",
    )
    corr_subparser.add_argument(
        "--data.correlation",
        required=True,
        help="Output path for correlation table (CSV/TSV decided by extension).",
    )
    corr_subparser.add_argument(
        "--metrics",
        default="pearson,spearman",
        help="Comma-separated list of correlation metrics to compute (Available: pearson,spearman).",
    )
    corr_subparser.add_argument(
        "--features",
        default="ptm,plddt,iptm",
        help="Comma-separated list of feature columns (Available: ptm,plddt,iptm).",
    )
    corr_subparser.add_argument(
        "--methods",
        default="all,rank001,best_dockq",
        help="Comma-separated list of correlation methods to use (Available: all,rank001,best_dockq).",
    )
    corr_subparser.add_argument(
        "--name",
        default="AFMultimer",
        help="Name of the modeling method (default: AFMultimer).",
    )

    return parser

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "dockq":
        from metrics.dockqcal import main as dockq_main
        
        original_argv = sys.argv
        sys.argv = ['dockq', '--data.models', getattr(args, 'data.models'), '--data.natives', getattr(args, 'data.natives'), '--output', args.output, '--name', args.name]
        try:
            dockq_main()
        finally:
            sys.argv = original_argv
            
    elif args.command == "correlation":
        try:
            from metrics.correlation import main as corr_main
            
            original_argv = sys.argv
            sys.argv = ['correlation', '--data.dockq', getattr(args, 'data.dockq'), '--data.correlation', getattr(args, 'data.correlation'), 
                       '--metrics', args.metrics, '--features', args.features, '--methods', args.methods, '--name', args.name]
            try:
                corr_main()
            finally:
                sys.argv = original_argv
        except Exception as e:
            print(f"Error occurred while computing correlations: {e}")

if __name__ == "__main__":
    main()