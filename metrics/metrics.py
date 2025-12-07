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

    # Add mutually exclusive group for commands
    command_group = parser.add_mutually_exclusive_group(required=True)
    command_group.add_argument(
        "--dockq", 
        action="store_true",
        help="Calculate DockQ score"
    )
    command_group.add_argument(
        "--correlation", 
        action="store_true",
        help="Compute correlations"
    )

    # DockQ arguments
    parser.add_argument(
        "--data.models",
        help="Path to model tar file (e.g., AFm.tar).",
    )
    parser.add_argument(
        "--data.natives",
        help="Path to native tar file (e.g., AFm.natives.tar).",
    )
    
    # Correlation arguments
    parser.add_argument(
        "--data.dockq",
        help="Path to model data table (CSV) with features and 'dockq' column.",
    )
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
    parser.add_argument(
        "--methods",
        default="all,rank001,best_dockq",
        help="Comma-separated list of correlation methods to use (Available: all,rank001,best_dockq).",
    )
    
    # Common arguments
    parser.add_argument(
        "--output_dir",
        help="Path to write output file.",
    )
    parser.add_argument(
        "--name",
        default="AFMultimer",
        help="Name of the modeling method (default: AFMultimer).",
    )

    return parser

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.dockq:
        # Validate required arguments for dockq
        if not getattr(args, 'data.models') or not getattr(args, 'data.natives') or not args.output_dir:
            parser.error("--dockq requires --data.models, --data.natives, and --output_dir")    
            
        from metrics.dockqcal import main as dockq_main
        
        original_argv = sys.argv
        sys.argv = ['-dockq', '--data.models', getattr(args, 'data.models'), '--data.natives', getattr(args, 'data.natives'), '--output_dir', args.output_dir, '--name', args.name]
        try:
            dockq_main()
        finally:
            sys.argv = original_argv
            
    elif args.correlation:
        # Validate required arguments for correlation
        if not getattr(args, 'data.dockq') or not getattr(args, 'output_dir'):
            parser.error("--correlation requires --data.dockq and --output_dir")

        try:
            from metrics.correlation import main as corr_main
            
            original_argv = sys.argv
            sys.argv = ['--correlation', '--data.dockq', getattr(args, 'data.dockq'), '--output_dir', getattr(args, 'output_dir'),
                       '--metrics', args.metrics, '--features', args.features, '--methods', args.methods, '--name', args.name]
            try:
                corr_main()
            finally:
                sys.argv = original_argv
        except Exception as e:
            print(f"Error occurred while computing correlations: {e}")

if __name__ == "__main__":
    main()