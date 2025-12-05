import sys, os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
from metrics.dockqcal import run_dockq, build_dockq_parser
from metrics.correlation import correlation, build_correlation_parser
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

    # Get the standalone parsers and copy their arguments to subparsers
    dockq_parser = build_dockq_parser()
    dockq_subparser = subparsers.add_parser("dockq", help="Calculate DockQ score")
    
    # Copy arguments from the standalone parser to the subparser
    for action in dockq_parser._actions:
        if action.dest != 'help':  
            dockq_subparser._add_action(action)
    
    corr_parser = build_correlation_parser()
    corr_subparser = subparsers.add_parser("correlation", help="Compute correlations")
    
    # Copy arguments from the standalone parser to the subparser
    for action in corr_parser._actions:
        if action.dest != 'help':  
            corr_subparser._add_action(action)

    return parser

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Load main data
    if args.command == "dockq":
        from metrics.dockqcal import main as dockq_main
        dockq_main()
    elif args.command == "correlation":
        try:
            from metrics.correlation import main as corr_main
            corr_main()
        except Exception as e:
            print(f"Error occurred while computing correlations: {e}")

if __name__ == "__main__":
    main()