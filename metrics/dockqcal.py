#!/usr/bin/env python
from __future__ import annotations
import os
import argparse
import json
import subprocess
from typing import Dict, Any, List

import pandas as pd

BASE = parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NATIVES_DIR = os.path.join(BASE, "data", "natives", "structures")
MODELS_DIR = os.path.join(BASE, "data", "models", "structures")


def define_path(model_metadata: str, native_metadata: str) -> str:

    df_model = pd.read_csv(model_metadata)
    df_native = pd.read_csv(native_metadata)

    ids = df_model["id"].tolist()
    ranks = df_model["rank"].tolist()
    model_paths = []
    native_paths = []
    json_paths = []
    for id, rank in zip(ids, ranks):
        rank = str(rank).zfill(3)
        model_path = os.path.join(MODELS_DIR, f"{id}-{rank}.pdb")
        native_path = os.path.join(NATIVES_DIR, f"{id}.pdb")
        json_path = os.path.join(BASE, "data", "out", f"{id}-{rank}.json")
        model_paths.append(model_path)
        native_paths.append(native_path)
        json_paths.append(json_path)
    df = df_model.copy()
    df[["model_path", "native_path", "json_path"]] = pd.DataFrame(
        list(zip(model_paths, native_paths, json_paths)),
        columns=["model_path", "native_path", "json_path"],
    )
    df = df.merge(df_native, on="id", how="left", suffixes=("_model", "_native"))
    return df


def run_dockq(
    model_pdb: str,
    native_pdb: str,
    json_output: str,
    mapping: str = "chains:chains",
):
    """
    Run the DockQ command-line program and return parsed JSON metrics.
        DockQ <model_pdb> <native_pdb> --mapping chains:chains --json json_output

    Parameters
    ----------
    model_pdb : str
        Path to model PDB file.
    native_pdb : str
        Path to model/native/reference PDB file.
    json_output : str
        Path where DockQ will write its JSON output.
    mapping : str, optional
        Chain mapping string passed to DockQ (default: "chains:chains").

    Returns
    -------
    dict
        Parsed JSON content produced by DockQ.
    """
    cmd = [
        "DockQ",
        model_pdb,
        native_pdb,
        "--mapping",
        mapping,
        "--json",
        json_output,
    ]

    result = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
    )

    # debugging
    # print(result.stdout)
    # print(result.stderr)


def parse_json(json_path: str) -> Dict[str, Any]:
    """Parse DockQ JSON output file."""
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


def build_dockq_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Wrapper to run DockQ on a model/native pair."
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Path to model metadata CSV file.",
    )
    parser.add_argument(
        "--native",
        required=True,
        help="Path to native metadata CSV file.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write DockQ values added to the metadata table.",
    )

    return parser


def main() -> pd.DataFrame:
    parser = build_dockq_parser()
    args = parser.parse_args()
    model_metadata = args.model
    native_metadata = args.native

    df = define_path(model_metadata, native_metadata)
    results = []

    for index, row in df.iterrows():
        model_pdb = row["model_path"]
        native_pdb = row["native_path"]
        native_chain = row["chains_native"]
        model_chain = row["chains_model"]
        mapping = f"{model_chain}:{native_chain}"
        json_output = row["json_path"]
        run_dockq(
            model_pdb=model_pdb,
            native_pdb=native_pdb,
            json_output=json_output,
            mapping=mapping,
        )
        metrics = parse_json(json_output)
        dockq_score = round(metrics.get("GlobalDockQ", None), 2)
        results.append(dockq_score)
    df_results = df.copy()
    df_results["dockq"] = pd.DataFrame(results, columns=["dockq"])
    df_results = df_results.drop(columns=["model_path", "native_path", "json_path"])
    df_results.to_csv(args.output, index=False)
    return df_results


if __name__ == "__main__":
    main()
