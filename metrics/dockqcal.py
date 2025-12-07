#!/usr/bin/env python
from __future__ import annotations
import os
import argparse
import json
import shutil
import subprocess
from typing import Dict, Any, List
import sys
from pathlib import Path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import pandas as pd
import tarfile

def define_path(model_dir: str, native_dir: str) -> pd.DataFrame:

    model_metadata = next(Path(model_dir).glob("*.csv"), None)
    native_metadata = next(Path(native_dir).glob("*.csv"), None)
    df_model = pd.read_csv(model_metadata)
    df_native = pd.read_csv(native_metadata)

    ids = df_model["id"].tolist()
    ranks = df_model["rank"].tolist()

    pdb_files = list(Path(model_dir).glob("*.pdb"))
    cif_files = list(Path(model_dir).glob("*.cif"))
    
    if pdb_files:
        format = "pdb"
    elif cif_files:
        format = "cif"
    else:
        raise FileNotFoundError(f"No PDB or CIF files found in {model_dir}")

    model_paths = []
    native_paths = []
    json_paths = []
    os.makedirs(os.path.join(model_dir, "tmp"), exist_ok=True)

    for id, rank in zip(ids, ranks):
        rank = str(rank)
        model_path = os.path.join(model_dir, f"{id}_{rank}.{format}")
        native_path = os.path.join(native_dir, f"{id}.pdb")
        json_path = os.path.join(model_dir, "tmp", f"{id}_{rank}.json")
        model_paths.append(model_path)
        native_paths.append(native_path)
        json_paths.append(json_path)

    df = df_model.copy()
    df[["model_path", "native_path", "json_path"]] = pd.DataFrame(
        list(zip(model_paths, native_paths, json_paths)),
        columns=["model_path", "native_path", "json_path"],
    )
    df = df.merge(df_native, on="id", how="right", suffixes=("_model", "_native"))
    df.to_csv(os.path.join(model_dir, "debug_dockq_input.csv"), index=False)
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
        "--data.models",
        required=True,
        help="Path to model tar file (e.g., AFm.tar).",
    )
    parser.add_argument(
        "--data.natives",
        required=True,
        help="Path to native tar file (e.g., AFm.natives.tar).",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Path to write DockQ values added to the metadata table.",
    )
    parser.add_argument(
        "--name",
        default="AFMultimer",
        help="Name of the modeling method (default: AFMultimer).",
    )

    return parser


def main() -> pd.DataFrame:
    parser = build_dockq_parser()
    args = parser.parse_args()
    model_tar = getattr(args, 'data.models')
    native_tar = getattr(args, 'data.natives')
    output_path = getattr(args, 'output_dir')
    name = getattr(args, 'name')

    with tarfile.open(model_tar, 'r') as tar:
        tar.extractall(os.path.dirname(model_tar))
    with tarfile.open(native_tar, 'r') as tar:
        tar.extractall(os.path.dirname(native_tar))

    model_dir = model_tar.replace('.tar', '')
    native_dir = os.path.join(os.path.dirname(native_tar), 'natives')
    # /home/nguyen/benchmarks/modeling/out/data/Chai-1/.c65a848c554721cacf9e59e8f1319d5df0ad6037bfecff2949df2f3f2a4ba569/Chai-1.tar
    # /home/nguyen/benchmarks/modeling/out/data/Chai-1/.c65a848c554721cacf9e59e8f1319d5df0ad6037bfecff2949df2f3f2a4ba569/Chai-1/Beta_endorphin-mu_opioid_1.cif
    df = define_path(model_dir, native_dir)
    results = []
    tmp_dir = os.path.join(model_dir, "tmp")
    log_file = os.path.join(model_dir, f"{name}_model_dir.log")
    with open(log_file, 'w') as f:
        f.write(f"Model directory: {model_dir}\n Created tmp dir at: {tmp_dir}\n")


    os.makedirs(tmp_dir, exist_ok=True)
    
    try:
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
            dockq_score = metrics.get("GlobalDockQ", None)
            if dockq_score is not None:
                with open(log_file, 'a') as f:
                    f.write(f"DockQ score for model {model_pdb} vs native {native_pdb}: {dockq_score}\n")
                dockq_score = round(dockq_score, 3)
            results.append(dockq_score)
    except Exception as e:
        with open(log_file, 'a') as f:
            f.write(f"Error during DockQ calculation: {e}\n")
        raise e
    # finally:
    #     if os.path.exists(tmp_dir):
    #         shutil.rmtree(tmp_dir)  
    
    with open(log_file, 'a') as f:
        f.write(f"DockQ calculation completed. Results for {len(results)} models.\nLength of results: {len(results)}\n")
    df_results = df.copy()
    df_results["dockq"] = results
    with open(log_file, 'a') as f:
        f.write(f"Final DataFrame shape: {df_results.shape}\n")
    df_results = df_results.drop(columns=["model_path", "native_path", "json_path"])
    output_path_parent = os.path.dirname(output_path)
    output_path_final = os.path.join(output_path_parent, f"{name}.dockq.csv")

    df_results.to_csv(output_path_final, index=False)
    return df_results


if __name__ == "__main__":
    main()
