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
    df_model = df_model[df_model['id'].isin(df_native['id'])]

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

    os.makedirs(os.path.join(model_dir, "tmp"), exist_ok=True)

    # Collect all rows for the final dataframe
    all_rows = []

    for id, rank in zip(ids, ranks):
        rank = str(rank)
        model_path = os.path.join(model_dir, f"{id}_{rank}.{format}")
        
        # Find all native PDB files matching the pattern id*.pdb
        native_pattern = f"{id}*.pdb"
        native_files = list(Path(native_dir).glob(native_pattern))
        
        if not native_files:
            raise FileNotFoundError(f"No native PDB files found for {id} in {native_dir}")
        
        # Create a row for each native file pairing
        for native_file in native_files:
            native_path = str(native_file)
            # Extract version identifier from native filename (e.g., 6DDE from DAMGO-mu_opioid_6DDE.pdb)
            native_basename = native_file.stem  
            version = native_basename.split('_')[-1] 
            json_path = os.path.join(model_dir, "tmp", f"{id}_{rank}_{version}.json")
            
            # Get the corresponding row from df_model
            model_row = df_model[(df_model["id"] == id) & (df_model["rank"] == int(rank))].iloc[0].to_dict()
            
            # Add path information
            model_row["model_path"] = model_path
            model_row["native_path"] = native_path
            model_row["json_path"] = json_path
            model_row["native_pdb"] = version
            
            all_rows.append(model_row)

    # Create new dataframe from all rows
    df = pd.DataFrame(all_rows)
    
    # Merge with native metadata based on id
    df = df.merge(df_native, left_on=["id", "native_pdb"], right_on=["id", "pdb_id"], how="left", suffixes=("_model", "_native"))
    df.sort_values(by=["id","native_pdb", "rank" ], ascending=[True, True, True], inplace=True)
    df.drop(columns=["pdb_id"], inplace=True)
    # df.to_csv(os.path.join(model_dir, "debug_dockq_input.csv"), index=False)
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
    df = define_path(model_dir, native_dir)
    results = []
    tmp_dir = os.path.join(model_dir, "tmp")

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
            others = metrics.get("best_result", {})
            other_scores = others.get(native_chain, {})

            irmsd = None
            lrsd = None
            fnat = None
            
            if other_scores:
                irmsd = other_scores.get("iRMSD", None)
                lrsd = other_scores.get("LRMSD", None)
                fnat = other_scores.get("fnat", None)
            if dockq_score is not None:
                dockq_score = round(dockq_score, 3)
            if irmsd is not None:
                irmsd = round(irmsd, 3)
            if lrsd is not None:
                lrsd = round(lrsd, 3)
            if fnat is not None:
                fnat = round(fnat, 3)
            
            result_row = {
                "dockq": dockq_score,
                "irmsd": irmsd,
                "lrsd": lrsd,
                "fnat": fnat,
            }
            results.append(result_row)
    finally:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)  
    
    df_results = df.copy()
    
    # Convert results to separate columns instead of single nested column
    results_df = pd.DataFrame(results)
    for col in results_df.columns:
        df_results[col] = results_df[col]
    
    df_results = df_results.drop(columns=["model_path", "native_path", "json_path"])
    df_results.to_csv(os.path.join(args.output_dir, f"{name}.dockq.csv"), index=False)
    return df_results


if __name__ == "__main__":
    main()
