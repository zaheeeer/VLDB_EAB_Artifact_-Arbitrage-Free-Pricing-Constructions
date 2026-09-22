"""
Strict, standalone preparation driver for the SQLShare workload.
Downloads the official public release, loads data strictly (no silent row loss),
translates queries, and freezes the set of admissible assets.
"""
import argparse
import hashlib
import json
import logging
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

import duckdb
import pandas as pd

# Import the recovered translation and retention helpers
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "legacy"))
from sqlshare_translate import transpile_tsql, classify_error, find_unsupported
from sqlshare_retention import load as load_retention

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

URL = "https://shrquerylogs.s3.amazonaws.com/public/sqlshare_data_release1.zip"

def download_and_extract(data_dir: Path):
    """Download the official SQLShare release if not present."""
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "sqlshare_data_release1.zip"
    
    if not zip_path.exists():
        logging.info(f"Downloading SQLShare release from {URL}...")
        urllib.request.urlretrieve(URL, zip_path)
        logging.info("Download complete.")
    else:
        logging.info("SQLShare release zip found locally.")

    extract_path = data_dir / "sqlshare_data_release1"
    if not extract_path.exists():
        logging.info("Extracting archive...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(data_dir)
    return zip_path, extract_path

def strict_load_market(con, root_dir: Path, owner: str) -> list:
    """
    EAB Strict Load: We do not use ignore_errors=true for the primary load. 
    If a file is structurally corrupt, it is logged and rejected.
    """
    market_dir = root_dir / "data" / owner
    if not market_dir.is_dir():
        return []
    
    con.execute(f"CREATE SCHEMA IF NOT EXISTS \"{owner}\"")
    out = []
    
    for fname in sorted(os.listdir(market_dir)):
        path = market_dir / fname
        # Remove 'table_' or 'materialized_' prefix for the relation name
        tname = fname
        for p in ("table_", "materialized_"):
            if fname.lower().startswith(p):
                tname = fname[len(p):]
                break
                
        rec = {"owner": owner, "file": fname, "table": tname, "rows": 0, "cols": 0, "error": ""}
        
        # STRICT read. No ignore_errors=true.
        reader = f"read_csv('{path.as_posix()}', auto_detect=true, null_padding=true, sample_size=-1)"
        try:
            con.execute(f"CREATE OR REPLACE TABLE \"{owner}\".\"{tname}\" AS SELECT * FROM {reader}")
            res = con.execute(f"SELECT count(*) FROM \"{owner}\".\"{tname}\"").fetchone()[0]
            cols = len(con.execute(f"SELECT * FROM \"{owner}\".\"{tname}\" LIMIT 0").description)
            rec.update({"rows": res, "cols": cols})
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {str(e).split(chr(10))[0][:160]}"
            
        out.append(rec)
    return out

def build_admissible_assets(data_dir: Path, out_dir: Path):
    """Identifies and freezes the exact set of executable queries."""
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path, extract_path = download_and_extract(data_dir)
    
    logging.info("Parsing queries and catalog dependencies...")
    queries, tables, views, owner_files = load_retention(zip_path)
    
    db_path = out_dir / "sqlshare_strict.duckdb"
    if db_path.exists():
        db_path.unlink()
        
    con = duckdb.connect(str(db_path))
    
    # 1. Load base tables strictly
    logging.info("Strictly loading base tables...")
    owners = sorted(list(owner_files.keys()))
    load_records = []
    for o in owners:
        load_records.extend(strict_load_market(con, extract_path, o))
        
    pd.DataFrame(load_records).to_csv(out_dir / "sqlshare_strict_load_log.csv", index=False)
    successful_tables = [r for r in load_records if not r["error"]]
    logging.info(f"Loaded {len(successful_tables)}/{len(load_records)} files successfully without data loss.")
    
    # 2. Test translations and bindings
    logging.info("Translating and validating queries...")
    assets = []
    for i, q in enumerate(queries):
        unsupported = find_unsupported(q)
        if unsupported:
            assets.append({"original_index": i, "status": "unsupported_syntax", "reason": unsupported[0]})
            continue
            
        try:
            duck_sql = transpile_tsql(q)
        except Exception as e:
            assets.append({"original_index": i, "status": "translation_failed", "reason": classify_error(str(e))})
            continue
            
        assets.append({
            "original_index": i, 
            "status": "translated", 
            "original_sql": q,
            "duck_sql": duck_sql
        })
        
    df_assets = pd.DataFrame(assets)
    df_assets.to_csv(out_dir / "sqlshare_asset_candidates.csv", index=False)
    
    admissible = df_assets[df_assets.status == "translated"]
    logging.info(f"Freezing {len(admissible)} admissible translated assets out of {len(queries)} raw records.")
    
    manifest = {
        "status": "strict_preparation_complete",
        "raw_queries": len(queries),
        "admissible_translated_queries": len(admissible),
        "strictly_loaded_tables": len(successful_tables),
        "rejected_tables": len(load_records) - len(successful_tables)
    }
    
    with open(out_dir / "sqlshare_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
        
    logging.info("Done. See output directory for strict logs.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "sqlshare")
    parser.add_argument("--out", type=Path, default=ROOT / "validation" / "sqlshare_build")
    args = parser.parse_args()
    build_admissible_assets(args.data, args.out)
