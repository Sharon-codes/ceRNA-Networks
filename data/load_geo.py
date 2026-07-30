"""
GEO Data Loader: Download and process circRNA/miRNA datasets with robust labeling and staging.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import GEOparse
import numpy as np
import pandas as pd
import requests

_ROOT = Path(__file__).resolve().parent.parent
import sys
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config.config import cfg

logging.basicConfig(
    level=cfg.log_level,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(cfg.logs_dir / "load_geo.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("load_geo")


def _parse_characteristics(gsm: Any) -> Dict[str, str]:
    """Parse GEO characteristics_ch1 list into a key-value dict."""
    out: Dict[str, str] = {}
    for char in gsm.metadata.get("characteristics_ch1", []):
        if isinstance(char, str) and ":" in char:
            parts = char.split(":", 1)
            if len(parts) == 2:
                k, v = parts
                out[k.strip().lower()] = v.strip()
    return out


def normalize_cancer_label(text: str, geo_id: str = "", title: str = "") -> str:
    """Map messy characteristics to standard cancer types."""
    text = str(text).lower()
    title = str(title).lower()
    full = text + " " + title
    
    # Healthy / Control detection
    # Use word boundaries and exclude 'carcinoma' if 'non-' or 'benign' is present
    if any(x in full for x in ["healthy", "normal", "non-cancer", "non cancer", "benign", "control"]):
        # Special case: 'benign breast disease' is Healthy (control group)
        # Ensure we don't accidentally match 'liver cancer' if it says 'normal liver'
        if "cancer" not in full.replace("non-cancer", "").replace("non cancer", "") or "normal" in full or "benign" in full:
             # But if it says 'colon cancer' and 'healthy', we need to be careful.
             # Usually titles like 'Normal Mucosa' or 'Healthy control' are clear.
             return "Healthy"
    if re.search(r"\bnc\b", full):
        return "Healthy"

    # Cancer types - order of specificity
    if any(x in full for x in ["colorectal", "crc", "colon", "rectal", "carcinoma"]):
        # Note: 'carcinoma' alone might be too broad, but 'tissue: Carcinoma' in GSE115513
        # paired with 'Rectum' in characteristics (which we'll pass in) should work.
        if any(x in full for x in ["colorectal", "crc", "colon", "rectal", "mucosa", "adenoma"]):
             return "CRC"
    
    if "breast" in full:
        return "Breast"
    if "lung" in full:
        return "Lung"
    if "prostate" in full:
        return "Prostate"
    if "bladder" in full:
        return "Bladder"
    if any(x in full for x in ["stomach", "gastric"]):
        return "Gastric"
    if any(x in full for x in ["liver", "hcc"]):
        return "HCC"
    if "pancreat" in full:
        return "Pancreatic"
    if "ovarian" in full:
        return "Ovarian"
    if "esophag" in full:
        return "Esophageal"

    # Fallback to Carcinoma if no organ found but Carcinoma is mentioned
    if "carcinoma" in full or "cancer" in full:
        return "Unknown Cancer"

    return "Unknown"


def normalize_stage(text: str) -> str:
    """Map stage strings (I, 1, stage i) to I, II, III, IV."""
    if not text or str(text).lower() in ("nan", "unknown", "na", "n/a"):
        return "unknown"
    t = str(text).strip().upper()
    
    if t.isdigit():
        return {"1": "I", "2": "II", "3": "III", "4": "IV"}.get(t, "unknown")
    
    # Check for Roman numerals as standalone words
    m = re.search(r"\b(IV|III|II|I)\b", t)
    if m:
        return m.group(1)
        
    m2 = re.search(r"STAGE[\s:]*([1-4])", t)
    if m2:
        return {"1": "I", "2": "II", "3": "III", "4": "IV"}.get(m2.group(1), "unknown")
        
    return "unknown"


def _extract_meta(gsm: Any, geo_id: str) -> Dict[str, str]:
    """Robustly extract labels and stage from a GSM."""
    ch = _parse_characteristics(gsm)
    title = gsm.metadata.get("title", [""])[0]
    
    # Create a full string of all characteristics for broader search
    full_ch = " ".join(ch.values())
    
    # Cancer Type
    disease_keys = ["disease state", "disease", "patient status", "diagnosis", "status", "tissue", "condition", "organ", "site_summary"]
    disease_raw = ""
    for k in disease_keys:
        if k in ch:
            disease_raw += " " + ch[k]
    
    ctype = normalize_cancer_label(disease_raw + " " + full_ch, geo_id, title)
    
    # Stage
    stage_keys = ["ajcc stage", "tumor stage", "clinical stage", "stage", "pathological stage"]
    stage_raw = ""
    for k in stage_keys:
        if k in ch:
            stage_raw = ch[k]
            break
    if not stage_raw:
        # Search all values for 'stage'
        for v in ch.values():
            if "stage" in str(v).lower():
                stage_raw = v
                break
    # Finally check title for stage
    if not stage_raw and "stage" in title.lower():
        stage_raw = title
        
    return {
        "cancer_type": ctype,
        "stage": normalize_stage(stage_raw),
        "age": ch.get("age", "unknown"),
        "sex": ch.get("gender", ch.get("sex", "unknown")),
    }


def load_mirbase_map() -> Dict[str, str]:
    """Map MIMAT accessions to names."""
    path = cfg.raw_dir / "databases" / "mature_all.fa"
    if not path.exists():
        return {}
    mapping = {}
    with open(path, "r") as f:
        for line in f:
            if line.startswith(">"):
                parts = line[1:].strip().split()
                if len(parts) >= 2:
                    name = parts[0]
                    for p in parts[1:]:
                        if p.startswith("MIMAT"):
                            mapping[p] = name
                            break
    return mapping


def parse_geo_soft(geo_id: str, dest: Path) -> GEOparse.GSE:
    soft_file = dest / f"{geo_id}_family.soft.gz"
    if soft_file.exists() and soft_file.stat().st_size == 0:
        log.warning("Removing empty/incomplete SOFT file for %s: %s", geo_id, soft_file)
        soft_file.unlink()
    if not soft_file.exists():
        log.info("Downloading %s to %s", geo_id, dest)
        dest.mkdir(parents=True, exist_ok=True)
        # Try direct download if GEOparse download fails
        try:
            gse = GEOparse.get_GEO(geo_id, destdir=str(dest), silent=True)
        except Exception:
            log.warning("GEOparse download failed for %s, trying curl...", geo_id)
            import subprocess
            url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{geo_id[:-3]}nnn/{geo_id}/soft/{geo_id}_family.soft.gz"
            subprocess.run(["curl", "-L", url, "-o", str(soft_file)], check=True)
            gse = GEOparse.get_GEO(filepath=str(soft_file), silent=True)
    else:
        gse = GEOparse.get_GEO(filepath=str(soft_file), silent=True)
    return gse


def load_circrna_gse(geo_id: str, raw_sub: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    gse = parse_geo_soft(geo_id, raw_sub)
    series: List[pd.Series] = []
    meta_rows: List[Dict[str, Any]] = []
    
    for gsm_id, gsm in gse.gsms.items():
        meta = _extract_meta(gsm, geo_id)
        meta["sample_id"] = gsm_id
        meta["dataset"] = geo_id
        meta_rows.append(meta)
        
        # Table data
        if not gsm.table.empty and "VALUE" in gsm.table.columns:
            s = gsm.table.set_index(gsm.table.columns[0])["VALUE"]
            s.name = gsm_id
            series.append(s)
            
    if series:
        counts = pd.concat(series, axis=1).fillna(0.0)
    else:
        # Check for supplementary matrix
        candidates = list(raw_sub.glob("*matrix.txt.gz")) + list(raw_sub.glob("*counts.txt.gz"))
        if candidates:
            counts = pd.read_csv(candidates[0], sep="\t", index_col=0, comment="!")
        else:
            raise FileNotFoundError(f"No data table for {geo_id}")
            
    return counts, pd.DataFrame(meta_rows).set_index("sample_id")


def load_mirna_gse(geo_id: str, raw_sub: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    counts, meta = load_circrna_gse(geo_id, raw_sub)
    # Aggressive ID cleanup for miRNA
    mir_map = load_mirbase_map()
    if mir_map:
        def _remap(x):
            ps = [p.strip() for p in str(x).split(",")]
            for p in ps:
                if p in mir_map: return mir_map[p]
            return x
        counts.index = [_remap(i) for i in counts.index]
        counts = counts.groupby(level=0).mean()
    return counts, meta


def cpm_normalize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    libsizes = df.sum(axis=0)
    return df.div(libsizes, axis=1) * 1e6


def merge_and_save():
    unique_circ = sorted(
        {
            cfg.geo_datasets["circRNA_CRC"],
            cfg.geo_datasets["circRNA_lung"],
            cfg.geo_datasets["circRNA_atlas"],
            *cfg.extra_circrna_geo,
        }
    )
    unique_mirna = sorted(
        {
            cfg.geo_datasets["miRNA_CRC"],
            cfg.geo_datasets["miRNA_multicancer"],
            *cfg.extra_mirna_geo,
        }
    )
    
    circ_counts = pd.DataFrame()
    circ_metas = []
    
    for gid in unique_circ:
        try:
            c, m = load_circrna_gse(gid, cfg.raw_dir / gid)
            if circ_counts.empty:
                circ_counts = c
            else:
                new_cols = c.columns.difference(circ_counts.columns)
                circ_counts = circ_counts.join(c[new_cols], how="outer").fillna(0.0)
            circ_metas.append(m)
        except Exception as e:
            log.warning("Skipping circRNA %s: %s", gid, e)
            
    # miRNA
    mir_counts = pd.DataFrame()
    mir_metas = []
    for gid in unique_mirna:
        try:
            c, m = load_mirna_gse(gid, cfg.raw_dir / gid)
            if mir_counts.empty:
                mir_counts = c
            else:
                new_cols = c.columns.difference(mir_counts.columns)
                mir_counts = mir_counts.join(c[new_cols], how="outer").fillna(0.0)
            mir_metas.append(m)
        except Exception as e:
            log.warning("Skipping miRNA %s: %s", gid, e)

    if mir_counts.empty:
        raise RuntimeError("No miRNA datasets loaded successfully")

    circ_cpm = cpm_normalize(circ_counts)
    mir_rpm = cpm_normalize(mir_counts)
    
    # Save
    cfg.processed_dir.mkdir(parents=True, exist_ok=True)
    circ_cpm.to_csv(cfg.processed_dir / "circRNA_counts.csv")
    mir_rpm.to_csv(cfg.processed_dir / "miRNA_counts.csv")
    
    all_meta = pd.concat(circ_metas + mir_metas, axis=0)
    all_meta = all_meta[~all_meta.index.duplicated(keep="first")]
    all_meta.to_csv(cfg.processed_dir / "metadata.csv")
    
    log.info("Processing complete. Metadata saved with %d samples.", len(all_meta))


if __name__ == "__main__":
    merge_and_save()
