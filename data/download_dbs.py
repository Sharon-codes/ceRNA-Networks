"""
Download and cache external databases for ceRNA network construction.

Caches under ``data/raw/databases/``. Skips re-download when outputs exist.
"""

from __future__ import annotations

import gzip
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import requests
from requests.adapters import HTTPAdapter

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config.config import cfg, write_phase_metrics

logging.basicConfig(
    level=cfg.log_level,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(cfg.logs_dir / "download_dbs.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("download_dbs")

TOTAL_TASKS = 3

_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

_HTTP_SESSION: Optional[requests.Session] = None


def _robust_session() -> requests.Session:
    """HTTP session with TLS settings compatible with some legacy academic hosts.

    Python 3.12+ / OpenSSL 3 defaults can reject older cipher chains; lowering
    SECLEVEL works around ``SSLV3_ALERT_HANDSHAKE_FAILURE`` for a few servers.

    Returns:
        Cached ``requests.Session`` with a custom HTTPS adapter.
    """
    global _HTTP_SESSION
    if _HTTP_SESSION is not None:
        return _HTTP_SESSION

    sess = requests.Session()
    try:
        from urllib3.util.ssl_ import create_urllib3_context

        class _CompatTLSAdapter(HTTPAdapter):
            def init_poolmanager(self, *args, **kwargs):
                ctx = create_urllib3_context()
                for cipher_spec in ("DEFAULT:@SECLEVEL=1", "ALL:@SECLEVEL=1"):
                    try:
                        ctx.set_ciphers(cipher_spec)
                        break
                    except Exception:
                        continue
                else:
                    log.warning("Could not set relaxed cipher list; using urllib3 default context only.")
                kwargs["ssl_context"] = ctx
                return super().init_poolmanager(*args, **kwargs)

        sess.mount("https://", _CompatTLSAdapter())
    except Exception:
        log.warning("TLS compatibility adapter not available; using default session.", exc_info=True)
    sess.headers.setdefault("User-Agent", _BROWSER_UA)
    sess.headers.setdefault("Accept", "*/*")
    _HTTP_SESSION = sess
    return sess


def _download_via_curl(url: str, dest: Path, max_time_sec: int = 1800) -> bool:
    """Download ``url`` with the system ``curl`` binary (WAF / 403 workarounds).

    Some hosts return 403 to ``requests`` but allow ``curl`` with a browser UA.

    Args:
        url: Remote URL.
        dest: Output path (overwritten).
        max_time_sec: ``curl --max-time`` cap.

    Returns:
        True if ``dest`` exists and is non-empty.
    """
    curl_exe = shutil.which("curl") or shutil.which("curl.exe")
    if not curl_exe:
        log.warning("curl not found on PATH; cannot use curl download fallback.")
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [
                curl_exe,
                "-L",
                "-f",
                "-S",
                "-A",
                _BROWSER_UA,
                "-o",
                str(dest),
                "--connect-timeout",
                "120",
                "--max-time",
                str(max_time_sec),
                url,
            ],
            check=True,
        )
    except Exception:
        log.exception("curl fallback failed for %s", url)
        if dest.exists():
            dest.unlink(missing_ok=True)
        return False
    if not dest.exists() or dest.stat().st_size == 0:
        if dest.exists():
            dest.unlink(missing_ok=True)
        return False
    return True


def _phase_progress(task_index: int) -> None:
    """Log coarse download progress as a fraction of Phase 1 database tasks.

    Args:
        task_index: 1-based index of the current task (1..TOTAL_TASKS).
    """
    pct = 100.0 * task_index / TOTAL_TASKS
    log.info("Phase 1 data download progress: %.0f%% (%s/%s tasks)", pct, task_index, TOTAL_TASKS)


def download_file(url: str, dest: Path, chunk_size: int = 8192, timeout: int = 120) -> Path:
    """Stream ``url`` to ``dest`` unless the file already exists.

    Args:
        url: Remote URL.
        dest: Local filesystem path.
        chunk_size: Read chunk size in bytes.
        timeout: Request timeout in seconds.

    Returns:
        Path to ``dest`` (existing or newly written).

    Raises:
        requests.HTTPError: On non-success HTTP status after ``raise_for_status``.
        OSError: On local I/O failure.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        log.info("Skipping (already exists): %s", dest.name)
        return dest

    log.info("Downloading %s -> %s", url, dest)
    try:
        r = _robust_session().get(url, stream=True, timeout=timeout)
        try:
            if r.status_code in (401, 403):
                code = r.status_code
                log.warning(
                    "HTTP %s from server — trying curl fallback (common for automated clients).",
                    code,
                )
                r.close()
                if _download_via_curl(url, dest, max_time_sec=max(600, timeout * 5)):
                    log.info("Saved via curl: %s (%s bytes)", dest.name, dest.stat().st_size)
                    return dest
                raise requests.HTTPError(f"HTTP {code} and curl fallback failed for {url}")
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
        finally:
            r.close()
    except Exception:
        log.exception("Download failed for %s", url)
        if dest.exists() and dest.stat().st_size == 0:
            dest.unlink(missing_ok=True)
        raise
    return dest


def _normalize_mirtarbase_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map miRTarBase MTI columns to ``miRNA_id`` and ``mRNA_id``.

    Args:
        df: Raw MTI dataframe.

    Returns:
        Dataframe with at least ``miRNA_id``, ``mRNA_id`` columns.

    Raises:
        ValueError: If required columns cannot be resolved.
    """
    col_map = {c.lower().strip(): c for c in df.columns}
    mir_col = None
    for key in ("mirna", "mirna id", "mirna_id", "mature mirna"):
        if key in col_map:
            mir_col = col_map[key]
            break
    if mir_col is None:
        for c in df.columns:
            if "mirna" in c.lower():
                mir_col = c
                break
    tgt_col = None
    for key in ("target gene", "target_gene", "gene", "genes"):
        if key in col_map:
            tgt_col = col_map[key]
            break
    if tgt_col is None:
        for c in df.columns:
            if "target" in c.lower() and "gene" in c.lower():
                tgt_col = c
                break
    exp_col = None
    for key in ("experiments", "experiment", "support type", "support_type"):
        if key in col_map:
            exp_col = col_map[key]
            break
    if mir_col is None or tgt_col is None:
        raise ValueError(f"Cannot resolve miRNA/target columns from: {list(df.columns)}")
    out = df.rename(columns={mir_col: "miRNA_id", tgt_col: "mRNA_id"})
    if exp_col is not None:
        out["Experiments"] = df[exp_col]
    elif "Experiments" not in out.columns:
        out["Experiments"] = ""
    return out


def download_mirtarbase_filtered() -> Tuple[Path, int]:
    """Download miRTarBase MTI, filter strong evidence, save CSV.

    Keeps rows whose experiment field matches Luciferase reporter assay,
    Western blot, or CLIP-Seq (case-insensitive substring match).

    Returns:
        Tuple of (output_csv_path, row_count).

    Raises:
        FileNotFoundError: If the workbook cannot be read after download.
        RuntimeError: On parsing or filtering failure.
    """
    log.info("=== miRTarBase ===")
    log.info("Candidate URLs: %s", cfg.mirtarbase_urls)
    db_dir = cfg.db_subdir
    db_dir.mkdir(parents=True, exist_ok=True)
    xlsx_path = db_dir / "miRTarBase_MTI.xlsx"
    out_csv = db_dir / "mirtarbase_filtered.csv"

    if out_csv.exists():
        n = len(pd.read_csv(out_csv))
        log.info("Cached mirtarbase_filtered.csv: %s rows", n)
        return out_csv, n

    try:
        last_download_err: Optional[Exception] = None
        for mb_url in cfg.mirtarbase_urls:
            log.info("Trying miRTarBase URL: %s", mb_url)
            if xlsx_path.exists():
                try:
                    xlsx_path.unlink(missing_ok=True)
                except OSError:
                    pass
            try:
                download_file(mb_url, xlsx_path, timeout=300)
                last_download_err = None
                break
            except Exception as exc:
                last_download_err = exc
                log.warning("Download failed for this mirror: %s", mb_url, exc_info=True)
        if last_download_err is not None:
            raise last_download_err
        df_raw = pd.read_excel(xlsx_path)
        df = _normalize_mirtarbase_columns(df_raw)
        exp = df["Experiments"].astype(str)
        strong = (
            exp.str.contains("Luciferase", case=False, na=False)
            | exp.str.contains("Western blot", case=False, na=False)
            | exp.str.contains("CLIP-Seq", case=False, na=False)
            | exp.str.contains("CLIP seq", case=False, na=False)
        )
        df_f = df.loc[strong, ["miRNA_id", "mRNA_id"]].copy()
        df_f = df_f[df_f["miRNA_id"].astype(str).str.startswith("hsa-")]
        df_f["edge_weight"] = 1.0
        df_f.to_csv(out_csv, index=False)
        log.info("Saved %s (%s rows)", out_csv.name, len(df_f))
        return out_csv, len(df_f)
    except Exception:
        log.exception("miRTarBase download/processing failed")
        log.error(
            "MANUAL FALLBACK: download miRTarBase_MTI.xlsx from https://mirtarbase.cuhk.edu.cn "
            "and place at %s, then re-run this script.",
            xlsx_path,
        )
        raise


def download_circinteractome() -> Tuple[Path, int]:
    """Fetch CircInteractome circRNA–miRNA interactions or write a placeholder.

    Returns:
        Tuple of (csv_path, row_count).

    Raises:
        None: API failures produce an empty placeholder and log MANUAL FALLBACK (per spec).
    """
    log.info("=== CircInteractome ===")
    out = cfg.db_subdir / "circinteractome_interactions.csv"
    cfg.db_subdir.mkdir(parents=True, exist_ok=True)

    if out.exists():
        df0 = pd.read_csv(out)
        log.info("Cached %s: %s rows", out.name, len(df0))
        return out, len(df0)

    api = cfg.circinteractome_api_url
    log.info("Trying API: %s", api)
    try:
        resp = _robust_session().get(api, timeout=120)
        resp.raise_for_status()
        text = resp.text.strip()
        if not text:
            raise ValueError("empty API response")
        lines = [ln for ln in text.splitlines() if ln.strip()]
        rows = [ln.split("\t") for ln in lines]
        headers = rows[0]
        body = rows[1:]
        df = pd.DataFrame(body, columns=headers)
        # Normalise common column names
        lower = {c.lower(): c for c in df.columns}
        circ_col = lower.get("circrna_id") or lower.get("circrna") or list(df.columns)[0]
        mir_col = lower.get("mirna_id") or lower.get("mirna") or list(df.columns)[1]
        score_col = None
        for k in ("score", "miranda score", "energy"):
            if k in lower:
                score_col = lower[k]
                break
        df = df.rename(columns={circ_col: "circRNA_id", mir_col: "miRNA_id"})
        if score_col is not None:
            df["score"] = pd.to_numeric(df[score_col], errors="coerce").fillna(0.0)
        else:
            df["score"] = 1.0
        smin = float(df["score"].min())
        smax = float(df["score"].max())
        denom = smax - smin + 1e-9
        df["edge_weight"] = (df["score"] - smin) / denom
        df[["circRNA_id", "miRNA_id", "score", "edge_weight"]].to_csv(out, index=False)
        log.info("CircInteractome saved: %s rows", len(df))
        return out, len(df)
    except requests.exceptions.Timeout:
        log.warning("CircInteractome API timed out — writing placeholder.")
    except Exception:
        log.warning("CircInteractome API failed — writing placeholder.", exc_info=True)

    log.error(
        "MANUAL FALLBACK: visit https://circinteractome.nia.nih.gov for batch downloads "
        "or retry later; placeholder written to %s",
        out,
    )
    empty = pd.DataFrame(columns=["circRNA_id", "miRNA_id", "score", "edge_weight"])
    empty.to_csv(out, index=False)
    return out, 0


def download_mirbase_hsa_fasta() -> Path:
    """Download miRBase mature sequences (``.fa`` or ``.fa.gz``) and write human-only FASTA.

    Returns:
        Path to ``mature_hsa_only.fa``.

    Raises:
        OSError: On archive or filesystem errors.
        RuntimeError: If every configured URL fails.
    """
    log.info("=== miRBase mature sequences ===")
    db_dir = cfg.db_subdir
    db_dir.mkdir(parents=True, exist_ok=True)
    all_fa = db_dir / "mature_all.fa"
    hsa_fa = db_dir / "mature_hsa_only.fa"

    if hsa_fa.exists():
        log.info("Skipping (already exists): %s", hsa_fa.name)
        return hsa_fa

    last_err: Optional[Exception] = None
    try:
        for mb_url in cfg.mirbase_mature_fa_urls:
            log.info("Trying miRBase URL: %s", mb_url)
            suffix = ".fa.gz" if mb_url.lower().endswith(".gz") else ".fa"
            archive = db_dir / f"mature_download{suffix}"
            try:
                if archive.exists():
                    archive.unlink(missing_ok=True)
                if all_fa.exists():
                    all_fa.unlink(missing_ok=True)
                download_file(mb_url, archive, timeout=300)
                if suffix.endswith(".gz"):
                    with gzip.open(archive, "rb") as f_in, open(all_fa, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
                else:
                    shutil.copyfile(archive, all_fa)
                last_err = None
                break
            except Exception as exc:
                last_err = exc
                log.warning("miRBase mirror failed: %s", mb_url, exc_info=True)
        if last_err is not None:
            raise last_err
        with open(all_fa, encoding="utf-8", errors="ignore") as f_in, open(
            hsa_fa, "w", encoding="utf-8"
        ) as f_out:
            write_block = False
            for line in f_in:
                if line.startswith(">"):
                    write_block = "hsa-" in line or " hsa" in line.lower()
                if write_block:
                    f_out.write(line)
        log.info("Wrote %s", hsa_fa)
        return hsa_fa
    except Exception:
        log.exception("miRBase download or extraction failed")
        raise


def main() -> None:
    """Run all database download steps and write Phase 1 metrics."""
    t0 = time.time()
    log.info("Starting external database downloads (cache-safe).")
    metrics: dict = {"tasks": []}
    try:
        p, n = download_mirtarbase_filtered()
        metrics["tasks"].append({"name": "miRTarBase_filtered", "path": str(p), "rows": n})
        _phase_progress(1)

        p2, n2 = download_circinteractome()
        metrics["tasks"].append({"name": "circinteractome", "path": str(p2), "rows": n2})
        _phase_progress(2)

        p3 = download_mirbase_hsa_fasta()
        metrics["tasks"].append({"name": "mirbase_hsa_fa", "path": str(p3)})
        _phase_progress(3)

        metrics["status"] = "ok"
    except Exception:
        metrics["status"] = "error"
        log.exception("Phase 1 failed")
        raise
    finally:
        elapsed = time.time() - t0
        write_phase_metrics(1, metrics, elapsed_sec=elapsed)
    log.info("All database downloads complete.")


if __name__ == "__main__":
    main()
