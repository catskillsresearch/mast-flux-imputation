"""Path resolution. Nothing here ever writes.

Resolution order for every location: explicit environment variable, then the
lab-drive layout under ``$CATSKILLS_FUSION_LAB`` (if set), then the repo-local
``data/`` fallback that the fetch / ingest scripts populate.

    CATSKILLS_FUSION_LAB   lab root, e.g. /media/catskills/CRC_FUSION_LAB/fusion-equilibrium-challenge
    FEC_MAST_DIR           directory holding the 1,206 ``mast_public_test`` parquets
                           (or its parent containing ``mast_public_test/``)
    FEC_DEMO_DIR           directory holding the 3 labelled starter-kit MAST demo parquets
    FAIR_MAST_DIR          directory of FAIR-MAST shots ingested to the FEC parquet schema
    FAIR_MAST_LABELS_DIR   EFIT scalar / X-point sidecars written by data/fair_mast/fetch_efit_labels.py
    M9_WEIGHTS_DIR         directory with CRC_FEQ_mast_M9_win50_i{101,202,303}.pt
    MFI_OUTPUT_DIR         where predictions / scores / viewer caches are written (default outputs/)
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"


def _env(name: str) -> Path | None:
    v = os.environ.get(name, "").strip()
    return Path(v).expanduser() if v else None


def lab_root() -> Path | None:
    return _env("CATSKILLS_FUSION_LAB")


def fec_public_test_dir() -> Path:
    p = _env("FEC_MAST_DIR")
    if p is not None:
        return p / "mast_public_test" if (p / "mast_public_test").is_dir() else p
    lab = lab_root()
    if lab is not None:
        cand = lab / "datasets" / "Sophelio__fusion-equilibrium-challenge" / "data" / "mast_public_test"
        if cand.is_dir():
            return cand
    return DATA / "fec" / "mast_public_test"


def fec_demo_dir() -> Path:
    p = _env("FEC_DEMO_DIR")
    return p if p is not None else DATA / "fec" / "demo"


def fair_mast_dir() -> Path:
    p = _env("FAIR_MAST_DIR")
    if p is not None:
        return p
    lab = lab_root()
    if lab is not None and (lab / "datasets" / "fair_mast_train").is_dir():
        return lab / "datasets" / "fair_mast_train"
    return DATA / "fair_mast" / "parquet"


def fair_mast_labels_dir() -> Path:
    p = _env("FAIR_MAST_LABELS_DIR")
    return p if p is not None else DATA / "fair_mast" / "labels"


def m9_weights_dir() -> Path:
    p = _env("M9_WEIGHTS_DIR")
    if p is not None:
        return p
    lab = lab_root()
    if lab is not None and (lab / "archive" / "artifacts").is_dir():
        # The shipped (Phase-2 Final) M9 trio; see DATA_PROVENANCE.md for checksums.
        return lab / "archive" / "artifacts"
    return REPO / "models" / "m9" / "weights"


def output_dir() -> Path:
    p = _env("MFI_OUTPUT_DIR")
    return p if p is not None else REPO / "outputs"


def assert_not_on_lab(path: Path) -> Path:
    """Refuse to write anywhere under the read-only lab drive."""
    rp = Path(path).resolve()
    for guard in ("/media/catskills/CRC_FUSION_LAB", str(lab_root() or "/nonexistent")):
        g = Path(guard)
        try:
            rp.relative_to(g.resolve() if g.exists() else g)
        except ValueError:
            continue
        raise SystemExit(f"refusing to write under the read-only lab drive: {rp}")
    return path
