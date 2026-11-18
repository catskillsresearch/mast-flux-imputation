#!/usr/bin/env python3
"""Download the Fusion Equilibrium Challenge (FEC) MAST data.

* ``--demo``         the 3 labelled MAST starter-kit shots 28348 / 28350 / 28351
                     (github.com/Sophelio/fusion-equilibrium-challenge-starter, ``parquet_data/``,
                     stored in Git LFS; fetched from media.githubusercontent.com and sha256-checked)
                     -> ``data/fec/demo/``
* ``--public-test``  the 1,206 unlabelled ``mast_public_test`` shots from the Hugging Face dataset
                     ``Sophelio/fusion-equilibrium-challenge`` (CC BY 4.0) -> ``data/fec/mast_public_test/``

Both targets are git-ignored. Nothing is written outside ``--root``.

    python data/fec/fetch.py --demo
    python data/fec/fetch.py --public-test [--revision <commit>]
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipeline.paths import assert_not_on_lab  # noqa: E402

HF_REPO = "Sophelio/fusion-equilibrium-challenge"
STARTER = "https://media.githubusercontent.com/media/Sophelio/fusion-equilibrium-challenge-starter/main/parquet_data"
DEMO_SHA256 = {
    "mast_shot_28348.parquet": "302bf9e8bf6b3031c59975e90f7fbb360f2e19609ebbf7aa8355b252bff003fd",
    "mast_shot_28350.parquet": "0b7d875f4fab26900c28c0e5b065f7011a378855a4bdb68b65c23ecfd94858d7",
    "mast_shot_28351.parquet": "d6659f8c963cb9f5dbcb2206ca15914db1a4aed579d3f04b097fc4fa1bc1ed52",
}


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fetch_demo(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for name, sha in DEMO_SHA256.items():
        p = dest / name
        if not (p.is_file() and _sha(p) == sha):
            print(f"  GET {STARTER}/{name}")
            tmp = p.with_suffix(".part")
            urllib.request.urlretrieve(f"{STARTER}/{name}", tmp)
            tmp.replace(p)
        got = _sha(p)
        print(f"  {name}  sha256 {got[:12]}  {'OK' if got == sha else 'MISMATCH'}")
        if got != sha:
            raise SystemExit(f"sha256 mismatch for {p} (got an LFS pointer?)")


def fetch_public_test(dest: Path, revision: str | None) -> None:
    from huggingface_hub import snapshot_download

    tmp_root = dest.parent / ".hf_snapshot"
    snap = snapshot_download(repo_id=HF_REPO, repo_type="dataset", revision=revision,
                             allow_patterns=["data/mast_public_test/*", "README.md"],
                             local_dir=str(tmp_root), max_workers=8)
    src = Path(snap) / "data" / "mast_public_test"
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted(src.glob("*.parquet")):
        q = dest / p.name
        if not q.exists():
            p.replace(q)
        n += 1
    print(f"  {n} parquets in {dest}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--public-test", action="store_true")
    ap.add_argument("--revision", default=None, help="HF dataset commit (default: main)")
    ap.add_argument("--root", type=Path, default=ROOT / "data" / "fec")
    a = ap.parse_args()
    root = assert_not_on_lab(a.root)
    if not (a.demo or a.public_test):
        ap.error("pick --demo and/or --public-test")
    if a.demo:
        fetch_demo(root / "demo")
    if a.public_test:
        fetch_public_test(root / "mast_public_test", a.revision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
