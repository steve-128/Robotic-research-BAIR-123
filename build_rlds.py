"""
Build an RLDS/TFDS dataset from downloaded RH20T data.

Supports cfg1–cfg7.  Auto-detects the data source:
  1. Raw RH20T format (Google Drive)  → rh20t_api + RH20tRlds builder
  2. LeRobot HF format                → parquet/mp4 + RH20tRldsHF builder

Usage
-----
    # Build cfg3 (already downloaded):
    python build_rlds.py

    # Build a specific config:
    python build_rlds.py --cfg cfg1

    # Force a specific source:
    python build_rlds.py --cfg cfg5 --source hf
    python build_rlds.py --cfg cfg2 --source raw

Output
------
    rlds_output/{builder_name}/{cfg_id}/1.0.0/
"""

import argparse
import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "rh20t_api"))
sys.path.insert(0, str(_ROOT))

from rh20t_rlds._config import ALL_CFGS

DATA_ROOT = _ROOT / "data"
OUTPUT_DIR = _ROOT / "rlds_output"


# ── Source detection ──────────────────────────────────────────────────────────

def _has_hf_data(cfg_id: str) -> bool:
    hf_dir = DATA_ROOT / f"RH20T_hf_{cfg_id}"
    return (
        (hf_dir / "data" / "chunk-000" / "file-000.parquet").exists()
        and any((hf_dir / "videos").rglob("*.mp4"))
    )


def _has_raw_data(cfg_id: str) -> bool:
    raw_root = DATA_ROOT / "RH20T"
    if not raw_root.exists():
        return False
    for name in [f"RH20T_{cfg_id}", cfg_id]:
        p = raw_root / name
        if p.is_dir() and any(p.rglob("color/*.jpg")):
            return True
    return False


# ── Build functions ───────────────────────────────────────────────────────────

def build_hf(cfg_id: str, delete_source: bool = False):
    """Output → rlds_output/r_h20t_rlds_hf/{cfg_id}/1.0.0/"""
    import tensorflow_datasets as tfds
    from rh20t_rlds.lerobot_rlds_dataset_builder import RH20tRldsHF

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RH20tRldsHF.hf_root = DATA_ROOT
    builder = RH20tRldsHF(config=cfg_id, data_dir=str(OUTPUT_DIR))
    expected = OUTPUT_DIR / "r_h20t_rlds_hf" / cfg_id / "1.0.0"
    print(f"\n=== Building RLDS (HF source) for {cfg_id} ===")
    print(f"    Output → {expected}")
    builder.download_and_prepare(
        download_config=tfds.download.DownloadConfig(verify_ssl=False)
    )
    _sanity_check(builder, cfg_id)
    if delete_source:
        src = DATA_ROOT / f"RH20T_hf_{cfg_id}"
        print(f"    Deleting source: {src}")
        shutil.rmtree(src, ignore_errors=True)


def build_raw(cfg_id: str, delete_source: bool = False):
    """Output → rlds_output/r_h20t_rlds_raw/{cfg_id}/1.0.0/"""
    import tensorflow_datasets as tfds
    from rh20t_rlds.rh20t_rlds_dataset_builder import RH20tRlds

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RH20tRlds.raw_root = DATA_ROOT / "RH20T"
    builder = RH20tRlds(config=cfg_id, data_dir=str(OUTPUT_DIR))
    expected = OUTPUT_DIR / "r_h20t_rlds_raw" / cfg_id / "1.0.0"
    print(f"\n=== Building RLDS (raw source) for {cfg_id} ===")
    print(f"    Output → {expected}")
    builder.download_and_prepare(
        download_config=tfds.download.DownloadConfig(verify_ssl=False)
    )
    _sanity_check(builder, cfg_id)
    if delete_source:
        src = DATA_ROOT / "RH20T"
        print(f"    Deleting source: {src}")
        shutil.rmtree(src, ignore_errors=True)


# ── Sanity check ──────────────────────────────────────────────────────────────

def _sanity_check(builder, cfg_id: str):
    print(f"\nSanity check for {cfg_id} …")
    ds = builder.as_dataset(split="train")
    for ep in ds.take(1):
        steps = list(ep["steps"].as_numpy_iterator())
        s0 = steps[0]
        print(f"  Steps per episode : {len(steps)}")
        print(f"  image shape       : {s0['observation']['image'].shape}")
        print(f"  state shape       : {s0['observation']['state'].shape}")
        print(f"  action shape      : {s0['action'].shape}")
        lang = s0.get("language_instruction", b"")
        if isinstance(lang, bytes):
            lang = lang.decode()
        print(f"  instruction       : '{lang}'")
    print("Done.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build RLDS dataset from RH20T")
    parser.add_argument(
        "--cfg", choices=ALL_CFGS, default=None,
        help="Config to build (default: all that have data downloaded)",
    )
    parser.add_argument(
        "--source", choices=["auto", "hf", "raw"], default="auto",
        help="Data source (default: auto-detect)",
    )
    parser.add_argument(
        "--delete-source", action="store_true",
        help="Delete original downloaded data after successful RLDS conversion",
    )
    args = parser.parse_args()

    cfgs_to_build = [args.cfg] if args.cfg else ALL_CFGS

    built = 0
    for cfg_id in cfgs_to_build:
        src = args.source
        if src == "auto":
            if _has_raw_data(cfg_id):
                src = "raw"
            elif _has_hf_data(cfg_id):
                src = "hf"
            else:
                if args.cfg:
                    sys.exit(
                        f"No data found for {cfg_id}.\n"
                        f"Run:  python download_rh20t.py --cfg {cfg_id} [--hf]"
                    )
                continue  # skip configs without data when building all

        if src == "hf":
            build_hf(cfg_id, delete_source=args.delete_source)
        else:
            build_raw(cfg_id, delete_source=args.delete_source)
        built += 1

    if built == 0:
        sys.exit(
            "No downloaded data found for any config.\n"
            "Run:  python download_rh20t.py --hf [--cfg cfg3]"
        )

    print(f"\nBuilt {built} config(s). Output in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
