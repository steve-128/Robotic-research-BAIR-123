"""
Download RH20T data for any configuration (cfg1–cfg7), then convert to RLDS.

Primary path  : Google Drive via gdown (may be rate-limited on public files).
Fallback path : HuggingFace Hub – robot-lev/rh20t_{cfg} (LeRobot v3 format).

HuggingFace download grabs:
  • All meta/ files
  • data/chunk-000/file-000.parquet  (state + action for all episodes)
  • Videos for the first camera found, N_EPISODES video files only

After download, RLDS conversion runs automatically unless --skip-build is given.
By default BOTH versions are kept on disk (source under ../../data/rh20t/):
  • the downloaded source  → ../../data/rh20t/RH20T_hf_{cfg}/  or
                             ../../data/rh20t/RH20T/RH20T_{cfg}/
  • the converted RLDS      → rlds_output/…/{cfg}/1.0.0/  (next to the script)
Pass --delete-source to remove the downloaded source after a successful build.

Usage
-----
    python download_rh20t.py                          # auto, cfg3, 5 episodes
    python download_rh20t.py --cfg cfg1               # cfg1 via auto (GDrive→HF)
    python download_rh20t.py --hf --cfg cfg5          # cfg5 from HuggingFace
    python download_rh20t.py --gdrive --cfg cfg2      # cfg2 from Google Drive
    python download_rh20t.py --hf --n-episodes 10     # 10-episode sample
    python download_rh20t.py --skip-build             # download only, no RLDS
    python download_rh20t.py --delete-source          # convert, then drop source

Default (no flags) = download → auto-convert → keep both source and RLDS output.
"""

import argparse
import tarfile
from pathlib import Path

from rh20t_rlds._config import CFG_META, ALL_CFGS, PATCH_GDRIVE_ID, PATCHED_CFGS

DATA_ROOT = (Path(__file__).resolve().parent / ".." / ".." / "data" / "rh20t").resolve()


# ── Google Drive ──────────────────────────────────────────────────────────────

def _apply_patch(extract_dir: Path):
    """Official patch.tar.gz: fixed gripper widths + joint angles for cfg1/cfg2.
    Extracted last so its files overwrite the ones from the main archives."""
    import gdown
    patch_archive = DATA_ROOT / "RH20T_patch.tar.gz"
    if not patch_archive.exists():
        print("[patch] Downloading patch.tar.gz (cfg1/cfg2 gripper+joint fix) …")
        try:
            gdown.download(id=PATCH_GDRIVE_ID, output=str(patch_archive), quiet=False)
        except Exception as exc:
            print(f"[patch] Download failed: {exc}\n"
                  f"[patch] WARNING: cfg1/cfg2 gripper widths and joint angles "
                  f"will be unpatched.")
            return
    print(f"[patch] Merging patch into {extract_dir} …")
    with tarfile.open(patch_archive, "r:gz") as tar:
        tar.extractall(extract_dir)


def try_gdrive(cfg_id: str) -> bool:
    import gdown
    meta = CFG_META[cfg_id]
    archive = DATA_ROOT / f"RH20T_{cfg_id}.tar.gz"
    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    if not archive.exists():
        print(f"[{cfg_id}] Downloading from Google Drive (may be rate-limited) …")
        try:
            gdown.download(id=meta.gdrive_file_id, output=str(archive), quiet=False)
        except Exception as exc:
            print(f"[{cfg_id}] Google Drive failed: {exc}")
            return False

    extract_dir = DATA_ROOT / "RH20T"
    extract_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{cfg_id}] Extracting {archive.name} → {extract_dir} …")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(extract_dir)
    if cfg_id in PATCHED_CFGS:
        _apply_patch(extract_dir)
    print(f"[{cfg_id}] Done. Raw data at {extract_dir}")
    return True


# ── HuggingFace ───────────────────────────────────────────────────────────────

def download_hf(cfg_id: str, n_episodes: int):
    from huggingface_hub import hf_hub_download, list_repo_files

    meta = CFG_META[cfg_id]
    hf_repo = f"robot-lev/rh20t_{cfg_id}"
    hf_dir = DATA_ROOT / f"RH20T_hf_{cfg_id}"
    hf_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{cfg_id}] Downloading from HuggingFace: {hf_repo}")
    print(f"  Destination : {hf_dir}")
    print(f"  Episodes    : first {n_episodes} video files")

    # Meta + data parquet
    meta_files = [
        "meta/info.json",
        "meta/tasks.parquet",
        "meta/rh20t_episodes.json",
        "meta/rh20t_config.json",
        "meta/stats.json",
        "meta/episodes/chunk-000/file-000.parquet",
    ]
    for rel in meta_files:
        dest = hf_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            print(f"  Downloading {rel} …")
            try:
                hf_hub_download(
                    repo_id=hf_repo, repo_type="dataset",
                    filename=rel, local_dir=str(hf_dir),
                )
            except Exception as exc:
                print(f"  [WARN] {rel}: {exc}")

    data_parquet = "data/chunk-000/file-000.parquet"
    dest = hf_dir / data_parquet
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        print(f"  Downloading {data_parquet} …")
        hf_hub_download(
            repo_id=hf_repo, repo_type="dataset",
            filename=data_parquet, local_dir=str(hf_dir),
        )

    # Pick a primary camera — the one with the most video files
    all_files = list(list_repo_files(hf_repo, repo_type="dataset"))
    vid_files = [f for f in all_files if f.startswith("videos/")]

    cam_counts: dict[str, int] = {}
    for f in vid_files:
        cam = f.split("/")[1]
        cam_counts[cam] = cam_counts.get(cam, 0) + 1
    if not cam_counts:
        print(f"  [WARN] No video files found in {hf_repo}")
        return

    primary_cam = max(cam_counts, key=cam_counts.__getitem__)
    cam_videos = sorted([f for f in vid_files if f.split("/")[1] == primary_cam])
    selected = cam_videos[:n_episodes]
    print(f"  Camera: {primary_cam}  ({len(cam_videos)} total files)")
    print(f"  Downloading {len(selected)} video file(s) …")

    for rel in selected:
        dest = hf_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            continue
        hf_hub_download(
            repo_id=hf_repo, repo_type="dataset",
            filename=rel, local_dir=str(hf_dir),
        )

    print(f"[{cfg_id}] HuggingFace download complete → {hf_dir}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Download RH20T dataset")
    parser.add_argument(
        "--cfg", choices=ALL_CFGS, default="cfg3",
        help="Which configuration to download (default: cfg3)",
    )
    parser.add_argument("--gdrive", action="store_true",
                        help="Try Google Drive only")
    parser.add_argument("--hf", action="store_true",
                        help="Use HuggingFace (skip Google Drive)")
    parser.add_argument(
        "--n-episodes", type=int, default=5,
        help="Number of video files to download from HF (default: 5)",
    )
    parser.add_argument(
        "--skip-build", action="store_true",
        help="Download only — do not convert to RLDS after download",
    )
    parser.add_argument(
        "--delete-source", action="store_true",
        help="Delete original downloaded data after successful RLDS conversion",
    )
    args = parser.parse_args()

    source_used: str | None = None  # "hf" or "raw"

    if args.hf:
        download_hf(args.cfg, args.n_episodes)
        source_used = "hf"
    elif args.gdrive:
        ok = try_gdrive(args.cfg)
        if ok:
            source_used = "raw"
        else:
            print("Falling back to HuggingFace …")
            download_hf(args.cfg, args.n_episodes)
            source_used = "hf"
    else:
        print(f"[{args.cfg}] Trying Google Drive first, then HuggingFace …")
        ok = try_gdrive(args.cfg)
        if ok:
            source_used = "raw"
        else:
            print("Google Drive unavailable — using HuggingFace.")
            download_hf(args.cfg, args.n_episodes)
            source_used = "hf"

    if not args.skip_build:
        from build_rlds import build_hf, build_raw
        print(f"\n[{args.cfg}] Starting RLDS conversion …")
        if source_used == "hf":
            build_hf(args.cfg, delete_source=args.delete_source)
        else:
            build_raw(args.cfg, delete_source=args.delete_source)


if __name__ == "__main__":
    main()
