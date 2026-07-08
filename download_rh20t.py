"""
Download RH20T data for any configuration (cfg1–cfg7), then convert to RLDS.

Primary path  : Google Drive via gdown (may be rate-limited on public files).
Fallback path : HuggingFace Hub – robot-lev/rh20t_{cfg} (LeRobot v3 format).

HuggingFace download grabs (for the selected episode range — default: ALL):
  • All meta/ files (episodes index, tasks, stats, info.json)
  • Every data/*.parquet chunk covering the range (state + action)
  • The primary camera's video files covering the range

After download, RLDS conversion runs automatically unless --skip-build is given.
By default BOTH versions are kept on disk under the data root ../../../data/rh20t/:
  • the downloaded source  → ../../../data/rh20t/RH20T_hf_{cfg}/  or
                             ../../../data/rh20t/RH20T/RH20T_{cfg}/
  • the converted RLDS      → ../../../data/rh20t/rlds_output/…/{cfg}/1.0.0/
Pass --delete-source to remove the downloaded source after a successful build.

Usage
-----
    python download_rh20t.py                          # auto, cfg3, ALL episodes
    python download_rh20t.py --cfg cfg1               # cfg1 via auto (GDrive→HF)
    python download_rh20t.py --hf --cfg cfg5          # cfg5 from HF, ALL episodes
    python download_rh20t.py --gdrive --cfg cfg2      # cfg2 from Google Drive
    python download_rh20t.py --hf --ep-start 0 --ep-end 99    # episodes 0–99
    python download_rh20t.py --hf --ep-start 500              # episode 500 → end
    python download_rh20t.py --skip-build             # download only, no RLDS
    python download_rh20t.py --delete-source          # convert, then drop source

Default (no flags) = download ALL → auto-convert → keep both source and RLDS.
"""

import argparse
import tarfile
from pathlib import Path

from rh20t_rlds._config import CFG_META, ALL_CFGS, PATCH_GDRIVE_ID, PATCHED_CFGS

DATA_ROOT = (Path(__file__).resolve().parent / ".." / ".." / ".." / "data" / "rh20t").resolve()


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

def download_hf(cfg_id: str, ep_start: int | None = None, ep_end: int | None = None):
    """Download a cfg from HuggingFace.

    ep_start / ep_end select an inclusive episode_index range. If both are
    None, the WHOLE config is downloaded (all episodes, all data parquets,
    all video files of the primary camera).
    """
    from huggingface_hub import hf_hub_download, list_repo_files
    import pandas as pd

    hf_repo = f"robot-lev/rh20t_{cfg_id}"
    hf_dir = DATA_ROOT / f"RH20T_hf_{cfg_id}"
    hf_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{cfg_id}] Downloading from HuggingFace: {hf_repo}")
    print(f"  Destination : {hf_dir}")

    all_files = list(list_repo_files(hf_repo, repo_type="dataset"))

    def fetch(rel: str):
        dest = hf_dir / rel
        if dest.exists():
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        hf_hub_download(
            repo_id=hf_repo, repo_type="dataset",
            filename=rel, local_dir=str(hf_dir),
        )

    # 1. All meta files (small): info.json, tasks/stats, episodes index parquets
    meta_files = [f for f in all_files if f.startswith("meta/")]
    print(f"  Meta files  : {len(meta_files)}")
    for rel in meta_files:
        try:
            fetch(rel)
        except Exception as exc:
            print(f"  [WARN] {rel}: {exc}")

    # 2. Resolve episode range from the episodes index
    eps_files = sorted((hf_dir / "meta" / "episodes").rglob("*.parquet"))
    if not eps_files:
        raise SystemExit(f"No episodes index under {hf_dir}/meta/episodes/")
    eps = pd.concat([pd.read_parquet(p) for p in eps_files], ignore_index=True)
    last_ep = int(eps["episode_index"].max())
    lo = 0 if ep_start is None else max(0, ep_start)
    hi = last_ep if ep_end is None else min(ep_end, last_ep)
    if lo > hi:
        raise SystemExit(
            f"Invalid episode range [{lo}, {hi}] — dataset has episodes 0–{last_ep}"
        )
    sel = eps[(eps["episode_index"] >= lo) & (eps["episode_index"] <= hi)]
    print(f"  Episodes    : {lo}–{hi}  ({len(sel)} of {last_ep + 1})")

    # 3. Data parquets covering the range (cfg1 etc. split these into chunks)
    data_files = sorted({
        f"data/chunk-{int(c):03d}/file-{int(f):03d}.parquet"
        for c, f in zip(sel["data/chunk_index"], sel["data/file_index"])
    })
    print(f"  Data files  : {len(data_files)}")
    for rel in data_files:
        fetch(rel)

    # 4. Videos: primary camera = the one with the most files in the repo
    vid_files = [f for f in all_files if f.startswith("videos/")]
    cam_counts: dict[str, int] = {}
    for f in vid_files:
        cam = f.split("/")[1]
        cam_counts[cam] = cam_counts.get(cam, 0) + 1
    if not cam_counts:
        print(f"  [WARN] No video files found in {hf_repo}")
        return
    primary_cam = max(cam_counts, key=cam_counts.__getitem__)

    ck = f"videos/{primary_cam}/chunk_index"
    fk = f"videos/{primary_cam}/file_index"
    if ck in sel.columns:
        selected = sorted({
            f"videos/{primary_cam}/chunk-{int(c):03d}/file-{int(f):03d}.mp4"
            for c, f in zip(sel[ck], sel[fk])
        })
    else:  # camera not in the episodes index — fall back to every file
        selected = sorted(f for f in vid_files if f.split("/")[1] == primary_cam)
    print(f"  Camera      : {primary_cam}")
    print(f"  Video files : {len(selected)}")
    for i, rel in enumerate(selected, 1):
        print(f"    [{i}/{len(selected)}] {rel}")
        fetch(rel)

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
        "--ep-start", type=int, default=None,
        help="First episode_index to download/convert (default: 0)",
    )
    parser.add_argument(
        "--ep-end", type=int, default=None,
        help="Last episode_index (inclusive) to download/convert "
             "(default: last episode — i.e. everything)",
    )
    parser.add_argument(
        "--n-episodes", type=int, default=None,
        help="DEPRECATED: shorthand for --ep-start 0 --ep-end N-1",
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

    ep_start, ep_end = args.ep_start, args.ep_end
    if args.n_episodes is not None and ep_start is None and ep_end is None:
        print(f"[deprecated] --n-episodes {args.n_episodes} → "
              f"--ep-start 0 --ep-end {args.n_episodes - 1}")
        ep_start, ep_end = 0, args.n_episodes - 1

    source_used: str | None = None  # "hf" or "raw"

    if args.hf:
        download_hf(args.cfg, ep_start, ep_end)
        source_used = "hf"
    elif args.gdrive:
        ok = try_gdrive(args.cfg)
        if ok:
            source_used = "raw"
        else:
            print("Falling back to HuggingFace …")
            download_hf(args.cfg, ep_start, ep_end)
            source_used = "hf"
    else:
        print(f"[{args.cfg}] Trying Google Drive first, then HuggingFace …")
        ok = try_gdrive(args.cfg)
        if ok:
            source_used = "raw"
        else:
            print("Google Drive unavailable — using HuggingFace.")
            download_hf(args.cfg, ep_start, ep_end)
            source_used = "hf"

    if not args.skip_build:
        from build_rlds import build_hf, build_raw
        print(f"\n[{args.cfg}] Starting RLDS conversion …")
        if source_used == "hf":
            build_hf(args.cfg, delete_source=args.delete_source,
                     ep_start=ep_start, ep_end=ep_end)
        else:
            build_raw(args.cfg, delete_source=args.delete_source,
                      ep_start=ep_start, ep_end=ep_end)


if __name__ == "__main__":
    main()
