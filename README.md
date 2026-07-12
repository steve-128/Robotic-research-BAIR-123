# RH20T → RLDS

Download the [RH20T](https://rh20t.github.io/) robot-manipulation dataset (any of
configurations **cfg1–cfg7**) and convert it to
[RLDS](https://github.com/google-research/rlds)/TFDS, the format used by
[Open X-Embodiment](https://robotics-transformer-x.github.io/).

Two data sources are supported and auto-detected:

| Source | What it is | Used when |
|--------|------------|-----------|
| **HuggingFace** (`robot-lev/rh20t_{cfg}`) | LeRobot v3 reformat (parquet + AV1 mp4) | default / `--hf`; streams cleanly, supports episode ranges |
| **Google Drive** (`RH20T_cfg{n}.tar.gz`) | authors' original 640×360 RGB archive | `--gdrive`; ~2× smaller but rate-limited |

---

## Setup

```bash
conda activate rh20t_rlds          # Python 3.11
# key deps: tensorflow, tensorflow-datasets, av (PyAV), pandas,
#           opencv-python, gdown, huggingface_hub
```

The `rh20t_api/` submodule (scene loader for the raw source) is vendored in-repo.

---

## Quick start

```bash
# Download a config from HuggingFace and convert ALL episodes to RLDS:
python download_rh20t.py --hf --cfg cfg3

# Only a slice (inclusive episode_index range):
python download_rh20t.py --hf --cfg cfg1 --ep-start 0 --ep-end 99

# Open-ended range (episode 500 to the end):
python download_rh20t.py --hf --cfg cfg1 --ep-start 500

# Download only, skip conversion:
python download_rh20t.py --hf --cfg cfg5 --skip-build

# Convert data already on disk (no re-download):
python build_rlds.py --cfg cfg3 --ep-start 100 --ep-end 199
```

**Default (no `--ep-*` flags) = every episode.**

---

## Where files go

The data root is resolved **relative to the scripts**, three levels up, at
`../../../data/rh20t/` (e.g. if the repo is at `~/proj/Research`, data lands in
`~/data/rh20t/`). The converted RLDS output is saved **alongside the source**,
under the same data root.

```
../../../data/rh20t/
├── RH20T_hf_{cfg}/                     # HuggingFace source (kept by default)
│   ├── meta/
│   │   ├── info.json                   # fps, feature shapes
│   │   ├── tasks.parquet               # task_index → language instruction
│   │   └── episodes/…/*.parquet        # per-episode length + video-file map
│   ├── data/chunk-*/file-*.parquet     # state + action rows (chunked)
│   └── videos/{cam}/chunk-*/file-*.mp4 # primary camera only
├── RH20T/RH20T_{cfg}/                  # Google Drive source (if --gdrive)
│   └── <scene>/cam_*/color.mp4, transformed/*.npy, metadata.json
└── rlds_output/                        # converted RLDS, under the data root
    └── r_h20t_rlds_hf/{cfg}/1.0.0/     # (or r_h20t_rlds_raw/…)
        ├── dataset_info.json, features.json
        └── *-train.tfrecord-*
```

Pass `--delete-source` to drop the downloaded source after a successful build.

> **TFDS reuses an existing output dir and skips rebuilding.** To rebuild a
> config with a different episode range, delete
> `../../../data/rh20t/rlds_output/<builder>/<cfg>/` first.

---

## Configurations

Verified against `rh20t_api/configs/configs.json` and the HuggingFace
`info.json` files.

| cfg | Robot | DOF | Gripper | F/T sensor | HF `state` | `action` | Episodes |
|-----|-------|-----|---------|------------|-----------|----------|----------|
| cfg1 | Flexiv Rizon | 7 | Dahuan AG-95 | Dahuan | 15 | 8 | 4258 |
| cfg2 | Flexiv Rizon | 7 | Dahuan AG-95 | Dahuan | 15 | 8 | 1789 |
| cfg3 | UR5 | 6 | WSG-50 | ATI | 14 | 8 | 798 |
| cfg4 | UR5 | 6 | Robotiq 2F-85 | ATI | 14 | 8 | 2182 |
| cfg5 | Franka Panda | 7 | Franka | none | 15 | 8 | 1225 |
| cfg6 | KUKA iiwa | 7 | Robotiq 2F-85 | ATI | 15 | 8 | 1477 |
| cfg7 | KUKA iiwa | 7 | Robotiq 2F-85 | ATI | 15 | 8 | 896 |

- HF `state` = `ee_pose(7) + joint(DOF) + gripper(1)` → 15 for 7-DOF arms, 14 for the UR5s.
- `action` = 8 for all configs.
- cfg7 additionally has fingertip tactile in the full dataset (not used here).

### RLDS step schema

```
observation.image : uint8   [360, 640, 3]      # one external RGB camera
observation.state : float32 [state_dim]        # 14 (UR5) or 15
action            : float32 [8]
reward, is_first, is_last, is_terminal, discount
language_instruction : str
```

The **raw** builder produces a unified 8-D state/action for every config
(aligned TCP is always xyz+quaternion(7) + gripper(1)); the **HF** builder
preserves the LeRobot 14/15-D state.

---

## Estimated download sizes

The pipeline downloads **one camera only** (the RLDS schema has a single image
field), so the on-disk footprint is far smaller than the full HF repo, which
mirrors every camera. All numbers below are **whole-config, all episodes**.

| cfg | **Pipeline download** (HF, 1 cam) | GDrive `.tar.gz` | Full HF repo | primary-cam mp4s | ~MB / episode |
|-----|----------------------------------:|-----------------:|-------------:|-----------------:|--------------:|
| cfg1 | **≈ 45 GB** | 178 GB | 330 GB | 44.4 GB / 229 files | 10.4 |
| cfg2 | **≈ 21 GB** |  80 GB | 141 GB | 20.9 GB / 115 files | 11.7 |
| cfg3 | **≈ 7.5 GB** |  26 GB |  51 GB |  7.4 GB / 41 files  |  9.3 |
| cfg4 | **≈ 24 GB** |  88 GB | 166 GB | 24.1 GB / 133 files | 11.1 |
| cfg5 | **≈ 11 GB** |  37 GB |  76 GB | 11.2 GB / 66 files  |  9.1 |
| cfg6 | **≈ 15 GB** |  76 GB | 102 GB | 15.1 GB / 79 files  | 10.2 |
| cfg7 | **≈ 8 GB**  |  37 GB |  59 GB |  7.8 GB / 42 files  |  8.7 |

**Estimating a sub-range:** episodes are packed several-per-mp4, so multiply the
episode count by the `MB/episode` column, then round **up** to whole mp4 files
(downloads happen at file granularity). E.g. cfg1 episodes 0–499 ≈ 500 × 10.4 MB
≈ 5.2 GB → the ~27 mp4 files that cover them, ≈ 5–6 GB.

### How these were derived

Queried the HuggingFace datasets API for each `robot-lev/rh20t_{cfg}` repo:

- **Full HF repo** = the repo's `usedStorage` field (sum of all LFS blobs, all
  cameras).
- **primary-cam mp4s** = the camera directory with the most `.mp4` files (this is
  exactly what `download_hf()` selects); its size is the sum of those blobs'
  LFS sizes, file count is the number of `.mp4`s.
- **parquet** (state+action, ~0.06–0.29 GB per config) = sum of `data/**/*.parquet`
  blob sizes; folded into the pipeline-download column.
- **Pipeline download** = primary-cam mp4s + parquet + a few MB of meta,
  rounded.
- **MB/episode** = primary-cam mp4 bytes ÷ episode count (from `info.json`
  `total_episodes`).
- **GDrive `.tar.gz`** = the sizes printed on the RH20T download page
  (`rh20t.github.io`).

> Sizes are current as of 2026-07 and can shift if the HF mirror is re-encoded.
> Re-run the numbers with the HF API (`usedStorage` + the `tree?recursive=true`
> endpoint) if you need exact current values.

---

## Verifying a build

```bash
# sample-check 30 episodes of the cfg1 HF build (default):
python verify_rlds.py --cfg cfg1

# deeper sample, or a raw-source build, or an explicit path:
python verify_rlds.py --cfg cfg1 --episodes 50
python verify_rlds.py --cfg cfg2 --source raw
python verify_rlds.py --path /data/rh20t/rlds_output/r_h20t_rlds_hf/cfg1/1.0.0
```

Per sampled episode it checks: feature shapes/dtypes, RLDS flag consistency
(`is_first`/`is_last`/`is_terminal`/`discount`), image sanity (uint8, right
size, not black/frozen), state/action finiteness, and language presence.

When the HF source is still on disk it also **cross-checks faithfulness**:
state/action must match the source parquet rows bit-exactly, step count must
equal the row count, and the instruction must match `tasks.parquet` — this is
what catches episode/frame misalignment, not just formatting errors.
Exit code 0 = PASS, 1 = problems (listed).

Quick manual checks, independent of the script:
```bash
# episode count actually written:
python -c "import tensorflow_datasets as tfds; \
  print(tfds.builder_from_directory('/data/rh20t/rlds_output/r_h20t_rlds_hf/cfg1/1.0.0').info.splits['train'].num_examples)"
# a finished build has dataset_info.json + features.json + tfrecord shards,
# and NO leftover incomplete.* sibling directory.
ls /data/rh20t/rlds_output/r_h20t_rlds_hf/cfg1/
```

---

## Notes & caveats

- **Google Drive is unreliable for large public files.** `gdown` frequently hits
  the "quota exceeded / can't scan for viruses" wall on the multi-GB archives and
  cannot resume. Prefer `--hf`. If you must use the tarball, download it directly
  on the compute node (`gdown --fuzzy "<share-url>"`) rather than via your laptop.
- **cfg1/cfg2 patch.** The raw (`--gdrive`) source for cfg1 and cfg2 needs the
  official `patch.tar.gz` (corrected gripper widths + joint angles); the
  downloader fetches and merges it automatically.
- **Chunked parquet.** cfg1 (and others) split state/action across several
  `data/*.parquet` files. The downloader fetches exactly the chunks covering the
  selected episode range using the LeRobot episodes index — not just `file-000`.
- **Single view.** Only the primary external camera is downloaded. Multi-view
  RLDS (e.g. adding a wrist camera) would require downloading additional cameras
  and extending the builder schema.

---

## Files

| File | Purpose |
|------|---------|
| `download_rh20t.py` | Download a config (GDrive→HF fallback), then auto-build RLDS |
| `build_rlds.py` | Convert downloaded data to RLDS; source auto-detected |
| `verify_rlds.py` | Validate a built RLDS dataset (consistency + source faithfulness) |
| `rh20t_rlds/_config.py` | Per-config metadata (robot, dims, GDrive IDs, patch) |
| `rh20t_rlds/lerobot_rlds_dataset_builder.py` | TFDS builder for the HF source |
| `rh20t_rlds/rh20t_rlds_dataset_builder.py` | TFDS builder for the raw source |
| `rh20t_api/` | Vendored RH20T scene loader (raw source) |
