# Where things live

## Code (the repo)
The scripts (`download_rh20t.py`, `build_rlds.py`, `verify_rlds.py`,
`rh20t_rlds/`, `rh20t_api/`) live in the repo directory, e.g. on EC2:
`/home/steve92428/Robotic-research-BAIR-123/`.

## Data root (everything downloaded + built)
Resolved **relative to the scripts**, three levels up, at `../../../data/rh20t/`.
On EC2 that resolves to **`/data/rh20t/`**. Both the download and the RLDS
output go under here, so a config's source and dataset sit together.

```
/data/rh20t/
├── RH20T_hf_{cfg}/          ← HuggingFace source (parquet + primary-camera mp4)
│   ├── meta/               (info.json, tasks.parquet, episodes index)
│   ├── data/…parquet       (state + action rows)
│   └── videos/{cam}/…mp4    (one camera only)
├── RH20T/RH20T_{cfg}/       ← Google Drive source (only if --gdrive)
└── rlds_output/            ← the converted datasets
    ├── r_h20t_rlds_hf/{cfg}/1.0.0/    (built from the HF source)
    └── r_h20t_rlds_raw/{cfg}/1.0.0/   (built from the GDrive source)
```

By default both the source **and** the RLDS output are kept. Pass
`--delete-source` to remove the download after a successful build.

## The saved dataset (what `1.0.0/` contains)
A standard **RLDS/TFDS** dataset per config:
- `dataset_info.json`, `features.json` — metadata + schema
- `r_h20t_rlds_hf-train.tfrecord-*` — the sharded episodes

Each episode is a sequence of steps:
```
observation.image : uint8   [360, 640, 3]   (one external RGB camera, JPEG-encoded)
observation.state : float32 [14 or 15]      (14 for UR5 cfg3/4, else 15)
action            : float32 [8]
reward, is_first, is_last, is_terminal, discount, language_instruction
episode_metadata  : episode_index, task_index, config
```

Load it back with:
```python
import tensorflow_datasets as tfds
ds = tfds.builder_from_directory(
    "/data/rh20t/rlds_output/r_h20t_rlds_hf/cfg1/1.0.0").as_dataset(split="train")
```

## Rough sizes (whole config, all episodes)
Download (HF, 1 camera): cfg1 ≈45 GB, cfg2 ≈21, cfg3 ≈7.5, cfg4 ≈24,
cfg5 ≈11, cfg6 ≈15, cfg7 ≈8 GB.
RLDS output is ~4× the download (JPEG frames vs compressed video) — e.g.
full cfg1 ≈ 180 GB. Keep room for source **+** output unless `--delete-source`.
