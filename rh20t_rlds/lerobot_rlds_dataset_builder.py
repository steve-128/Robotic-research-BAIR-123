"""
TFDS DatasetBuilder: LeRobot-format RH20T (any cfg1–cfg7) → RLDS.

Source: HuggingFace robot-lev/rh20t_{cfg} datasets (LeRobot v3 format).

Usage
-----
# Build cfg3 (default):
    from rh20t_rlds.lerobot_rlds_dataset_builder import RH20tRldsHF
    RH20tRldsHF.hf_root = Path("data")   # parent of cfg3/, cfg1/, …
    builder = RH20tRldsHF(config="cfg3", data_dir="rlds_output/")
    builder.download_and_prepare()

# Build cfg1, only episodes 100–199:
    RH20tRldsHF.ep_start, RH20tRldsHF.ep_end = 100, 199
    builder = RH20tRldsHF(config="cfg1", data_dir="rlds_output/")
    builder.download_and_prepare()

On-disk LeRobot v3 layout (per cfg):
    {hf_root}/RH20T_hf_{cfg}/
        meta/info.json                        ← fps, features
        meta/tasks.parquet
        meta/episodes/chunk-*/file-*.parquet  ← per-episode length + video map
        data/chunk-*/file-*.parquet           ← state/action rows (chunked)
        videos/{cam_key}/chunk-*/file-*.mp4   ← multiple episodes per file

RLDS step schema (same for all cfgs):
    observation.image : uint8 [360, 640, 3]
    observation.state : float32 [state_dim]   ← 14 for UR5, 15 for others
    action            : float32 [8]           ← always 8
    reward / is_first / is_last / is_terminal / discount
    language_instruction : str
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Any

import av
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds

from ._config import CFG_META, ALL_CFGS, RH20TCfgMeta

IMAGE_H, IMAGE_W = 360, 640
_REPO_ROOT = Path(__file__).resolve().parent.parent


# ── TFDS BuilderConfig ────────────────────────────────────────────────────────

class RH20TBuilderConfig(tfds.core.BuilderConfig):
    """One BuilderConfig per RH20T configuration (cfg1–cfg7)."""

    def __init__(self, cfg_id: str, **kwargs):
        meta: RH20TCfgMeta = CFG_META[cfg_id]
        super().__init__(
            name=cfg_id,
            description=f"RH20T {cfg_id} ({meta.robot}, 640×360 RGB), "
                        f"{meta.total_episodes} episodes.",
            **kwargs,
        )
        self.meta = meta

    @property
    def hf_repo(self) -> str:
        return f"robot-lev/rh20t_{self.meta.cfg_id}"


# ── Metadata helpers ──────────────────────────────────────────────────────────

def _load_tasks(hf_dir: Path) -> dict[int, str]:
    df = pd.read_parquet(hf_dir / "meta" / "tasks.parquet").reset_index()
    if "task" in df.columns and "task_index" in df.columns:
        return dict(zip(df["task_index"].tolist(), df["task"].tolist()))
    if "task_name" in df.columns and "task_index" in df.columns:
        return dict(zip(df["task_index"].tolist(), df["task_name"].tolist()))
    return {}


def _load_data(hf_dir: Path) -> pd.DataFrame:
    """Concatenate every downloaded data parquet chunk."""
    files = sorted((hf_dir / "data").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No data parquets under {hf_dir}/data/")
    return pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)


def _load_episodes_index(hf_dir: Path) -> pd.DataFrame:
    """LeRobot v3 episodes index: per-episode length + video file mapping."""
    files = sorted((hf_dir / "meta" / "episodes").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No episodes index under {hf_dir}/meta/episodes/ — "
            f"re-run download_rh20t.py to fetch the meta files."
        )
    return pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)


def _load_fps(hf_dir: Path) -> float:
    with open(hf_dir / "meta" / "info.json") as f:
        return float(json.load(f).get("fps", 10))


def _find_cam_dir(hf_dir: Path) -> Path | None:
    vroot = hf_dir / "videos"
    if not vroot.is_dir():
        return None
    for d in sorted(vroot.iterdir()):
        if d.is_dir() and any(d.rglob("*.mp4")):
            return d
    return None


# ── Streaming episode generator ───────────────────────────────────────────────

def _stream_episodes(plans: dict[Path, list[tuple[int, int, int]]]):
    """
    plans: mp4 path → sorted [(start_frame, n_frames, episode_index), …]
    Decode each mp4 once; yield (episode_index, {frame_index: rgb}) as soon
    as an episode's frames are complete. One episode buffered at a time.
    """
    for mp4_path in sorted(plans):
        intervals = plans[mp4_path]
        it = iter(intervals)
        cur = next(it, None)
        buf: dict[int, np.ndarray] = {}
        pos = 0
        container = av.open(str(mp4_path))
        for av_frame in container.decode(video=0):
            if cur is None:
                break
            start, n_frames, ep_idx = cur
            if start <= pos < start + n_frames:
                rgb = av_frame.to_ndarray(format="rgb24")
                if rgb.shape[:2] != (IMAGE_H, IMAGE_W):
                    rgb = cv2.resize(rgb, (IMAGE_W, IMAGE_H),
                                     interpolation=cv2.INTER_LINEAR)
                buf[pos - start] = rgb
                if len(buf) == n_frames:
                    yield ep_idx, buf
                    buf = {}
                    cur = next(it, None)
            pos += 1
        container.close()
        if cur is not None and buf:
            # mp4 ended mid-episode (truncated file) — emit what we have
            yield cur[2], buf


# ── TFDS DatasetBuilder ────────────────────────────────────────────────────────

class RH20tRldsHF(tfds.core.GeneratorBasedBuilder):
    """
    RLDS dataset from HuggingFace LeRobot-format RH20T data.
    Supports cfg1–cfg7 via BuilderConfig.
    """

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "RH20T 640×360 RGB (LeRobot HF) → RLDS. Supports cfg1–cfg7."
    }

    BUILDER_CONFIGS = [RH20TBuilderConfig(cfg_id) for cfg_id in ALL_CFGS]
    DEFAULT_CONFIG_NAME = "cfg3"

    # Root directory that contains one sub-folder per cfg:
    #   {hf_root}/RH20T_hf_cfg3/   {hf_root}/RH20T_hf_cfg1/ …
    # Set before calling download_and_prepare().
    hf_root: Path = _REPO_ROOT / "data"

    # Inclusive episode_index range to convert (None → no bound).
    # Set before calling download_and_prepare().
    ep_start: int | None = None
    ep_end: int | None = None

    def _hf_dir(self) -> Path:
        cfg_id = self.builder_config.meta.cfg_id
        return Path(self.hf_root) / f"RH20T_hf_{cfg_id}"

    def _info(self) -> tfds.core.DatasetInfo:
        meta = self.builder_config.meta
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                "steps": tfds.features.Dataset({
                    "observation": tfds.features.FeaturesDict({
                        "image": tfds.features.Image(
                            shape=(IMAGE_H, IMAGE_W, 3),
                            dtype=tf.uint8,
                            encoding_format="jpeg",
                        ),
                        "state": tfds.features.Tensor(
                            shape=(meta.state_dim,), dtype=tf.float32
                        ),
                    }),
                    "action": tfds.features.Tensor(
                        shape=(meta.action_dim,), dtype=tf.float32
                    ),
                    "reward": tfds.features.Scalar(dtype=tf.float32),
                    "is_first": tfds.features.Scalar(dtype=tf.bool),
                    "is_last": tfds.features.Scalar(dtype=tf.bool),
                    "is_terminal": tfds.features.Scalar(dtype=tf.bool),
                    "discount": tfds.features.Scalar(dtype=tf.float32),
                    "language_instruction": tfds.features.Text(),
                }),
                "episode_metadata": tfds.features.FeaturesDict({
                    "episode_index": tfds.features.Scalar(dtype=tf.int32),
                    "task_index": tfds.features.Scalar(dtype=tf.int32),
                    "config": tfds.features.Text(),
                }),
            }),
            supervised_keys=None,
            homepage="https://rh20t.github.io/",
            citation="""
@article{chen2023rh20t,
  title={RH20T: A Comprehensive Robotic Dataset for Learning Diverse Skills},
  author={Chen, Hao and others},
  year={2023}
}
""",
        )

    def _split_generators(
        self, dl_manager: tfds.download.DownloadManager
    ) -> dict[str, Any]:
        hf_dir = self._hf_dir()
        cfg_id = self.builder_config.meta.cfg_id

        cam_dir = _find_cam_dir(hf_dir)
        if cam_dir is None:
            raise FileNotFoundError(
                f"No videos under {hf_dir}/videos/.\n"
                f"Run:  python download_rh20t.py --hf --cfg {cfg_id}"
            )
        cam_key = cam_dir.name

        eps = _load_episodes_index(hf_dir)
        fps = _load_fps(hf_dir)
        df = _load_data(hf_dir)
        tasks = _load_tasks(hf_dir)

        lo = 0 if self.ep_start is None else self.ep_start
        hi = (int(eps["episode_index"].max())
              if self.ep_end is None else self.ep_end)

        ck = f"videos/{cam_key}/chunk_index"
        fk = f"videos/{cam_key}/file_index"
        ts = f"videos/{cam_key}/from_timestamp"
        if ck not in eps.columns:
            raise KeyError(
                f"Episodes index has no columns for camera {cam_key!r} — "
                f"downloaded videos do not match meta/episodes."
            )

        # mp4 path → [(start_frame, n_frames, episode_index), …]
        plans: dict[Path, list[tuple[int, int, int]]] = {}
        n_missing_video = n_missing_rows = 0
        have_rows = set(df["episode_index"].unique().tolist())
        for _, row in eps.iterrows():
            ep_idx = int(row["episode_index"])
            if not (lo <= ep_idx <= hi):
                continue
            mp4 = (cam_dir / f"chunk-{int(row[ck]):03d}"
                   / f"file-{int(row[fk]):03d}.mp4")
            if not mp4.exists():
                n_missing_video += 1
                continue
            if ep_idx not in have_rows:
                n_missing_rows += 1
                continue
            start_frame = int(round(float(row[ts]) * fps))
            plans.setdefault(mp4, []).append(
                (start_frame, int(row["length"]), ep_idx)
            )
        for v in plans.values():
            v.sort()
        n_planned = sum(len(v) for v in plans.values())

        print(f"\n  Config   : {cfg_id}  ({self.builder_config.meta.robot})")
        print(f"  Camera   : {cam_key}")
        print(f"  Range    : episodes {lo}–{hi}")
        print(f"  Convert  : {n_planned} episodes across {len(plans)} mp4 file(s)")
        if n_missing_video:
            print(f"  Skipped  : {n_missing_video} (video file not downloaded)")
        if n_missing_rows:
            print(f"  Skipped  : {n_missing_rows} (state/action parquet missing)")
        if n_planned == 0:
            raise RuntimeError(
                f"No convertible episodes in range [{lo}, {hi}] — "
                f"run download_rh20t.py with a matching --ep-start/--ep-end."
            )

        return {
            "train": self._generate_examples(df, tasks, plans),
        }

    def _generate_examples(
        self,
        df: pd.DataFrame,
        tasks: dict[int, str],
        plans: dict[Path, list[tuple[int, int, int]]],
    ) -> Iterator:
        cfg_id = self.builder_config.meta.cfg_id

        for ep_idx, frames in _stream_episodes(plans):
            sub = df[df["episode_index"] == ep_idx].sort_values("frame_index")
            if sub.empty:
                continue
            task_idx = int(sub.iloc[0]["task_index"])
            lang = tasks.get(task_idx, "")
            n = len(sub)

            steps: list[dict] = []
            for i, (_, row) in enumerate(sub.iterrows()):
                frame = frames.get(int(row["frame_index"]))
                if frame is None:
                    continue
                is_last = bool(i == n - 1)
                steps.append({
                    "observation": {
                        "image": frame,
                        "state": np.array(row["observation.state"],
                                          dtype=np.float32),
                    },
                    "action": np.array(row["action"], dtype=np.float32),
                    "reward": np.float32(0.0),
                    "is_first": bool(i == 0),
                    "is_last": is_last,
                    "is_terminal": is_last,
                    "discount": np.float32(0.0 if is_last else 1.0),
                    "language_instruction": lang,
                })

            if len(steps) < 2:
                continue

            yield f"ep_{ep_idx:04d}", {
                "steps": steps,
                "episode_metadata": {
                    "episode_index": np.int32(ep_idx),
                    "task_index": np.int32(task_idx),
                    "config": cfg_id,
                },
            }
