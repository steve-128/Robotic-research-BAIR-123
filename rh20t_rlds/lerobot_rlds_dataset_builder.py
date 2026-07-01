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

# Build cfg1:
    builder = RH20tRldsHF(config="cfg1", data_dir="rlds_output/")
    builder.download_and_prepare()

On-disk LeRobot layout (per cfg):
    {hf_root}/RH20T_hf_{cfg}/
        meta/info.json
        meta/tasks.parquet
        data/chunk-000/file-000.parquet   ← ALL rows (state, action, index)
        videos/{cam_key}/chunk-000/file-NNN.mp4

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


# ── Video helpers ─────────────────────────────────────────────────────────────

def _mp4_frame_count(path: Path) -> int:
    c = av.open(str(path))
    n = c.streams.video[0].frames
    c.close()
    return n


def _build_cum_ends(mp4s: list[Path]) -> list[int]:
    cum, total = [], 0
    for p in mp4s:
        total += _mp4_frame_count(p)
        cum.append(total)
    return cum


# ── Metadata helpers ──────────────────────────────────────────────────────────

def _load_tasks(hf_dir: Path) -> dict[int, str]:
    df = pd.read_parquet(hf_dir / "meta" / "tasks.parquet").reset_index()
    if "task" in df.columns and "task_index" in df.columns:
        return dict(zip(df["task_index"].tolist(), df["task"].tolist()))
    if "task_name" in df.columns and "task_index" in df.columns:
        return dict(zip(df["task_index"].tolist(), df["task_name"].tolist()))
    return {}


def _load_data(hf_dir: Path) -> pd.DataFrame:
    return pd.read_parquet(hf_dir / "data" / "chunk-000" / "file-000.parquet")


def _find_cam_dir(hf_dir: Path) -> Path | None:
    vroot = hf_dir / "videos"
    if not vroot.is_dir():
        return None
    for d in sorted(vroot.iterdir()):
        if d.is_dir() and any(d.rglob("*.mp4")):
            return d
    return None


# ── Streaming episode generator ───────────────────────────────────────────────

def _stream_episodes(
    df: pd.DataFrame,
    tasks: dict[int, str],
    ep_indices: list[int],
    mp4s: list[Path],
    cum_ends: list[int],
):
    """
    Stream mp4 files once and yield (ep_idx, row_list, frames_dict, tasks).
    At most one episode's frames live in memory at a time.
    """
    ep_rows: dict[int, list] = {}
    for ep_idx in ep_indices:
        sub = df[df["episode_index"] == ep_idx].sort_values("index")
        ep_rows[ep_idx] = [
            (int(r["index"]), int(r["frame_index"]), r)
            for _, r in sub.iterrows()
        ]

    needed: dict[int, tuple[int, int]] = {}
    for ep_idx, rows in ep_rows.items():
        for gidx, lfi, _ in rows:
            needed[gidx] = (ep_idx, lfi)

    frames_buf: dict[int, dict[int, np.ndarray]] = {e: {} for e in ep_indices}
    ep_expected = {ep_idx: len(rows) for ep_idx, rows in ep_rows.items()}
    ep_order = sorted(ep_indices, key=lambda e: ep_rows[e][0][0])
    next_ptr = 0

    global_pos = 0
    for mp4_path in mp4s:
        container = av.open(str(mp4_path))
        for av_frame in container.decode(video=0):
            if global_pos in needed:
                ep_idx, lfi = needed[global_pos]
                rgb = av_frame.to_ndarray(format="rgb24")
                if rgb.shape[:2] != (IMAGE_H, IMAGE_W):
                    rgb = cv2.resize(rgb, (IMAGE_W, IMAGE_H),
                                     interpolation=cv2.INTER_LINEAR)
                frames_buf[ep_idx][lfi] = rgb

                while next_ptr < len(ep_order):
                    cur = ep_order[next_ptr]
                    if len(frames_buf[cur]) < ep_expected[cur]:
                        break
                    yield cur, ep_rows[cur], frames_buf.pop(cur), tasks
                    next_ptr += 1
            global_pos += 1
        container.close()


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

        mp4s = sorted(cam_dir.rglob("*.mp4"))
        print(f"\n  Config   : {cfg_id}  ({self.builder_config.meta.robot})")
        print(f"  Camera   : {cam_dir.name}")
        print(f"  MP4 files: {len(mp4s)}")
        print("  Counting frames ...")
        cum_ends = _build_cum_ends(mp4s)
        max_global = cum_ends[-1] - 1 if cum_ends else -1
        print(f"  Total frames : {max_global + 1:,}")

        df = _load_data(hf_dir)
        tasks = _load_tasks(hf_dir)

        ep_groups = df[df["index"] <= max_global].groupby("episode_index")["index"].max()
        ep_indices = sorted(
            [int(ep) for ep, last in ep_groups.items() if last <= max_global]
        )
        print(f"  Episodes fully covered: {len(ep_indices)}")

        return {
            "train": self._generate_examples(df, tasks, ep_indices, mp4s, cum_ends),
        }

    def _generate_examples(
        self,
        df: pd.DataFrame,
        tasks: dict[int, str],
        ep_indices: list[int],
        mp4s: list[Path],
        cum_ends: list[int],
    ) -> Iterator:
        cfg_id = self.builder_config.meta.cfg_id

        for ep_idx, row_list, frames, tasks_ref in _stream_episodes(
            df, tasks, ep_indices, mp4s, cum_ends
        ):
            task_idx = int(row_list[0][2]["task_index"])
            lang = tasks_ref.get(task_idx, "")
            n = len(row_list)

            steps: list[dict] = []
            for i, (_, lfi, row) in enumerate(row_list):
                frame = frames.get(lfi)
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
