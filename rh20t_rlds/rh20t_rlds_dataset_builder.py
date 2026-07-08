"""
TFDS DatasetBuilder: raw RH20T format (any cfg1–cfg7) → RLDS.

Requires raw data from Google Drive (see download_rh20t.py --gdrive --cfg X).
Uses rh20t_api.RH20TScene to read mp4 videos and .npy sensor files.

On-disk layout expected after extraction:
    {raw_root}/RH20T_{cfg_id}/
        {scene_id}/
            cam_{serial}/color/{ts}.jpg  (if already extracted)
            force_torque_tcp_joint_timestamp.npy
            metadata.json
            …

RLDS step schema (identical for all cfgs — aligned TCP is always 7-D xyz+quat):
    observation.image : uint8 [360, 640, 3]  – primary external camera RGB
    observation.state : float32 [8]          – TCP pose (xyz+quat) + gripper
    action            : float32 [8]          – ΔTCP + next gripper command
    reward / is_first / is_last / is_terminal / discount
    language_instruction : str  (empty – not provided in raw format)

Note: raw archives ship cam_*/color.mp4 + timestamps.npy; frames for the
primary camera are extracted to cam_*/color/*.jpg on first use.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterator, Any

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from ._config import CFG_META, ALL_CFGS, ALIGNED_TCP_DIM, RH20TCfgMeta

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "rh20t_api"))

IMAGE_H, IMAGE_W = 360, 640
SAMPLE_INTERVAL_MS = 100  # 10 Hz


# ── TFDS BuilderConfig ────────────────────────────────────────────────────────

class RH20TRawBuilderConfig(tfds.core.BuilderConfig):
    """One BuilderConfig per RH20T configuration (cfg1–cfg7)."""

    def __init__(self, cfg_id: str, **kwargs):
        meta: RH20TCfgMeta = CFG_META[cfg_id]
        super().__init__(
            name=cfg_id,
            description=f"RH20T {cfg_id} ({meta.robot}, raw format), "
                        f"~{meta.total_episodes} episodes.",
            **kwargs,
        )
        self.meta = meta

    @property
    def state_dim(self) -> int:
        return ALIGNED_TCP_DIM + 1  # unified TCP pose (xyz+quat) + gripper

    @property
    def action_dim(self) -> int:
        return ALIGNED_TCP_DIM + 1  # ΔTCP + gripper command


# ── Scene helpers ─────────────────────────────────────────────────────────────

def _load_image(path: str) -> np.ndarray | None:
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.shape[:2] != (IMAGE_H, IMAGE_W):
        img = cv2.resize(img, (IMAGE_W, IMAGE_H), interpolation=cv2.INTER_LINEAR)
    return img


def _pick_primary_serial(scene) -> str | None:
    """Primary camera: external (non in-hand) cam with the most frames."""
    candidates = {s: ts for s, ts in scene.low_freq_timestamps.items() if ts}
    if not candidates:
        return None
    in_hand = set(getattr(scene._conf, "in_hand", None) or [])
    external = {s: ts for s, ts in candidates.items() if s not in in_hand}
    pool = external or candidates
    return max(pool, key=lambda s: len(pool[s]))


def _ensure_color_frames(scene, serial: str) -> bool:
    """Extract cam_{serial}/color.mp4 → color/*.jpg if not already done."""
    cam_dir = Path(scene.folder) / f"cam_{serial}"
    color_dir = cam_dir / "color"
    if color_dir.is_dir() and any(color_dir.glob("*.jpg")):
        return True
    mp4, ts = cam_dir / "color.mp4", cam_dir / "timestamps.npy"
    if not (mp4.exists() and ts.exists()):
        return False
    from rh20t_api.extract import convert_dir
    convert_dir(str(mp4), str(ts), str(cam_dir))
    return any(color_dir.glob("*.jpg"))


def _build_steps(scene, scene_id: str) -> list[dict] | None:
    """Extract timestep data from an RH20TScene. Returns None on failure."""
    tcp_dim = ALIGNED_TCP_DIM
    primary_serial = _pick_primary_serial(scene)
    if primary_serial is None:
        return None
    if not _ensure_color_frames(scene, primary_serial):
        print(f"  [WARN] {scene_id}: no color frames for cam_{primary_serial}")
        return None

    t_start = int(scene.start_timestamp)
    t_end = int(scene.end_timestamp)
    timestamps = list(range(t_start, t_end, SAMPLE_INTERVAL_MS))
    if len(timestamps) < 2:
        return None

    raw: list[dict] = []
    for t in timestamps:
        try:
            pairs = scene.get_image_path_pairs(t, image_types=["color"])
            if primary_serial not in pairs or not pairs[primary_serial]:
                continue
            img = _load_image(pairs[primary_serial][0])
            if img is None:
                continue
        except Exception:
            continue

        try:
            tcp = np.asarray(
                scene.get_tcp_aligned(t, serial="base"), dtype=np.float32
            ).flatten()[:tcp_dim]
            if tcp.shape[0] < tcp_dim:
                tcp = np.pad(tcp, (0, tcp_dim - tcp.shape[0]))
        except Exception:
            tcp = np.zeros(tcp_dim, dtype=np.float32)

        try:
            gripper = float(scene.get_gripper_command(t))
        except Exception:
            gripper = 0.0

        raw.append({"img": img, "tcp": tcp, "gripper": gripper})

    if len(raw) < 2:
        return None

    steps: list[dict] = []
    n = len(raw)
    for i, r in enumerate(raw):
        state = np.concatenate([r["tcp"], [r["gripper"]]]).astype(np.float32)
        if i < n - 1:
            nxt = raw[i + 1]
            delta_tcp = (nxt["tcp"] - r["tcp"]).astype(np.float32)
            next_grip = float(nxt["gripper"])
        else:
            delta_tcp = np.zeros(tcp_dim, dtype=np.float32)
            next_grip = r["gripper"]
        action = np.concatenate([delta_tcp, [next_grip]]).astype(np.float32)

        is_last = bool(i == n - 1)
        steps.append({
            "observation": {"image": r["img"], "state": state},
            "action": action,
            "reward": np.float32(0.0),
            "is_first": bool(i == 0),
            "is_last": is_last,
            "is_terminal": is_last,
            "discount": np.float32(0.0 if is_last else 1.0),
            "language_instruction": "",
        })
    return steps


# ── TFDS DatasetBuilder ────────────────────────────────────────────────────────

class RH20tRlds(tfds.core.GeneratorBasedBuilder):
    """
    RLDS dataset from raw RH20T data (rh20t_api).
    Supports cfg1–cfg7 via BuilderConfig.
    """

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "RH20T 640×360 RGB (raw format via rh20t_api) → RLDS. cfg1–cfg7."
    }

    BUILDER_CONFIGS = [RH20TRawBuilderConfig(cfg_id) for cfg_id in ALL_CFGS]
    DEFAULT_CONFIG_NAME = "cfg3"

    # Root directory that contains one sub-folder per cfg:
    #   {raw_root}/RH20T_cfg3/   {raw_root}/RH20T_cfg1/ …
    raw_root: Path = _REPO_ROOT / "data" / "RH20T"

    def _cfg_dir(self) -> Path:
        cfg_id = self.builder_config.meta.cfg_id
        root = Path(self.raw_root)
        for name in [f"RH20T_{cfg_id}", cfg_id, f"rh20t_{cfg_id}"]:
            p = root / name
            if p.is_dir():
                return p
        for p in sorted(root.iterdir()):
            if p.is_dir() and cfg_id in p.name.lower():
                return p
        raise FileNotFoundError(
            f"No {cfg_id} directory under {root}.\n"
            f"Run:  python download_rh20t.py --gdrive --cfg {cfg_id}"
        )

    def _info(self) -> tfds.core.DatasetInfo:
        cfg = self.builder_config
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
                            shape=(cfg.state_dim,), dtype=tf.float32
                        ),
                    }),
                    "action": tfds.features.Tensor(
                        shape=(cfg.action_dim,), dtype=tf.float32
                    ),
                    "reward": tfds.features.Scalar(dtype=tf.float32),
                    "is_first": tfds.features.Scalar(dtype=tf.bool),
                    "is_last": tfds.features.Scalar(dtype=tf.bool),
                    "is_terminal": tfds.features.Scalar(dtype=tf.bool),
                    "discount": tfds.features.Scalar(dtype=tf.float32),
                    "language_instruction": tfds.features.Text(),
                }),
                "episode_metadata": tfds.features.FeaturesDict({
                    "scene_id": tfds.features.Text(),
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
        from rh20t_api.configurations import load_conf
        cfg_id = self.builder_config.meta.cfg_id
        cfg_dir = self._cfg_dir()
        scene_paths = sorted([p for p in cfg_dir.iterdir() if p.is_dir()])
        robot_configs = load_conf(
            str(_REPO_ROOT / "rh20t_api" / "configs" / "configs.json")
        )
        print(f"\n  Config  : {cfg_id}  ({self.builder_config.meta.robot})")
        print(f"  Scenes  : {len(scene_paths)}")
        return {
            "train": self._generate_examples(scene_paths, robot_configs),
        }

    def _generate_examples(
        self, scene_paths: list[Path], robot_configs
    ) -> Iterator:
        from rh20t_api.scene import RH20TScene
        meta = self.builder_config.meta

        for scene_path in scene_paths:
            scene_id = scene_path.name
            try:
                scene = RH20TScene(str(scene_path), robot_configs)
            except Exception as exc:
                print(f"  [WARN] {scene_id}: {exc}")
                continue

            steps = _build_steps(scene, scene_id)
            if steps is None:
                continue

            yield scene_id, {
                "steps": steps,
                "episode_metadata": {
                    "scene_id": scene_id,
                    "config": meta.cfg_id,
                },
            }
