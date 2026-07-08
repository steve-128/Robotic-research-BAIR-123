"""
Per-configuration metadata for all 7 RH20T configurations.

State/action shapes verified from robot-lev/rh20t_cfgN HuggingFace info.json.
Google Drive file IDs taken from https://rh20t.github.io/ (640×360 RGB column).
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class RH20TCfgMeta:
    cfg_id: str          # "cfg1" … "cfg7"
    robot: str           # "flexiv" | "ur5" | "franka" | "kuka"
    joint_dim: int       # robot DOF
    tcp_dim: int         # RAW on-robot TCP dimension (robot_tcp_field width in
                         # rh20t_api configs.json). The *aligned* TCP returned by
                         # RH20TScene.get_tcp_aligned() / stored in transformed/
                         # tcp_base.npy is always 7-D (xyz + quaternion) for every
                         # config — do not use tcp_dim to slice aligned poses.
    # LeRobot / HF shapes (verified from info.json 2026-06-28)
    state_dim: int       # observation.state shape[0]
    action_dim: int      # action shape[0]
    total_episodes: int  # approximate (from HF info.json)
    gdrive_file_id: str  # 640×360 "RGB with Robot Information" archive on Google
                         # Drive (verified against rh20t.github.io hrefs 2026-07-08)


CFG_META: dict[str, RH20TCfgMeta] = {
    "cfg1": RH20TCfgMeta(
        cfg_id="cfg1", robot="flexiv", joint_dim=7, tcp_dim=7,
        state_dim=15, action_dim=8, total_episodes=4258,
        gdrive_file_id="1xbFMNQDYZKMf_jL4f6e06iT95BZQe4eG",
    ),
    "cfg2": RH20TCfgMeta(
        cfg_id="cfg2", robot="flexiv", joint_dim=7, tcp_dim=7,
        state_dim=15, action_dim=8, total_episodes=1789,
        gdrive_file_id="1dCRwmdn3cg2330zhY0lIPvG6Q9YGoCYz",
    ),
    "cfg3": RH20TCfgMeta(
        cfg_id="cfg3", robot="ur5", joint_dim=6, tcp_dim=6,
        state_dim=14, action_dim=8, total_episodes=798,
        gdrive_file_id="1uwieq-EbA_eTXE668ekypQV1cO9PDfES",
    ),
    "cfg4": RH20TCfgMeta(
        cfg_id="cfg4", robot="ur5", joint_dim=6, tcp_dim=6,
        state_dim=14, action_dim=8, total_episodes=2182,
        gdrive_file_id="1fmVJMyiiKw8qOemU5FPzsW1NT3f5Kyjx",
    ),
    "cfg5": RH20TCfgMeta(
        cfg_id="cfg5", robot="franka", joint_dim=7, tcp_dim=6,
        state_dim=15, action_dim=8, total_episodes=1225,
        gdrive_file_id="17QgZ2HNdOAzF4krJ4eegH1rWnUTXfWDm",
    ),
    "cfg6": RH20TCfgMeta(
        cfg_id="cfg6", robot="kuka", joint_dim=7, tcp_dim=6,
        state_dim=15, action_dim=8, total_episodes=1477,
        gdrive_file_id="1Ytio7KTeU4gFlZNzl0-oX8wG-57VAbE9",
    ),
    "cfg7": RH20TCfgMeta(
        cfg_id="cfg7", robot="kuka", joint_dim=7, tcp_dim=6,
        state_dim=15, action_dim=8, total_episodes=896,
        gdrive_file_id="1ddwXNcRV3oi2mpMTLyhDvttwdd0lGRgX",
    ),
}

ALL_CFGS: list[str] = list(CFG_META.keys())

# Aligned/unified TCP pose (transformed/tcp_base.npy, get_tcp_aligned) is
# xyz + quaternion for all seven configs.
ALIGNED_TCP_DIM: int = 7

# patch.tar.gz on rh20t.github.io — fixes gripper widths and robot joint
# angles for cfg1 and cfg2; must be unzipped and merged over those configs.
PATCH_GDRIVE_ID: str = "1nMYHHvwOUeWwJK-2zTlwz1fdZ7lqwio8"
PATCHED_CFGS: tuple[str, ...] = ("cfg1", "cfg2")
