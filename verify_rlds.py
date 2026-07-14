"""
Verify a built RH20T RLDS dataset is correct and loadable.

Loads the TFDS dataset back and checks, per sampled episode:
  - feature shapes / dtypes match the declared info
  - RLDS step flags are consistent
        is_first : true only on step 0
        is_last / is_terminal : true only on the final step
        discount : 1.0 except 0.0 on the final step
  - images are real (uint8, correct HxW). Black/constant frames are reported
    as a WARNING (known dead-camera issue in the source mirror) together with
    the source mp4 file and the episode's start-end time and duration.
  - state / action are finite (no NaN/Inf) and not all-zero
  - language_instruction present (warn if empty)
  - each episode has >= 2 steps

Source cross-check (HF source only, on by default when the download is still
on disk; skipped otherwise):
  - per sampled episode, state/action arrays must EXACTLY match the parquet
    rows for that episode_index (proves no row/episode misalignment)
  - step count must equal the parquet row count for the episode
  - language_instruction must match tasks.parquet for the episode's task_index

Exit code 0 = all good, 1 = problems found.

Usage
-----
    python verify_rlds.py --cfg cfg1                 # HF build (default)
    python verify_rlds.py --cfg cfg1 --source raw    # raw build
    python verify_rlds.py --cfg cfg1 --episodes 20   # sample 20 episodes
    python verify_rlds.py --cfg cfg1 --seed 12345    # reproduce a prior sample
    python verify_rlds.py --cfg cfg1 --no-source-check   # skip parquet check
    python verify_rlds.py --path /data/rh20t/rlds_output/r_h20t_rlds_hf/cfg1/1.0.0
"""

import argparse
import random
import sys
from pathlib import Path

# tolerate non-UTF8 consoles (Windows cp1252) — never crash on printing
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

import numpy as np
import tensorflow_datasets as tfds

_ROOT = Path(__file__).resolve().parent
DATA_ROOT = (_ROOT / ".." / ".." / ".." / "data" / "rh20t").resolve()
OUTPUT_DIR = DATA_ROOT / "rlds_output"

BUILDER = {"hf": "r_h20t_rlds_hf", "raw": "r_h20t_rlds_raw"}


def _dataset_dir(cfg: str, source: str) -> Path:
    return OUTPUT_DIR / BUILDER[source] / cfg / "1.0.0"


def _load_source(cfg: str):
    """Load parquet rows + task strings from the HF download, if present.
    Returns (df, tasks) or (None, None) when the source isn't on disk."""
    hf_dir = DATA_ROOT / f"RH20T_hf_{cfg}"
    files = sorted((hf_dir / "data").rglob("*.parquet"))
    if not files:
        return None, None
    import pandas as pd
    df = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    tasks = {}
    tp = hf_dir / "meta" / "tasks.parquet"
    if tp.exists():
        tdf = pd.read_parquet(tp).reset_index()
        if "task" in tdf.columns and "task_index" in tdf.columns:
            tasks = dict(zip(tdf["task_index"], tdf["task"]))
    return df, tasks


def _load_video_locations(cfg: str) -> dict[int, tuple[str, float, float]] | None:
    """episode_index → (mp4 relpath, from_s, to_s) for the camera on disk.
    Used to point at the source clip when a black/frozen episode is found."""
    hf_dir = DATA_ROOT / f"RH20T_hf_{cfg}"
    files = sorted((hf_dir / "meta" / "episodes").rglob("*.parquet"))
    vroot = hf_dir / "videos"
    if not files or not vroot.is_dir():
        return None
    cam = None
    for d in sorted(vroot.iterdir()):
        if d.is_dir() and any(d.rglob("*.mp4")):
            cam = d.name
            break
    import pandas as pd
    eps = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    ck = f"videos/{cam}/chunk_index"
    if cam is None or ck not in eps.columns:
        return None
    locs: dict[int, tuple[str, float, float]] = {}
    for _, r in eps.iterrows():
        locs[int(r["episode_index"])] = (
            f"videos/{cam}/chunk-{int(r[ck]):03d}"
            f"/file-{int(r[f'videos/{cam}/file_index']):03d}.mp4",
            float(r[f"videos/{cam}/from_timestamp"]),
            float(r[f"videos/{cam}/to_timestamp"]),
        )
    return locs


def _check_against_source(tag, steps, eid, df, tasks, problems, warnings):
    """Compare an RLDS episode with its source parquet rows."""
    sub = df[df["episode_index"] == int(eid)].sort_values("frame_index")
    if sub.empty:
        warnings.append(f"{tag}: episode not in source parquet (range mismatch?)")
        return
    src_state = np.stack(sub["observation.state"].to_numpy()).astype(np.float32)
    src_act = np.stack(sub["action"].to_numpy()).astype(np.float32)
    if len(steps) != len(sub):
        problems.append(f"{tag}: {len(steps)} steps vs {len(sub)} parquet rows")
        return
    rlds_state = np.stack([s["observation"]["state"] for s in steps])
    rlds_act = np.stack([s["action"] for s in steps])
    if not np.array_equal(rlds_state, src_state):
        problems.append(f"{tag}: observation.state != source parquet "
                        f"(max|delta|={np.abs(rlds_state - src_state).max():.3g})")
    if not np.array_equal(rlds_act, src_act):
        problems.append(f"{tag}: action != source parquet "
                        f"(max|delta|={np.abs(rlds_act - src_act).max():.3g})")
    if tasks:
        src_lang = str(tasks.get(int(sub.iloc[0]["task_index"]), ""))
        lang = steps[0].get("language_instruction", b"")
        lang = lang.decode() if isinstance(lang, bytes) else str(lang)
        if lang != src_lang:
            problems.append(f"{tag}: language {lang!r} != source {src_lang!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify an RH20T RLDS dataset")
    ap.add_argument("--cfg", default="cfg1")
    ap.add_argument("--source", choices=["hf", "raw"], default="hf")
    ap.add_argument("--path", default=None,
                    help="Explicit path to the 1.0.0 dataset dir "
                         "(overrides --cfg/--source)")
    ap.add_argument("--episodes", type=int, default=30,
                    help="How many episodes to sample-check (default: 30)")
    ap.add_argument("--no-source-check", action="store_true",
                    help="Skip the cross-check against the source parquets")
    ap.add_argument("--seed", type=int, default=None,
                    help="Random seed for episode sampling. Default: a fresh "
                         "seed is drawn and printed; pass it back to "
                         "reproduce the exact same sample.")
    args = ap.parse_args()

    ds_dir = Path(args.path) if args.path else _dataset_dir(args.cfg, args.source)
    if not (ds_dir / "dataset_info.json").exists():
        print(f"FAIL: no dataset_info.json at {ds_dir}\n"
              f"      (build not finished, or wrong --cfg/--source/--path)")
        return 1

    print(f"[1/4] Loading dataset metadata via tfds.builder_from_directory()")
    print(f"      {ds_dir}")
    builder = tfds.builder_from_directory(str(ds_dir))
    info = builder.info
    n_eps = info.splits["train"].num_examples
    print(f"      Episodes in split : {n_eps}")
    print(f"      Features          :")
    step = info.features["steps"]
    for k in ("observation", "action"):
        print(f"        {k}: {step[k]}")

    seed = args.seed if args.seed is not None else random.randrange(2**31)
    print(f"[2/4] Opening tf.data pipeline via builder.as_dataset(split='train')")
    print(f"      sampling seed : {seed}"
          f"  (re-run with --seed {seed} to reproduce this exact sample)")
    ds = builder.as_dataset(
        split="train",
        shuffle_files=True,
        read_config=tfds.ReadConfig(shuffle_seed=seed),
    )
    # small element-level shuffle for within-shard randomness (episode records
    # hold their steps lazily, so the buffer stays light)
    ds = ds.shuffle(64, seed=seed, reshuffle_each_iteration=False)

    src_df = src_tasks = None
    video_locs = None
    if args.source == "hf" and not args.no_source_check:
        print(f"[3/4] Loading source parquets for the faithfulness cross-check")
        src_df, src_tasks = _load_source(args.cfg)
        if src_df is None:
            print("      source parquets not on disk — cross-check SKIPPED")
        else:
            print(f"      cross-check ON: {src_df['episode_index'].nunique()} "
                  f"episodes, {len(src_df):,} rows in parquets")
        video_locs = _load_video_locations(args.cfg)
        if video_locs:
            print(f"      episode->video map loaded "
                  f"({len(video_locs)} episodes) for locating bad frames")
    else:
        print(f"[3/4] Source cross-check disabled — skipping")

    problems: list[str] = []
    warnings: list[str] = []
    tot_steps = 0
    img_means: list[float] = []
    checked = 0

    n_to_check = min(args.episodes, n_eps)
    print(f"[4/4] Checking {n_to_check} episode(s) "
          f"(flags, images, numerics{', source match' if src_df is not None else ''}) …")
    for ep in ds.take(args.episodes):
        p0, w0 = len(problems), len(warnings)
        steps = list(ep["steps"].as_numpy_iterator())
        eid = ep["episode_metadata"].get("episode_index")
        tag = f"ep[{int(eid)}]" if eid is not None else f"ep#{checked}"
        n = len(steps)
        tot_steps += n

        if n < 2:
            problems.append(f"{tag}: only {n} step(s)")
            checked += 1
            print(f"      [{checked}/{n_to_check}] {tag}: {n} steps — 1 PROBLEM(S)")
            continue

        # --- RLDS flag consistency -------------------------------------------
        firsts = [i for i, s in enumerate(steps) if s["is_first"]]
        lasts = [i for i, s in enumerate(steps) if s["is_last"]]
        terms = [i for i, s in enumerate(steps) if s["is_terminal"]]
        if firsts != [0]:
            problems.append(f"{tag}: is_first at {firsts}, expected [0]")
        if lasts != [n - 1]:
            problems.append(f"{tag}: is_last at {lasts}, expected [{n-1}]")
        if terms != [n - 1]:
            problems.append(f"{tag}: is_terminal at {terms}, expected [{n-1}]")
        disc = np.array([s["discount"] for s in steps])
        if not (np.allclose(disc[:-1], 1.0) and np.isclose(disc[-1], 0.0)):
            problems.append(f"{tag}: discount pattern wrong "
                            f"(first={disc[0]}, last={disc[-1]})")

        # --- image sanity (first + middle + last) ----------------------------
        H, W = info.features["steps"]["observation"]["image"].shape[:2]
        const_steps = []
        for j in (0, n // 2, n - 1):
            img = steps[j]["observation"]["image"]
            if img.dtype != np.uint8:
                problems.append(f"{tag} step{j}: image dtype {img.dtype} != uint8")
            if img.shape != (H, W, 3):
                problems.append(f"{tag} step{j}: image shape {img.shape} "
                                f"!= {(H, W, 3)}")
            if img.std() < 1.0:
                const_steps.append(j)
        if const_steps:
            # black/frozen camera — known SOURCE data issue, so a warning,
            # located in the source video so it can be inspected directly.
            loc = ""
            if video_locs and eid is not None and int(eid) in video_locs:
                mp4, t0, t1 = video_locs[int(eid)]
                loc = (f"\n        source: {mp4} @ {t0:.1f}s-{t1:.1f}s "
                       f"(duration {t1 - t0:.1f}s)")
            warnings.append(
                f"{tag}: black/constant frames at steps {const_steps} "
                f"(std<1) — likely dead/covered camera in the source; "
                f"exclude from training{loc}")
        # frames should differ across the episode (not a frozen video)
        d = np.abs(steps[0]["observation"]["image"].astype(np.int16)
                   - steps[-1]["observation"]["image"].astype(np.int16)).mean()
        if d < 0.5:
            warnings.append(f"{tag}: first and last frame nearly identical "
                            f"(mean|delta|={d:.2f})")
        img_means.append(float(steps[0]["observation"]["image"].mean()))

        # --- state / action numeric sanity -----------------------------------
        for key in ("state",):
            arr = np.stack([s["observation"][key] for s in steps])
            if not np.isfinite(arr).all():
                problems.append(f"{tag}: observation.{key} has NaN/Inf")
            if np.allclose(arr, 0.0):
                problems.append(f"{tag}: observation.{key} all zero")
        act = np.stack([s["action"] for s in steps])
        if not np.isfinite(act).all():
            problems.append(f"{tag}: action has NaN/Inf")
        if np.allclose(act, 0.0):
            warnings.append(f"{tag}: action all zero")

        # --- language --------------------------------------------------------
        lang = steps[0].get("language_instruction", b"")
        lang = lang.decode() if isinstance(lang, bytes) else lang
        if not lang.strip():
            warnings.append(f"{tag}: empty language_instruction")

        # --- faithfulness vs source parquet -----------------------------------
        if src_df is not None and eid is not None:
            _check_against_source(tag, steps, int(eid), src_df, src_tasks,
                                  problems, warnings)

        checked += 1
        dp, dw = len(problems) - p0, len(warnings) - w0
        status = "ok" if dp == 0 else f"{dp} PROBLEM(S)"
        if dw:
            status += f", {dw} warning(s)"
        print(f"      [{checked}/{n_to_check}] {tag}: {n} steps — {status}")

    # --- report --------------------------------------------------------------
    print(f"\nChecked {checked} episode(s) of {n_eps} in the split "
          f"({checked / max(n_eps, 1):.1%} sample), {tot_steps} steps total "
          f"(mean {tot_steps / max(checked,1):.0f} steps/episode).")
    if img_means:
        print(f"Image brightness across episodes: "
              f"min={min(img_means):.0f} max={max(img_means):.0f} "
              f"(healthy ~ spread of real scenes; all-equal => suspicious).")

    if warnings:
        print(f"\n{len(warnings)} warning(s):")
        for w in warnings:
            print(f"  WARN {w}")

    if problems:
        print(f"\n{len(problems)} PROBLEM(S) — reasons:")
        for p in problems:
            print(f"  X {p}")
            hint = _hint(p)
            if hint:
                print(f"      -> {hint}")
        # one-line failure summary grouped by kind
        kinds: dict[str, int] = {}
        for p in problems:
            kinds[_kind(p)] = kinds.get(_kind(p), 0) + 1
        summary = ", ".join(f"{v}x {k}" for k, v in sorted(kinds.items()))
        print(f"\nRESULT: FAIL — {len(problems)} problem(s): {summary}")
        return 1

    print("\nRESULT: PASS — dataset loads and all sampled episodes are consistent.")

    # even on PASS, surface anything that could still hide a failure
    caveats: list[str] = []
    if checked < n_eps:
        caveats.append(f"only {checked}/{n_eps} episodes sampled — rerun with "
                       f"--episodes {n_eps} for full coverage")
    if src_df is None and args.source == "hf" and not args.no_source_check:
        caveats.append("source cross-check SKIPPED (source parquets not on "
                       "disk) — faithfulness vs the download was NOT verified")
    if args.no_source_check:
        caveats.append("source cross-check disabled via --no-source-check — "
                       "faithfulness vs the download was NOT verified")
    if args.source == "raw":
        caveats.append("raw-source build: no parquet ground truth exists, so "
                       "only internal consistency was checked")
    if warnings:
        caveats.append(f"{len(warnings)} warning(s) above — not failures, but "
                       f"worth a look (e.g. static scenes, missing language)")
    if caveats:
        print("Potential weak spots (not failures):")
        for c in caveats:
            print(f"  - {c}")
    return 0


def _kind(problem: str) -> str:
    """Short category label for a problem message."""
    for key, label in (
        ("image ~constant", "constant-image"),
        ("image dtype", "image-dtype"),
        ("image shape", "image-shape"),
        ("!= source parquet", "source-mismatch"),
        ("parquet rows", "row-count-mismatch"),
        ("language", "language-mismatch"),
        ("NaN/Inf", "nan-inf"),
        ("all zero", "all-zero"),
        ("is_first", "rlds-flags"),
        ("is_last", "rlds-flags"),
        ("is_terminal", "rlds-flags"),
        ("discount", "rlds-flags"),
        ("step(s)", "too-few-steps"),
    ):
        if key in problem:
            return label
    return "other"


def _hint(problem: str) -> str:
    """Likely cause / what to do, per problem category."""
    return {
        "constant-image":
            "black/frozen frames. Often a dead or covered camera in the SOURCE "
            "recording (known for some episodes) — decode the source mp4 "
            "segment to confirm; if the source is also black, the conversion "
            "is faithful and the episode should just be excluded from training.",
        "source-mismatch":
            "state/action differs from the download parquets — episode/row "
            "misalignment in the conversion. Rebuild after deleting the "
            "output dir; if it persists, this is a builder bug.",
        "row-count-mismatch":
            "steps dropped during conversion (missing frames in the video?) — "
            "check the build log for skipped-episode warnings.",
        "language-mismatch":
            "instruction doesn't match tasks.parquet — task_index mapping bug.",
        "nan-inf":
            "corrupt numeric data — check the source parquet for the episode.",
        "all-zero":
            "sensor stream missing/zeroed for the whole episode in the source.",
        "rlds-flags":
            "step bookkeeping wrong (is_first/is_last/discount) — builder bug.",
        "image-dtype":
            "wrong dtype in serialized images — schema/builder mismatch.",
        "image-shape":
            "unexpected resolution — resize step failed or schema mismatch.",
        "too-few-steps":
            "episode nearly empty — most frames/rows were dropped; check the "
            "build log for this episode.",
    }.get(_kind(problem), "")


if __name__ == "__main__":
    sys.exit(main())
