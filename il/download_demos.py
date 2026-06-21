"""Fetch the WarehouseSort rgb demonstration datasets from the Kaggle competition into
il/demos/<level>/.

The demos (200 rgb episodes per level) are the competition data:
https://www.kaggle.com/competitions/marso-hack-berlin-2026-robot-parcel-sorting-challenge/data

  * On Kaggle: the competition data mounts under /kaggle/input/ — this finds it automatically.
  * Elsewhere: it downloads via `kagglehub` (you must have joined the competition and have a
    Kaggle API token — kaggle.com -> Settings -> Create New API Token, then put kaggle.json in
    ~/.kaggle/ or set KAGGLE_USERNAME / KAGGLE_KEY).

Either way the files are staged under il/demos/<level>/ so the trainer finds them.

  pixi run python il/download_demos.py
  pixi run python il/download_demos.py --competition <competition-slug>
"""

import argparse
import glob
import os
import shutil
import tarfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_COMPETITION = "marso-hack-berlin-2026-robot-parcel-sorting-challenge"


def _find_or_download(competition):
    """Return a directory containing the rgb demos — a mounted Kaggle input or a download."""
    for p in glob.glob("/kaggle/input/*"):
        if (glob.glob(os.path.join(p, "**/trajectory.*.pd_ee_delta_pos.physx_cuda.h5"), recursive=True)
                or glob.glob(os.path.join(p, "**/*.tar.gz"), recursive=True)):
            print("found attached Kaggle competition data:", p)
            return p
    import kagglehub
    return kagglehub.competition_download(competition)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--competition", default=DEFAULT_COMPETITION,
                    help="Kaggle competition slug")
    ap.add_argument("--dest", default=os.path.join(REPO, "il", "demos"))
    args = ap.parse_args()

    src = _find_or_download(args.competition)
    print("data at:", src)

    os.makedirs(args.dest, exist_ok=True)
    # handles either a tarball or an easy/medium/hard folder layout in the competition data
    tars = glob.glob(os.path.join(src, "**/*.tar.gz"), recursive=True)
    if tars:
        for t in tars:
            with tarfile.open(t) as tf:
                tf.extractall(args.dest)
    else:
        for h5 in glob.glob(os.path.join(src, "**/trajectory.*.pd_ee_delta_pos.physx_cuda.h5"), recursive=True):
            lvl = os.path.basename(os.path.dirname(h5))
            os.makedirs(os.path.join(args.dest, lvl), exist_ok=True)
            for f in glob.glob(h5[:-2] + "*"):   # the .h5 and its .json
                shutil.copy(f, os.path.join(args.dest, lvl, os.path.basename(f)))

    got = sorted(glob.glob(os.path.join(args.dest, "*", "trajectory.*.pd_ee_delta_pos.physx_cuda.h5")))
    print(f"staged {len(got)} level dataset(s) under {args.dest}:")
    for f in got:
        print(" ", f)


if __name__ == "__main__":
    main()
