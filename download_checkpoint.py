"""Fetch our trained checkpoints from Kaggle into checkpoints/ -- they're too large to commit
to GitHub directly, so they're hosted as a public Kaggle Dataset instead. Run this once before
eval.py / before the judge runs eval.py (see SUBMISSION.md and submission.yaml's checkpoint
paths, which point at the local checkpoints/ directory this script stages).

  pixi run python download_checkpoint.py
"""

import glob
import os
import shutil

import kagglehub

DATASET = "sahilrajpurkar/marco-checkpoint"
REPO = os.path.dirname(os.path.abspath(__file__))


def main():
    src = kagglehub.dataset_download(DATASET)
    print("data at:", src)

    dest = os.path.join(REPO, "checkpoints")
    os.makedirs(dest, exist_ok=True)
    for f in glob.glob(os.path.join(src, "**", "*.pt"), recursive=True):
        shutil.copy(f, os.path.join(dest, os.path.basename(f)))

    got = sorted(os.listdir(dest))
    print(f"staged {len(got)} checkpoint(s) under {dest}:")
    for f in got:
        print(" ", f)


if __name__ == "__main__":
    main()
