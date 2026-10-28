"""
Merge the background negative samples into the training split, backing up the
original split first.
"""

import os
import shutil
from pathlib import Path
import datetime


def merge_background_to_train(
    dataset_dir,
    bg_img_dir=None,
    bg_lbl_dir=None,
    backup=True,
):
    """
    Merge the background images and their empty labels into the training split.

    Args:
        dataset_dir: dataset root (cucumber_new)
        bg_img_dir: directory of the background images
        bg_lbl_dir: directory of the background labels
        backup: whether to back up the original training split
    """
    dataset_dir = Path(dataset_dir)
    train_img_dir = dataset_dir / "train" / "images"
    train_lbl_dir = dataset_dir / "train" / "labels"

    if bg_img_dir is None:
        bg_img_dir = dataset_dir / "background" / "images"
    if bg_lbl_dir is None:
        bg_lbl_dir = dataset_dir / "background" / "labels"

    bg_img_dir = Path(bg_img_dir)
    bg_lbl_dir = Path(bg_lbl_dir)

    if not bg_img_dir.exists():
        print(f"error: background image directory not found: {bg_img_dir}")
        return

    # backup
    if backup:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = dataset_dir / f"train_backup_{timestamp}"
        print(f"[1/3] backing up the training split to: {backup_dir}")
        shutil.copytree(train_img_dir, backup_dir / "images")
        shutil.copytree(train_lbl_dir, backup_dir / "labels")
        print(f"      backup done: {backup_dir}")

    # merge
    bg_images = sorted([
        f for f in os.listdir(str(bg_img_dir))
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
    ])
    bg_labels = sorted([
        f for f in os.listdir(str(bg_lbl_dir))
        if f.endswith('.txt')
    ])

    print(f"\n[2/3] merging background samples...")
    print(f"      background images: {len(bg_images)}")
    print(f"      background labels: {len(bg_labels)}")

    copied_img = 0
    copied_lbl = 0

    for img_file in bg_images:
        src = bg_img_dir / img_file
        dst = train_img_dir / img_file
        if not dst.exists():
            shutil.copy2(str(src), str(dst))
            copied_img += 1

    for lbl_file in bg_labels:
        src = bg_lbl_dir / lbl_file
        dst = train_lbl_dir / lbl_file
        if not dst.exists():
            shutil.copy2(str(src), str(dst))
            copied_lbl += 1

    print(f"      images copied: {copied_img}")
    print(f"      labels copied: {copied_lbl}")

    # count the merged dataset
    print(f"\n[3/3] counting the merged dataset...")
    final_images = len([
        f for f in os.listdir(str(train_img_dir))
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
    ])
    final_labels = len([
        f for f in os.listdir(str(train_lbl_dir))
        if f.endswith('.txt')
    ])

    empty_labels = 0
    for f in os.listdir(str(train_lbl_dir)):
        path = train_lbl_dir / f
        if os.path.getsize(str(path)) == 0:
            empty_labels += 1

    print(f"      training images in total: {final_images}")
    print(f"      training labels in total: {final_labels}")
    print(f"      empty labels (pure background): {empty_labels} ({empty_labels/final_images*100:.1f}%)")

    print(f"\n{'='*60}")
    print(f"  merge finished")
    print(f"{'='*60}")
    print(f"  original split backed up to: {dataset_dir}/train_backup_{timestamp}/")
    print(f"  training can start from: {dataset_dir}/cucumber_dataset.yaml")


if __name__ == "__main__":
    # Dataset root. Override with the environment variable CUCUMBER_DATASET_DIR,
    # otherwise a relative path is assumed (see the README).
    DATASET_DIR = os.environ.get("CUCUMBER_DATASET_DIR", "datasets/cucumber_new")

    merge_background_to_train(
        dataset_dir=DATASET_DIR,
        backup=True,
    )
