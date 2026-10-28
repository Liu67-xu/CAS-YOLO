"""
Extract background regions from the training images to build background
negative samples.

Strategy:
1. read the YOLO annotation of each training image (normalised boxes)
2. treat the box regions as "occupied foreground"
3. crop patches at random from the unoccupied region and keep them as background
   samples
4. optionally, also save the full image with the cucumber regions blacked out
"""

import cv2
import numpy as np
import os
import random
from pathlib import Path
from collections import defaultdict


def yolo_to_pixel(bbox, img_w, img_h):
    """YOLO normalised coordinates -> pixel coordinates (x1, y1, x2, y2)."""
    cx, cy, w, h = bbox
    x1 = int((cx - w / 2) * img_w)
    y1 = int((cy - h / 2) * img_h)
    x2 = int((cx + w / 2) * img_w)
    y2 = int((cy + h / 2) * img_h)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_w, x2)
    y2 = min(img_h, y2)
    return x1, y1, x2, y2


def bboxes_overlap(a, b):
    """Whether two boxes overlap (pixel coordinates)."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    # separated along at least one axis
    if ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1:
        return False
    return True


def compute_iou(a, b):
    """IoU of two boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter_area / min(area_a, area_b)


def patch_overlaps_any(patch_bbox, cucumber_bboxes, iou_thresh=0.05):
    """Whether the patch overlaps any cucumber box above the IoU threshold."""
    for cb in cucumber_bboxes:
        iou = compute_iou(patch_bbox, cb)
        if iou > iou_thresh:
            return True
    return False


def create_background_mask(img_h, img_w, cucumber_bboxes, expand_ratio=0.15):
    """
    Build a background mask: 1 = free for cropping, 0 = occupied by cucumber.
    expand_ratio: grow each box by this fraction so that the margins of a
    cucumber are not cropped as background.
    """
    mask = np.ones((img_h, img_w), dtype=np.uint8)
    for x1, y1, x2, y2 in cucumber_bboxes:
        bw, bh = x2 - x1, y2 - y1
        ex = int(bw * expand_ratio)
        ey = int(bh * expand_ratio)
        x1e = max(0, x1 - ex)
        y1e = max(0, y1 - ey)
        x2e = min(img_w, x2 + ex)
        y2e = min(img_h, y2 + ey)
        mask[y1e:y2e, x1e:x2e] = 0
    return mask


def extract_background_patches(
    image_dir,
    label_dir,
    output_img_dir,
    output_label_dir,
    num_patches_per_image=3,
    patch_sizes=(320, 416, 512, 640),
    iou_thresh=0.03,
    expand_ratio=0.12,
    max_attempts_per_image=50,
):
    """
    Main routine: extract background patches from the training images.

    Args:
        image_dir: directory of the training images
        label_dir: directory of the YOLO labels
        output_img_dir: output directory for the background images
        output_label_dir: output directory for the labels (all empty files)
        num_patches_per_image: maximum number of patches taken from one image
        patch_sizes: candidate crop sizes, chosen at random
        iou_thresh: IoU threshold against the cucumber boxes; above it the patch
            is discarded
        expand_ratio: how far each box is grown, to keep cucumber edges out
        max_attempts_per_image: maximum number of draws per image
    """
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    image_files = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
    ])

    stats = {
        "total_images": len(image_files),
        "processed": 0,
        "skipped_no_cucumbers": 0,
        "skipped_no_space": 0,
        "patches_created": 0,
        "sizes_used": defaultdict(int),
    }

    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        # matching label file
        base = os.path.splitext(img_file)[0]
        label_file = base + ".txt"
        label_path = os.path.join(label_dir, label_file)

        if not os.path.exists(label_path):
            continue

        # read the image
        img = cv2.imread(img_path)
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        # read the labels
        cucumber_bboxes = []
        with open(label_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                cls_id = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])
                x1, y1, x2, y2 = yolo_to_pixel([cx, cy, w, h], img_w, img_h)
                # skip boxes that are too small to be real annotations
                if (x2 - x1) * (y2 - y1) > 16:
                    cucumber_bboxes.append((x1, y1, x2, y2))

        if not cucumber_bboxes:
            stats["skipped_no_cucumbers"] += 1
            continue

        stats["processed"] += 1

        # build the background mask
        bg_mask = create_background_mask(img_h, img_w, cucumber_bboxes, expand_ratio)

        patches_got = 0
        attempts = 0

        while patches_got < num_patches_per_image and attempts < max_attempts_per_image:
            attempts += 1
            patch_size = random.choice(patch_sizes)

            # keep the patch inside the image
            if img_w < patch_size or img_h < patch_size:
                patch_size = min(img_w, img_h, patch_size)

            max_x = img_w - patch_size
            max_y = img_h - patch_size
            if max_x <= 0 or max_y <= 0:
                break

            px = random.randint(0, max_x)
            py = random.randint(0, max_y)

            patch_bbox = (px, py, px + patch_size, py + patch_size)

            # reject patches that overlap a cucumber
            if patch_overlaps_any(patch_bbox, cucumber_bboxes, iou_thresh):
                continue

            # require that the patch is almost entirely background
            patch_mask = bg_mask[py:py+patch_size, px:px+patch_size]
            bg_ratio = patch_mask.mean()
            if bg_ratio < 0.85:  # at least 85 % background
                continue

            # crop and save
            patch = img[py:py+patch_size, px:px+patch_size]
            out_name = f"bg_{base}_{patch_size}_{px}_{py}.jpg"
            out_img_path = os.path.join(output_img_dir, out_name)
            cv2.imwrite(out_img_path, patch)

            # empty label file
            out_label_path = os.path.join(output_label_dir, out_name.replace(".jpg", ".txt"))
            open(out_label_path, "w").close()

            patches_got += 1
            stats["patches_created"] += 1
            stats["sizes_used"][str(patch_size)] += 1

        if patches_got == 0:
            stats["skipped_no_space"] += 1

        # progress
        if stats["processed"] % 200 == 0:
            print(f"  processed {stats['processed']}/{stats['total_images']} images, "
                  f"{stats['patches_created']} background patches created")

    return stats


# ============================================================
# main
# ============================================================
if __name__ == "__main__":
    random.seed(42)

    # Dataset root. Override with the environment variable CUCUMBER_DATASET_DIR,
    # otherwise a relative path is assumed (see the README).
    DATASET_DIR = Path(os.environ.get("CUCUMBER_DATASET_DIR", "datasets/cucumber_new"))

    TRAIN_IMG_DIR = DATASET_DIR / "train" / "images"
    TRAIN_LBL_DIR = DATASET_DIR / "train" / "labels"

    BG_IMG_DIR = DATASET_DIR / "background" / "images"
    BG_LBL_DIR = DATASET_DIR / "background" / "labels"

    print("=" * 60)
    print("  background negative-sample extraction")
    print("=" * 60)
    print(f"\n  source images: {TRAIN_IMG_DIR}")
    print(f"  source labels: {TRAIN_LBL_DIR}")
    print(f"  output images: {BG_IMG_DIR}")
    print(f"  output labels: {BG_LBL_DIR}\n")

    # step 1: extract the background patches
    print("[1/2] extracting background patches...")
    stats = extract_background_patches(
        image_dir=str(TRAIN_IMG_DIR),
        label_dir=str(TRAIN_LBL_DIR),
        output_img_dir=str(BG_IMG_DIR),
        output_label_dir=str(BG_LBL_DIR),
        num_patches_per_image=4,
        patch_sizes=[320, 416, 512, 640],
        iou_thresh=0.03,
        expand_ratio=0.12,
        max_attempts_per_image=80,
    )

    print(f"\n{'='*60}")
    print(f"  extraction finished")
    print(f"{'='*60}")
    print(f"  images processed:      {stats['processed']}")
    print(f"  skipped, no cucumber:  {stats['skipped_no_cucumbers']}")
    print(f"  skipped, no space:     {stats['skipped_no_space']}")
    print(f"  background patches:    {stats['patches_created']}")
    print(f"  size distribution:")
    for sz, cnt in sorted(stats['sizes_used'].items(), key=lambda x: -x[1]):
        print(f"    {sz}px: {cnt}")

    print(f"\n  output directories:")
    print(f"    images: {BG_IMG_DIR}")
    print(f"    labels: {BG_LBL_DIR}")
    print(f"\n  next: merge the background images and the empty labels into the")
    print(f"  training split with merge_background.py")
