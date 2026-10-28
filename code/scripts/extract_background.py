"""
从训练图片中提取背景区域，生成纯背景负样本。

策略：
1. 读取每张训练图的 YOLO 标注 (归一化 bbox)
2. 将 bbox 区域视为"被占用的前景"
3. 从非占用区域随机裁取 patch，作为背景样本
4. 可选：也保存将黄瓜区域涂黑的完整图
"""

import cv2
import numpy as np
import os
import random
from pathlib import Path
from collections import defaultdict


def yolo_to_pixel(bbox, img_w, img_h):
    """YOLO 归一化坐标 -> 像素坐标 (x1, y1, x2, y2)"""
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
    """两个 bbox 是否重叠 (像素坐标)"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    # 不重叠的判断
    if ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1:
        return False
    return True


def compute_iou(a, b):
    """计算两个 bbox 的 IoU"""
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
    """检查 patch 是否与任何黄瓜 bbox 有显著重叠"""
    for cb in cucumber_bboxes:
        iou = compute_iou(patch_bbox, cb)
        if iou > iou_thresh:
            return True
    return False


def create_background_mask(img_h, img_w, cucumber_bboxes, expand_ratio=0.15):
    """
    创建背景 mask，1=背景可裁切区域，0=黄瓜占用区域。
    expand_ratio: 将 bbox 向外扩展 N%，避免裁到黄瓜边缘。
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
    主函数：从训练图中提取背景 patch。

    Args:
        image_dir: 训练图片目录
        label_dir: YOLO 标签目录
        output_img_dir: 输出背景图片目录
        output_label_dir: 输出标签目录（都是空文件）
        num_patches_per_image: 每张图最多提取几个 patch
        patch_sizes: 裁切尺寸候选（随机选择）
        iou_thresh: patch 与黄瓜 bbox 的 IoU 阈值，超过则跳过
        expand_ratio: bbox 向外扩展比例（避免裁到黄瓜边缘）
        max_attempts_per_image: 每张图最多尝试几次
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
        # 找对应的 label 文件
        base = os.path.splitext(img_file)[0]
        label_file = base + ".txt"
        label_path = os.path.join(label_dir, label_file)

        if not os.path.exists(label_path):
            continue

        # 读取图片
        img = cv2.imread(img_path)
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        # 读取标签
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
                # 跳过面积太小的 bbox（可能是标注错误）
                if (x2 - x1) * (y2 - y1) > 16:
                    cucumber_bboxes.append((x1, y1, x2, y2))

        if not cucumber_bboxes:
            stats["skipped_no_cucumbers"] += 1
            continue

        stats["processed"] += 1

        # 创建背景 mask
        bg_mask = create_background_mask(img_h, img_w, cucumber_bboxes, expand_ratio)

        patches_got = 0
        attempts = 0

        while patches_got < num_patches_per_image and attempts < max_attempts_per_image:
            attempts += 1
            patch_size = random.choice(patch_sizes)

            # 确保 patch 不超出图像边界
            if img_w < patch_size or img_h < patch_size:
                patch_size = min(img_w, img_h, patch_size)

            max_x = img_w - patch_size
            max_y = img_h - patch_size
            if max_x <= 0 or max_y <= 0:
                break

            px = random.randint(0, max_x)
            py = random.randint(0, max_y)

            patch_bbox = (px, py, px + patch_size, py + patch_size)

            # 检查是否与黄瓜重叠
            if patch_overlaps_any(patch_bbox, cucumber_bboxes, iou_thresh):
                continue

            # 检查 mask 区域内背景像素占比
            patch_mask = bg_mask[py:py+patch_size, px:px+patch_size]
            bg_ratio = patch_mask.mean()
            if bg_ratio < 0.85:  # 至少 85% 是纯背景
                continue

            # 裁切并保存
            patch = img[py:py+patch_size, px:px+patch_size]
            out_name = f"bg_{base}_{patch_size}_{px}_{py}.jpg"
            out_img_path = os.path.join(output_img_dir, out_name)
            cv2.imwrite(out_img_path, patch)

            # 创建空标签文件
            out_label_path = os.path.join(output_label_dir, out_name.replace(".jpg", ".txt"))
            open(out_label_path, "w").close()

            patches_got += 1
            stats["patches_created"] += 1
            stats["sizes_used"][str(patch_size)] += 1

        if patches_got == 0:
            stats["skipped_no_space"] += 1

        # 进度打印
        if stats["processed"] % 200 == 0:
            print(f"  已处理 {stats['processed']}/{stats['total_images']} 张图, "
                  f"已创建 {stats['patches_created']} 个背景 patch")

    return stats


# ============================================================
# 主流程
# ============================================================
if __name__ == "__main__":
    random.seed(42)

    # 路径配置
    DATASET_DIR = Path(r"D:\Yolo\ultralytics-8.3.163\datasets\cucumber_new")

    TRAIN_IMG_DIR = DATASET_DIR / "train" / "images"
    TRAIN_LBL_DIR = DATASET_DIR / "train" / "labels"

    BG_IMG_DIR = DATASET_DIR / "background" / "images"
    BG_LBL_DIR = DATASET_DIR / "background" / "labels"

    print("=" * 60)
    print("  背景负样本提取工具")
    print("=" * 60)
    print(f"\n  源图片目录: {TRAIN_IMG_DIR}")
    print(f"  源标签目录:  {TRAIN_LBL_DIR}")
    print(f"  输出图片:    {BG_IMG_DIR}")
    print(f"  输出标签:    {BG_LBL_DIR}\n")

    # 第二步：提取背景 patch
    print("[1/2] 提取背景 patch...")
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
    print(f"  提取完成!")
    print(f"{'='*60}")
    print(f"  处理图片:         {stats['processed']}")
    print(f"  无黄瓜跳过:       {stats['skipped_no_cucumbers']}")
    print(f"  空间不足跳过:     {stats['skipped_no_space']}")
    print(f"  创建背景 patch:   {stats['patches_created']}")
    print(f"  尺寸分布:")
    for sz, cnt in sorted(stats['sizes_used'].items(), key=lambda x: -x[1]):
        print(f"    {sz}px: {cnt} 张")

    print(f"\n  输出目录:")
    print(f"    图片: {BG_IMG_DIR}")
    print(f"    标签: {BG_LBL_DIR}")
    print(f"\n  下一步: 将背景图片和空标签合并到训练集中")
    print(f"  合并命令示例 (Windows PowerShell):")
    print(f"    Copy-Item '{BG_IMG_DIR}\\*' '{TRAIN_IMG_DIR}'")
    print(f"    Copy-Item '{BG_LBL_DIR}\\*' '{TRAIN_LBL_DIR}'")
