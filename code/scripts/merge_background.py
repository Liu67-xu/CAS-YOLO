"""
合并背景负样本到训练集，并备份原始训练集。
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
    将背景图片和空标签合并到训练集中。

    Args:
        dataset_dir: 数据集根目录 (cucumber_new)
        bg_img_dir: 背景图片目录
        bg_lbl_dir: 背景标签目录
        backup: 是否备份原始训练集
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
        print(f"错误: 背景图片目录不存在: {bg_img_dir}")
        return

    # 备份
    if backup:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = dataset_dir / f"train_backup_{timestamp}"
        print(f"[1/3] 备份训练集到: {backup_dir}")
        shutil.copytree(train_img_dir, backup_dir / "images")
        shutil.copytree(train_lbl_dir, backup_dir / "labels")
        print(f"      备份完成: {backup_dir}")

    # 合并
    bg_images = sorted([
        f for f in os.listdir(str(bg_img_dir))
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
    ])
    bg_labels = sorted([
        f for f in os.listdir(str(bg_lbl_dir))
        if f.endswith('.txt')
    ])

    print(f"\n[2/3] 合并背景样本...")
    print(f"      背景图片: {len(bg_images)} 张")
    print(f"      背景标签: {len(bg_labels)} 个")

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

    print(f"      复制图片: {copied_img} 张")
    print(f"      复制标签: {copied_lbl} 个")

    # 统计合并后的数据集
    print(f"\n[3/3] 统计合并后数据集...")
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

    print(f"      训练图片总数: {final_images}")
    print(f"      训练标签总数: {final_labels}")
    print(f"      空标签(纯背景): {empty_labels} ({empty_labels/final_images*100:.1f}%)")

    print(f"\n{'='*60}")
    print(f"  合并完成!")
    print(f"{'='*60}")
    print(f"  原始训练集: {dataset_dir}/train_backup_{timestamp}/")
    print(f"  可直接用: {dataset_dir}/cucumber_dataset.yaml 开始训练")


if __name__ == "__main__":
    DATASET_DIR = r"D:\Yolo\ultralytics-8.3.163\datasets\cucumber_new"

    merge_background_to_train(
        dataset_dir=DATASET_DIR,
        backup=True,
    )
