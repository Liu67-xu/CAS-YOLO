"""
Example training script for CAS-YOLO, mirroring the configuration used for the
results reported in the paper.

  1. pip install ultralytics==8.3.163
  2. copy the files in code/ultralytics/ over the corresponding paths of the
     installed package (block.py, tasks.py, __init__.py and Attmodules/)
  3. place the source images next to the annotations, run code/set_path.py
  4. python code/train_example.py
"""
from pathlib import Path

from ultralytics import YOLO

HERE = Path(__file__).resolve().parent  # <repository>/code

if __name__ == "__main__":
    model = YOLO(str(HERE / "configs" / "yolov8_C2fAtt_CA_asff.yaml"))
    model.train(
        data=str(HERE / "configs" / "cucumber_dataset.yaml"),
        epochs=100,
        patience=30,
        batch=20,
        imgsz=640,
        optimizer="auto",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        cos_lr=False,
        close_mosaic=15,
        degrees=8.0,
        translate=0.08,
        scale=0.5,
        fliplr=0.5,
        mosaic=0.7,
        mixup=0.0,
        copy_paste=0.0,
        pretrained=True,
        seed=0,
        amp=True,
        cache="ram",
        workers=4,
        project="runs/cas_yolo",
        name="C2fAtt_CA_asff",
    )
