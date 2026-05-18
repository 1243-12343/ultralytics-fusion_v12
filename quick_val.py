from ultralytics import YOLO
import torch

torch.use_deterministic_algorithms(False)

if __name__ == '__main__':
    model = YOLO('runs_second/yolov12nir-rgb-base-p2di-p3-hcfm16/weights/best.pt')

    print("Starting validation...")
    results = model.val(data=r'F:\datasets\mydata.yaml', batch=8, imgsz=512)

    print('\n=== Validation Results ===')
    print(f'mAP50: {results.box.map50:.4f}')
    print(f'mAP50-95: {results.box.map:.4f}')
