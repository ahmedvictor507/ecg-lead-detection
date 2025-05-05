# ECG Lead Detector using YOLOv8

This project trains a YOLOv8 model to detect ECG lead locations from ECG chart images. Useful in digitizing and analyzing clinical ECG data.

## 📦 Model

- **Model**: YOLOv8s (`ultralytics`)
- **Epochs**: 300
- **Image Size**: 640x640
- **Batch Size**: 8
- **Augmentations**: Rotation, scale, shear, mixup, mosaic
- **Optimizer**: AdamW
- **Exports**: TFLite (Android), CoreML (iOS), ONNX, PyTorch

## 🏁 Training Command

```python
from ultralytics import YOLO

model = YOLO('yolov8s.pt')
model.train(data='data.yaml', ...)
