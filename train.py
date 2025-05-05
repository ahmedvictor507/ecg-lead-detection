
model.train(
    data='', # Dataset here
    epochs=300,                # ↑ Increase for better convergence
    imgsz=640,
    batch=8,                   # ↓ Reduce if you have limited GPU VRAM
    project='ecg-lead-detection',
    name='yolov8s-ecg-optimized',
    workers=2,

    # Augmentation
    degrees=10,
    scale=0.6,                # Slightly higher range
    shear=4,                  # More aggressive
    perspective=0.001,
    flipud=0.2,               # Reduced upside-down flips
    fliplr=0.5,

    # Regularization and optimization
    patience=50,              # Early stopping patience
    optimizer='AdamW',        # Better for small/medium datasets
    lr0=0.001,                # Base learning rate
    lrf=0.01,                 # Final LR (cosine decay)
    warmup_epochs=3,          # Warmup can help stabilize early training
    close_mosaic=10,          # Mosaic off after X epochs to help fine-tune
    mosaic=1.0,               # Enable full mosaic
    mixup=0.1,                # Small mixup for regularization
)
