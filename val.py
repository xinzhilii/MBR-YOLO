from ultralytics import YOLO

# Load a model
# model = YOLO('yolov8n.pt')  # load an official model
model = YOLO('runs/detect/train4/weights/best.pt')  # load a custom model

# Validate the model
metrics = model.val(data='data/URPC_20.yaml', imgsz=640, batch=2, workers=0)  # no arguments needed, dataset and settings remembered
metrics.box.map    # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps   # a list contains map50-95 of each category