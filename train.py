from ultralytics import YOLO

# Load a model
model = YOLO('models/yolov8_c2fnext_gs_cspssfpn.yaml')  # build a new model from YAML

# model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
# model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

# Train the model
model.train(data='data/URPC_20.yaml', epochs=300, imgsz=640, batch=16)
