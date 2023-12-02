from ultralytics import YOLO
import os
model = YOLO('yolov8n.yaml')
model.train(data=os.path.abspath("data.yaml"), epochs=100, batch=64, imgsz=640, optimizer='Adam')