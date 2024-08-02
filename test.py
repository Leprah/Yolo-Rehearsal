from ultralytics import YOLO

model = YOLO("yolov8n-cls.yaml")
results = model.train(data="/home/kevin/leprah/dataset", epochs=2, imgsz=640)
