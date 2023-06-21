from ultralytics import YOLO

pt_path = 'yolov8s.pt'
model = YOLO(pt_path)
model.export(format='onnx', opset=12, simplify=True, imgsz=640)