from ultralytics import YOLO

model = YOLO('yolov8x.pt')
model.train(data='./yolo/data.yaml', batch=2, epochs=400)
model.val()