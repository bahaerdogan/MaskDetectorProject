from ultralytics import YOLO


model = YOLO("best.pt")

results = model(source="0", show=True)

for result in results:
    boxes =result.boxes
    classes= result.names