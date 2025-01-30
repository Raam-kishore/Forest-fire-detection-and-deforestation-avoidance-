from ultralytics import YOLO
model= YOLO("best.pt")
model.predict(source="1.jpEg",show=True,save=True)