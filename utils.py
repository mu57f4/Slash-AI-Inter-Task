import numpy as np
import cv2
from ultralytics import YOLO

def read_image(uploaded_file):
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        return image
    
model = YOLO("yolo-Weights/yolov8m.pt")
def predict(img):
    preds = []
    results = model(img, stream=True)
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            cls_name = result.names[cls]
            preds.append(cls_name)
    
    return preds