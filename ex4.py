from ultralytics import YOLO
import cv2
import torch
import numpy as np

model = YOLO("yolov8n.pt")

results = model("ex4-25.jpg",conf = 0.1) #confは閾値

img = results[0].orig_img

boxes = results[0].boxes

for box in boxes:
    xy1 = box.data[0][0:2]
    xy2 = box.data[0][2:4]
    cv2.rectangle(
        img,
        xy1.to(torch.int).tolist(),
        xy2.to(torch.int).tolist(),
        (0,0,255),
        thickness=3,
    )

cv2.imshow("", img)
cv2.drawMarker
cv2.waitKey(0)
cv2.destroyAllWindows()
