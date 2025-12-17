from ultralytics import YOLO
import cv2
from ultralytics import YOLO
import cv2
import torch
import numpy as np

# YOLOv8nモデルをロード
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture('ex5.mp4') 

if not cap.isOpened():
    exit()

while cap.isOpened():
    success, frame = cap.read()
    
    if not success:
        print("動画の最後まで到達しました。")
        break

    results = model(frame,conf = 0.35)

    for r in results:
        boxes = r.boxes
        
        for box in boxes:
            xy1 = box.data[0][0:2]
            xy2 = box.data[0][2:4]
            
            cv2.rectangle(
                frame,
                xy1.to(torch.int).tolist(),
                xy2.to(torch.int).tolist(),
                (0, 0, 255), 
                thickness=3,
            )

    cv2.imshow("", frame)



    if cv2.waitKey(10) == 27:
        break
        # ESCが押されれば終了

        
# 終了処理
cap.release()
cv2.destroyAllWindows()