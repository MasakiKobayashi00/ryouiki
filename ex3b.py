from ultralytics import YOLO
import cv2
import torch
import numpy as np

video_path_1 = "ex3a.mp4"
video_path_2 = "ex3b.mp4"

model = YOLO("yolo11x-pose.pt")

links = [
            [5, 7],
            [6, 8],
            [7, 9],
            [8, 10],
            [11, 13],
            [12, 14],
            [13, 15],
            [14, 16],
            [5, 11],
            [6, 12],
            [5, 6],
            [11, 12]
        ]


cap_1 = cv2.VideoCapture(video_path_1)
cap_2 = cv2.VideoCapture(video_path_2)
# 動画を開く
cnt = 0

while cap_1.isOpened() and cap_2.isOpened():
    success_1, frame_1 = cap_1.read()
    success_2, frame_2 = cap_2.read()
        
    # 両方の動画が終了したらループを抜ける
    if not success_2:
        break

    h,w,_ = frame_2.shape
    combined_frame = np.full((h, w, 3), (128, 128, 128), dtype=np.uint8)

    if success_1:
        results_1 = model(frame_1)    

        if results_1[0].keypoints is not None:
            nodes_1 = results_1[0].keypoints.xy[0]

            for n1, n2 in links:
                if nodes_1[n1][0] * nodes_1[n1][1] * nodes_1[n2][0] * nodes_1[n2][1] == 0:
                    continue
                cv2.line( combined_frame, nodes_1[n1].to(torch.int).tolist(), nodes_1[n2].to(torch.int).tolist(), (0, 0, 255), thickness=2)
                
            for i in range(5, nodes_1.size(0)):
                cv2.circle(combined_frame, (int(nodes_1[i][0]), int(nodes_1[i][1])), 5, (0, 255, 255), -1)
                

    if success_2:

        results_2 = model(frame_2)

        if results_2[0].keypoints is not None:

            nodes_2 = results_2[0].keypoints.xy[0]
            for n1, n2 in links:
                if nodes_2[n1][0] * nodes_2[n1][1] * nodes_2[n2][0] * nodes_2[n2][1] == 0:
                    continue
                cv2.line(combined_frame, nodes_2[n1].to(torch.int).tolist(), nodes_2[n2].to(torch.int).tolist(), (0, 0, 255), thickness=2)

            for i in range(5, nodes_2.size(0)):
                cv2.circle(combined_frame, (int(nodes_2[i][0]), int(nodes_2[i][1])), 5, (0, 255, 255), -1)

        
        
        cv2.imshow("", combined_frame)

        if cv2.waitKey(20) == 27:
            break
            # ESCが押されれば終了
    else:
        break


cap_1.release()
cap_2.release()
cv2.destroyAllWindows()
