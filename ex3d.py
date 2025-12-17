from ultralytics import YOLO
import cv2
import torch
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

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

def get_vec(person_keypoints):
    vec=[]
    for i in range(5,17):
        vec.append(person_keypoints[i][0].item())
        vec.append(person_keypoints[i][1].item())
    return np.array(vec)

keypoints_list1 = []
keypoints_list2 = []
features_list1 = []
features_list2 = []

cap_1 = cv2.VideoCapture(video_path_1)
cap_2 = cv2.VideoCapture(video_path_2)
# 動画を開く

if not cap_1.isOpened() or not cap_2.isOpened():
    print("エラー")
    if cap_1.isOpened():cap_1.release()
    if cap_2.isOpened():cap_2.release()
    cv2.destroyAllWindows()
    exit()


while cap_1.isOpened() and cap_2.isOpened():
    success_1, frame_1 = cap_1.read()
    success_2, frame_2 = cap_2.read()

    if not success_1 and not success_2:
        break


    if success_1:
        results_1 = model(frame_1)    
        if results_1[0].keypoints is not None:
            nodes_1 = results_1[0].keypoints.xy[0]
            keypoints_list1.append(nodes_1)
            features_list1.append(get_vec(nodes_1))
           

    if success_2:
        results_2 = model(frame_2)
        if results_2[0].keypoints is not None:
            nodes_2 = results_2[0].keypoints.xy[0]
            keypoints_list2.append(nodes_2)
            features_list2.append(get_vec(nodes_2))

cap_1.release()
cap_2.release()           
fea_1 = np.vstack(features_list1)
fea_2 = np.vstack(features_list2)

print("DTW計算")
dist,path = fastdtw(fea_1,fea_2,dist = euclidean)
path = np.array(path)
print(path)
path_dict = {}

for i_a,i_b in path:
    if i_a not in path_dict:
        path_dict[i_a] = i_b

new_path = []
for i in range(len(fea_1)):
    new_path.append((i,path_dict[i]))

print(f"DTW距離:{dist:.2f}")
print(f"対応付けたフレーム数:{len(path)}")

cap_temp = cv2.VideoCapture(video_path_2)
success_temp,frame_temp = cap_temp.read()
h,w,_ = frame_temp.shape
cap_temp.release()


for idx_a,idx_b in new_path:
    combined_frame = np.full((h, w, 3), (128, 128, 128), dtype=np.uint8)
    center_x = w//2
    center_y = h//2

    if idx_a < len(keypoints_list1):
        nodes_1 = keypoints_list1[idx_a]
        waist_1_x = (nodes_1[11][0] + nodes_1[12][0]) / 2
        waist_1_y = (nodes_1[11][1] + nodes_1[12][1]) / 2
        waist_1 = torch.tensor([waist_1_x, waist_1_y])

        offset_1 = torch.tensor([0.0, 0.0]) 

        for n1, n2 in links:
            if nodes_1[n1][0] * nodes_1[n1][1] * nodes_1[n2][0] * nodes_1[n2][1] == 0:
                continue
            cv2.line( combined_frame, nodes_1[n1].to(torch.int).tolist(), nodes_1[n2].to(torch.int).tolist(), (0, 0, 255), thickness=2)            
        for i in range(5, nodes_1.size(0)):
            cv2.circle(combined_frame, (int(nodes_1[i][0]), int(nodes_1[i][1])), 5, (0, 255, 255), -1)  

    if idx_b < len(keypoints_list2):
        nodes_2 = keypoints_list2[idx_b]
        waist_2_x = (nodes_2[11][0] + nodes_2[12][0]) / 2
        waist_2_y = (nodes_2[11][1] + nodes_2[12][1]) / 2
        waist_2 = torch.tensor([waist_2_x,waist_2_y])

        offset_2 = waist_1 - waist_2                

        for n1, n2 in links:
            node_1 = nodes_2[n1].cpu() + offset_2
            node_2 = nodes_2[n2].cpu() + offset_2
            if nodes_2[n1][0] * nodes_2[n1][1] * nodes_2[n2][0] * nodes_2[n2][1] == 0:
                continue
            cv2.line(combined_frame, node_1.to(torch.int).tolist(), node_2.to(torch.int).tolist(), (255, 0, 0), thickness=2)

        for i in range(5, nodes_2.size(0)):
            cv2.circle(combined_frame, (int(nodes_2[i][0]+offset_2[0]), int(nodes_2[i][1]+offset_2[1])), 5, (0, 255, 255), -1)
    
    
    cv2.imshow("", combined_frame)

    if cv2.waitKey(20) == 27:
        break
            # ESCが押されれば終了
cv2.destroyAllWindows()