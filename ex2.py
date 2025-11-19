from ultralytics import YOLO
import cv2
import torch

video_path = "ex2.mp4"

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


cap = cv2.VideoCapture(video_path)
# 動画を開く
cnt = 0
# フレーム番号を0にする

while cap.isOpened():
    success, frame = cap.read()
    # フレームを読み出す

    if success:
        # 読み出しに成功すれば以下を実行する
        results = model(frame)


        nodes = results[0].keypoints.xy[0]

        for n1, n2 in links:
            # 誤認識のリンクを描画しない．
            if nodes[n1][0] * nodes[n1][1] * nodes[n2][0] * nodes[n2][1] == 0:
                continue
            cv2.line(
                frame,
                # 2つの座標を整数化し，テンソルからリストにする．
                nodes[n1].to(torch.int).tolist(),
                nodes[n2].to(torch.int).tolist(),
                (0, 0, 255),
                thickness=2,
            )

        for i in range(5,nodes.size(0)):#5から始める
            cv2.circle(frame,(int(nodes[i][0]),int(nodes[i][1])),5,(0,255,255),-1)

        cv2.imshow("", frame)

        if cv2.waitKey(20) == 27:
            break
        # ESCが押されれば終了

        
    else:
        break

cap.release()
cv2.destroyAllWindows()
