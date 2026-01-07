import numpy as np
import matplotlib.pyplot as plt
import cv2

pts1 = np.array([(699, 148),(1082, 122),(1216, 176),(1157, 424)], dtype=np.float32)
pts2 = np.array([(2708, 594), (3143, 594),(3143, 858),(2721, 1625)], dtype=np.float32)

M = cv2.getPerspectiveTransform(pts1, pts2)
np.set_printoptions(precision=5, suppress=True)
print(M)

img1 = cv2.imread("ex4-25.jpg", cv2.IMREAD_COLOR)
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

w2,h2 = pts2.max(axis=0).astype(int) 

h2+=750

img2 = cv2.warpPerspective(img1, M, (w2, h2))

def transform_pt(pt, M):
    pt = np.append(pt, 1.0)
    pt = np.dot(M, pt)  
    pt = pt / pt[2] 
    pt = pt[:2]  
    return pt

fig = plt.figure(figsize=(8, 8))
fig.add_subplot(1, 2, 1).imshow(img1)
fig.add_subplot(1, 2, 2).imshow(img2)
plt.show()