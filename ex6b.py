import numpy as np
import matplotlib.pyplot as plt
import cv2

pts1 = np.array([(959, 67), (693, 140), (1088, 114), (1157, 424)], dtype=np.float32)
pts2 = np.array([(3140, 227), (2723, 575), (3143, 575), (2721, 1625)], dtype=np.float32)

M = cv2.getPerspectiveTransform(pts1, pts2)
np.set_printoptions(precision=5, suppress=True)
print(M)

def transform_pt(pt, M):
    pt = np.append(pt, 1.0)
    pt = np.dot(M, pt)  
    pt = pt / pt[2]     
    pt = pt[:2]         
    return pt

img_court = cv2.imread("soccer_field.jpg") 

player = np.array([
    (82, 178),
    (145, 294),
    (315, 171),
    (358, 246),
    (324, 408),
    (408, 383),
    (494, 260),
    (576, 363),
    (504, 275),
    (683, 423),
    (687, 558),
    (744, 489),
    (943, 443),
    (795, 351),
    (829, 329),
    (686, 256),
    (740, 246),
    (647, 200),
    (564, 145),
    (1213, 221)
], dtype=np.float32)

for src_pt in player:
    dst_pt = transform_pt(src_pt, M).astype(int)

    print(f"{src_pt} ===> {dst_pt}")

    cv2.circle(img_court, (dst_pt[0], dst_pt[1]), 25, (255, 0, 255), 15)

img_court = cv2.cvtColor(img_court, cv2.COLOR_BGR2RGB)
plt.imshow(img_court)
plt.show()