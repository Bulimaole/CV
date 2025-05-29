"""
陈小虎
透视变换
"""

import cv2
import numpy as np
img= cv2.imread('photo1.jpg')

result3 = img.copy()

src = np.float32([[207, 151], [517,258], [17, 601], [343, 731]])
dst = np.float32([[0, 0], [512, 0], [0, 512], [512, 512]])
print(img.shape)

m = cv2.getPerspectiveTransform(src, dst)
print("warpMatrix:")
print(m)
result = cv2.warpPerspective(result3, m, (512, 512))
cv2.imshow("src", img)
cv2.imshow("result", result)
cv2.waitKey(0)