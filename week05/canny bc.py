# !/usr/bin/env python
# encoding=gbk
"""
��С��
canny
"""
import cv2
import numpy as np

img = cv2.imread("lenna.png", 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("canny", cv2.Canny(gray, 100, 200))
cv2.waitKey()
cv2.destroyAllWindows()
