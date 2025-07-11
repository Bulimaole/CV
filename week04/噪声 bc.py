"""
陈小虎

噪声
"""
import cv2 as cv
import numpy as np
from PIL import Image
from skimage import util

img = cv.imread("lenna.png")
noise_gs_img=util.random_noise(img,mode='poisson')

cv.imshow("source", img)
cv.imshow("lenna",noise_gs_img)
cv.imwrite('lenna noise.png',noise_gs_img)
cv.waitKey(0)
cv.destroyAllWindows()
