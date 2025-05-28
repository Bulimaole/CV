"""
@陈小虎

最临近插值
"""
import cv2
import numpy as np
def function(img):
    # height,width,channels =img.shape
    # emptyImage=np.zeros((600,600,channels),np.uint8)
    # sh=600/height
    # sw=600/width
    # for i in range(600):
    #     for j in range(600):
    #         x=int(i/sh + 0.5)
    #         y=int(j/sw + 0.5)
    #         emptyImage[i,j]=img[x,y]
    # return emptyImage

      resized_img = cv2.resize(img ,(600,600),interpolation=cv2.INTER_NEAREST)
      return resized_img


img=cv2.imread("lenna.png")
zoom=function(img)
print(zoom)
print(zoom.shape)
cv2.imshow("nearest interp",zoom)
cv2.imshow("image",img)
cv2.waitKey(0)
