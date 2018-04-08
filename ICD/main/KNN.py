import numpy as np
import  cv2 as cv
import matplotlib.pyplot as plot
import operator
import os
imgListt = np.full((28,28), 0)
testImg = cv.imread("E:/cut_ImgsTest/28pix/T"+str(7)+".jpg")
for index1 in range(len(testImg)):
    for index2 in range(len(testImg[index1])):
        if(testImg[index1][index2][2] < 90):
            imgListt[index1][index2] = 16
imgListt = imgListt.reshape((-1,784))
plot.imshow(testImg)
plot.show()

dicList = []
for i in range(34,60):
    imgA = cv.imread("svmImgs/"+str(i)+".jpg")
    imgList = np.full((28,28),0)
    for index1 in range(len(imgA)):
        for index2 in range(len(imgA[index1])):
            if(imgA[index1][index2][2] < 90):
                imgList[index1][index2] = 16
    targetImgA = imgList.reshape((-1,784))
    dicList.append(np.linalg.norm(imgListt - targetImgA))

print(dicList)

minV = dicList.index(min(dicList))

print("最小距离：",min(dicList))
print("最小位置:",minV)

plot.imshow(cv.imread("svmImgs/"+str(minV+34)+".jpg"))
plot.show()

# Y = np.array([0,1,2,3,4,5,6,7,8,9,0,2,3,4,5,7,8,9,0,2,3,4,5,7,8,9,0,0,3,4,5,7,8,9,0,4,3,9,5,8,6,6,7,2,0,4,3,9,5,8,7,2])
Y = np.array([0,4,3,9,5,8,6,6,7,2,0,4,3,9,5,8,7,2,0,4,3,9,5,8,7,0])

print("预测值:",Y[minV])

