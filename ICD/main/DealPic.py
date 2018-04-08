import numpy as np
import cv2 as cv
import matplotlib.pyplot as plot
import math
from sklearn import svm
import os

#读取图片
img_data = cv.imread("imgs/qaz.jpg")

img_input = img_data.copy()

#初次滤波
for index1 in range(len(img_input)):
    for index2 in range(len(img_input[index1])):
        if(img_input[index1][index2][0]>50 and img_input[index1][index2][0] < 115 and img_input[index1][index2][1] >15 and img_input[index1][index2][1] <60 and img_input[index1][index2][2]>5 and img_input[index1][index2][2] <50):
            img_input[index1][index2] = np.array([255,255,255])
        else:
            img_input[index1][index2] = np.array([0,0,0])
plot.imshow(img_input)
plot.show()

newImg = img_input.copy()
imgTest = cv.cvtColor(newImg, cv.COLOR_RGB2GRAY)
erosion = cv.erode(imgTest, np.ones((2,2),np.uint8))   # 腐蚀
kernelf = np.ones((72, 72), np.uint8)
dilation = cv.dilate(erosion, kernelf)  # 膨胀
plot.imshow(dilation)
plot.show()


img, contoursT, hierarchy = cv.findContours(dilation, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) #cv.RETR_EXTERNAL
print("contoursT:",len(contoursT))
dictContoursT = {}
for i in range(len(contoursT)):
    xx,yy,w,h=cv.boundingRect(contoursT[i])
    dictContoursT[i] = w
    print("x=",xx," y=",yy," w=",w," h=",h)
print(dictContoursT)
maxWidth = sorted(dictContoursT, key=lambda x:dictContoursT[x])[-1]
print("maxWidth:",maxWidth)
x,y,ww,hh=cv.boundingRect(contoursT[maxWidth])
imgNew = img_data[y-20:y+hh+10,x-10:x+ww+50]
plot.imshow(imgNew)
plot.show()


erzhihua = imgNew.copy()
for index1 in range(len(erzhihua)):
    for index2 in range(len(erzhihua[index1])):
        if(erzhihua[index1][index2][2] < 130):
            erzhihua[index1][index2] = np.array([255,255,255])
        else:
            erzhihua[index1][index2] = np.array([0,0,0])
plot.imshow(erzhihua)
plot.show()
kernelf = np.ones((32, 8), np.uint8)
if len(erzhihua) > 500:
    kernelf = np.ones((64, 16), np.uint8)
erzhihua = cv.dilate(erzhihua, kernelf)  # 膨胀
plot.imshow(erzhihua)
plot.show()


erzhihua = cv.cvtColor(erzhihua, cv.COLOR_RGB2GRAY)


imgSec, contours, hierarchyk = cv.findContours(erzhihua, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

#将contours按坐标从左到右排序
print("contours-len:",len(contours))
for index1 in range(len(contours)):
    for index2 in range(index1,len(contours)):
        if contours[index1][0][0][0] > contours[index2][0][0][0]:
            contours[index1],contours[index2] = contours[index2],contours[index1]

predictNum = []
RectList = []
for iu in range(len(contours)):
    x,y,w,h = cv.boundingRect(contours[iu])
    thisImg = imgNew[y:y+h,x:x+w]
    forThisImg = thisImg.copy()
    for index1 in range(len(forThisImg)):
        for index2 in range(len(forThisImg[index1])):
            if(forThisImg[index1][index2][2] < 130):
                forThisImg[index1][index2] = np.array([255,255,255])
            else:
                forThisImg[index1][index2] = np.array([0,0,0])
    plot.imshow(forThisImg)
    plot.show()
    kernelf = np.ones((32, 8), np.uint8)
    forThisImg = cv.dilate(forThisImg, kernelf)  # 膨胀
    # plot.imshow(forThisImg)
    # plot.show()

    forThisImg = cv.cvtColor(forThisImg, cv.COLOR_RGB2GRAY)

    imgThis, contoursH, hierarchyh = cv.findContours(forThisImg, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    rect = cv.minAreaRect(contoursH[0])
    box = np.int0(cv.boxPoints(rect))
    imgDec = cv.drawContours(thisImg.copy(), [box], -1, (0, 255, 0), 0)
    rows, cols, ch = imgDec.shape
    angle = rect[2]
    print("angle:",angle)
    print("box:",box)
    # ang = math.atan2(box[2][1]-box[1][1], box[2][0]-box[1][0])
    # print("ang_fisret:",ang)
    # if ang > 0:
    #     ang = ang
    if angle < 0 and abs(angle) > 45:
        angle = abs(angle) - 90
    if abs(angle) < 45 :
        angle = abs(angle)
    if angle == 0 or angle == -0:
        angle = 0
    # print("ang:",ang)
    # if box[1][0] < box[0][0] :
    #     angle = -abs(angle)
    # if box[1][0] > box[0][0] :

    M = cv.getRotationMatrix2D((cols/2, rows/2), angle, 1.0)
    img = cv.warpAffine(imgDec, M, (cols, rows))
    cropImg = cv.resize(img,(144,224),interpolation=cv.INTER_LANCZOS4)
    for index1 in range(len(cropImg)):
        for index2 in range(len(cropImg[index1])):
            if(cropImg[index1][index2][0] ==cropImg[index1][index2][1] == cropImg[index1][index2][2] == 0):
                cropImg[index1][index2] = np.array([255,255,255])
    RectList.append(cropImg.tolist())
    nrootdir=("E:/cut_ImgsTest10/")
    if not os.path.isdir(nrootdir):
        os.makedirs(nrootdir)
    cv.imwrite( nrootdir+"T"+str(iu)+".jpg", cropImg)
    plot.imshow(img)
    plot.show()


    imgListt = np.zeros((224,144))
    testImg = cropImg.copy()
    print("testImg-length:",len(testImg))
    for index1 in range(len(testImg)):
        for index2 in range(len(testImg[index1])):
            if(testImg[index1][index2][2] < 90):
                imgListt[index1][index2] = 16
    imgListt = imgListt.reshape((-1,32256))
    dicList = []
    for i in range(32):
        imgA = cv.imread("KNNDir/"+str(i)+".jpg")
        imgList = np.zeros((224,144))
        for index1 in range(len(imgA)):
            for index2 in range(len(imgA[index1])):
                if(imgA[index1][index2][2] < 90):
                    imgList[index1][index2] = 16
        targetImgA = imgList.reshape((-1,32256))
        dicList.append(np.linalg.norm(imgListt - targetImgA))

    print(dicList)

    minV = dicList.index(min(dicList))

    print("最小距离：",min(dicList))
    print("最小位置:",minV)

    plot.imshow(cv.imread("KNNDir/"+str(minV)+".jpg"))
    plot.show()

    # Y = np.array([0,1,2,3,4,5,6,7,8,9,0,2,3,4,5,7,8,9,0,2,3,4,5,7,8,9,0,0,3,4,5,7,8,9,0,4,3,9,5,8,6,6,7,2,0,4,3,9,5,8,7,2])
    Y = np.array([0,4,3,9,5,8,6,6,0,4,3,9,5,8,6,9,0,4,3,9,5,8,7,1,0,4,3,9,5,8,7,2])

    print("预测值:",Y[minV])
    predictNum.append(Y[minV])

print(predictNum)








