from __future__ import division
import cv2
import time
import math
import numpy as np
import pandas
import glob
import random
import csv
import os
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import pickle
cap = cv2.VideoCapture(0)

check_pass = 0
font = cv2.FONT_HERSHEY_SIMPLEX

def detech(frame):
    protoFile = "hand/pose_deploy.prototxt"
    weightsFile = "hand/pose_iter_102000.caffemodel"
    nPoints = 21
    POSE_PAIRS = [ [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20] ]
    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

    framelen = cv2.imread('photohand.jpg')
    frameCopy = np.copy(frame)
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    aspect_ratio = frameWidth/frameHeight

    threshold = 0.1

    t = time.time()
    # input image dimensions for the network
    inHeight = 350
    inWidth = int(((aspect_ratio*inHeight)*8)//8)
    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)

    net.setInput(inpBlob)

    output = net.forward()
    # print("time taken by network : {:.3f}".format(time.time() - t))

    # Empty list to store the detected keypoints
    points = []
    sumx =0
    sumy =0
    countPoint =0

    for i in range(nPoints):
        # confidence map of corresponding body's part.
        probMap = output[0, i, :, :]
        probMap = cv2.resize(probMap, (frameWidth, frameHeight))

        # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

        if prob > threshold :
            cv2.circle(frameCopy, (int(point[0]), int(point[1])), 4, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frameCopy, "{}".format(i), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, lineType=cv2.LINE_AA)
            sumx=sumx+int(point[0])
            sumy=sumy+int(point[1])
            countPoint+=1

            # Add the point to the list if the probability is greater than the threshold
            points.append((int(point[0]), int(point[1])))
        else :
            points.append(None)
    # Calculate Centroid and Draw
    if countPoint!=0:
        cx = sumx//countPoint
        cy = sumy//countPoint

    row = []
    for p in range(len(points)):
        if points[p] == None:
            points[p]=(0,0)
        else:
            points[p]=(abs(points[p][0]-cx),abs(points[p][1]-cy))
        row.append(points[p][0])
        row.append(points[p][1])
            #print(points[p])
        # row.append(img)
        # print(row)
        # rowlist.append(row)
    print(row)
    return(row)

# def randome():
#     list1 = ['1','2','3','4','5','6','7','8','9']
#     rd = np.random.sample(list1,5)
#     print (rd)


def getResult(pointxy):
    #loadmodel 
    # filename = 'finalized_model2.sav'
    filename = 'loaded_dataAllNoRotate.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    result = loaded_model.predict([pointxy])  
    # print(result[0])
    return result[0]

# def sumlist(sumzero):
# res = sum(i for i in row if i % 2 != 0)
# return (res)

result='.'
kn=0
set = 0
cv_frame = []
imgg = []
state_random = True
# numberList2 = ['d1','d2','d4','d5','d6','d7','d8','d9']
numberList = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
# numberList = ['d1','d2','d3','d4','d5','d6','d7','d8','d9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
score = 0
count = 5
rd = []
while (True):
    framelen = cv2.imread('photohand.png')
    if(state_random and count != 0):
        rd = random.choice(numberList)
        state_random = False

    imgg = 'handcrop/testnum/text'+rd+'.jpg'
    frame2 = cv2.imread(imgg)
    cv_frame.append(frame2)
    
    ret,frame = cap.read()
    roi=frame[100:350, 100:250]
    kn = kn+1

    #Draw a square
    cv2.rectangle(frame,(100,100),(250,350),(0,255,0),1)
    cv2.putText(frame,'Put hand in the box',(90,50), font, 1.5, (0,0,255), 2, cv2.LINE_AA)
    cv2.putText(frame,result.replace('d',' '),(450,140), font, 1.5, (0,255,255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Score: "+str(score) ,(420,450), font, 1, (0,255,255), 2, cv2.LINE_AA)

    # cv2.imshow('hand',framelen)
    cv2.imshow('camera',frame)
    cv2.imshow('camera2',roi)
    cv2.imshow('frame2',frame2)

    k = cv2.waitKey(5) & 0xff
    if k == 27 or count == 0:
        break
    else:
        if kn == 40:
            points = detech(roi)
            Sum = sum(points)
            # txt = points.replace('d',' .')
            if Sum == 0 :
                result = 'none'
            else:
                result = getResult(points)
            
            print(result.replace('d',' '))
            kn=0
    
    if(result == rd):
        score = score+1
        count = count-1
        state_random = True
        print("PASS!!")

cap.release()
cv2.destroyAllWindows()

