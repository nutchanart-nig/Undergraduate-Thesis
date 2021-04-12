from __future__ import division
import cv2
import time
import math
import numpy as np
import csv
import os


# get data images
path = "E:/Com-Sci/hand/project/datatest"
labels =[]
print(path)
for root, directories, files in os.walk(path, topdown=True):
    for dir in directories:
        labels.append(dir)

threshold = 0.1

protoFile = "hand/pose_deploy.prototxt"
weightsFile = "hand/pose_iter_102000.caffemodel"
nPoints = 21
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

head=[]
for i in range(nPoints):
    head.append('x'+str(i))
    head.append('y'+str(i))
head.append('classpoint')

rowlist=[]


for label in labels:
    filePath = path+"/"+label
    imgFiles = os.listdir(filePath)
    for img in imgFiles:
        t = time.time()
        filename = filePath+"/"+img
        print(filename)
        frame = cv2.imread(filename)
        frameCopy = np.copy(frame)
        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]
        aspect_ratio = frameWidth/frameHeight

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
                cv2.circle(frameCopy, (int(point[0]), int(point[1])), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.putText(frameCopy, "{}".format(i), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)
                sumx=sumx+int(point[0])
                sumy=sumy+int(point[1])
                countPoint+=1

                # Add the point to the list if the probability is greater than the threshold
                points.append((int(point[0]), int(point[1])))
            else :
                points.append(None)
            #print("Point ",i,prob,points[i])

        # Calculate Centroid and Draw
        
        if countPoint!=0:
            cx = sumx//countPoint
            cy = sumy//countPoint
            
        # re-location
        row = []
        for p in range(len(points)):
            if points[p] == None:
                points[p]=(0,0)
            else:
                points[p]=(abs(points[p][0]-cx),abs(points[p][1]-cy))
            row.append(points[p][0])
            row.append(points[p][1])
            #print(points[p])
        row.append(label)
        row.append(img)
        print(row)
        rowlist.append(row)

        print(row)

with open('z.csv', 'w', newline='') as file:
     writer = csv.writer(file)
     writer.writerows([head])
     writer.writerows(rowlist)
            
# cv2.imshow('Output-Keypoints', frameCopy)
# cv2.imshow('Output-Skeleton', frame)


# cv2.imwrite('Output-Keypoints.jpg', frameCopy)
# cv2.imwrite('Output-Skeleton.jpg', frame)

# print("Total time taken : {:.3f}".format(time.time() - t))

cv2.waitKey(0)
