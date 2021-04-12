from __future__ import division

import cv2
import time
import math
import glob
import numpy as np
import csv
cap = cv2.VideoCapture(0)

protoFile = "hand/pose_deploy.prototxt"
weightsFile = "hand/pose_iter_102000.caffemodel"
nPoints = 22
POSE_PAIRS = [ [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20] ]
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

# Empty list to store the detected keypoints
points = []
# poiny xy        
pointxy = []
gg = []

#folder of file
folder = "Z"

# open on file
cv_frame = []
for img in glob.glob('datatest/'+folder+'/*.jpg'):
    frame = cv2.imread(img)
    cv_frame.append(frame)
    # print (str(img))

    #hasFrame, frame = cap.read()
    #frame = cv2.imread("data/5/5 (6).jpg")
    frameCopy = np.copy(frame)
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    aspect_ratio = frameWidth/frameHeight

    threshold = 0.1

    t = time.time()
    # input image dimensions for the network
    inHeight = 368
    inWidth = int(((aspect_ratio*inHeight)*8)//8)
    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)

    net.setInput(inpBlob)

    output = net.forward()
    #print("time taken by network : {:.3f}".format(time.time() - t))

    for i in range(nPoints):
        # confidence map of corresponding body's part.
        probMap = output[0, i, :, :]
        probMap = cv2.resize(probMap, (frameWidth, frameHeight))

        # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

        if prob > threshold :
            cv2.circle(frameCopy, (int(point[0]), int(point[1])), 4, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frameCopy, "{}".format(i), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, lineType=cv2.LINE_AA)

            # Add the point to the list if the probability is greater than the threshold
            gg.append((int(point[0]), int(point[1])))
        else :
            gg.append(None)
    points.append(gg)
    gg = []

folderpoint= open('data/'+folder+".txt","w+")
pointxy = []
for p in points:
    cc = []
    for keypoint in p:
        if keypoint != None:
            cc.append(keypoint[0])
            cc.append(keypoint[1])
    cc.append(folder)
    folderpoint.write(str(cc)+" = "+folder+"\n")
    print(cc, " = "+folder)
    pointxy.append(cc)
folderpoint.close()

with open('data2/'+folder+".csv", 'w', newline='') as csvfile:
    # filewriter = csv.writer(csvfile, delimiter=',', 
    #                         quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer = csv.writer(csvfile)
    writer.writerow(["x0", "y0","x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4", "x5", "y5", "x6", "y6", "x7", "y7", "x8", "y8", "x9", "y9", "x10", "y10", 
                    "x11", "y11", "x12", "y12", "x13", "y13", "x14", "y14", "x15", "y15", "x16", "y16", "x17", "y17", "x18", "y18", "x19", "y19", "x20", "y20", "classpoint"])
    for wrirow in pointxy:
        writer.writerow(wrirow)

objects = serialization.read_all("naivebayes.model")
classifier = Classifier(jobject=objects[0])
print(classifier)
        

# for p in points:
#     if p != None:
#         pointxy.append(p)
# print ('A',pointxy)


# ppoint = []
# for x in pointxy:
#     ppoint.append(str(x[0]) + "," + str(x[1]))
# print(ppoint)

