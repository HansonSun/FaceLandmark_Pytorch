import pandas as pd
import cv2
import os
import sys
sys.path.append("/home/hanson/facetools/lib")
import faceutils as fu
import numpy as np


  
def generateBBox():
    rootdir = "/home/hanson/work/FaceLandmark_Pytorch/dataset/data/JD106/images"
    for root,_,filenames in os.walk(rootdir):
        for filename in filenames:
            if not filename.endswith(".jpg"):
                continue

            imgpath = os.path.join(root, filename)
            labelfile = imgpath+".txt"
            rectfile = imgpath+".rect"

            rect=[]
            with open(rectfile, "r") as f:
                line=f.read()
                line = line.strip()
                rect =  [int(float(i)) for i in  line.split(" ") ] 
                rect = fu.p2p( [int(float(i)) for i in  line.split(" ") ] )


            landmark_106p=[]
            with open(labelfile, "r") as f:
                for line in  f.readlines():
                    line = line.strip()
                    line =  [int(float(i)) for i in line.split(" ")]
                    if len(line)!=2:
                        continue
                    else:
                        landmark_106p.append(line)
            landmark_5p = [landmark_106p[i] for i in [74, 83, 54, 84, 90] ]
            landmark_6p = [landmark_106p[i] for i in [74, 83, 54, 84, 90, 16] ]
            yield imgpath, rect, landmark_5p, landmark_6p, landmark_106p

        
if __name__ == "__main__":
    test = generateBBox()

    for imgpath, rect, landmark_5p, landmark_6p, landmark_106p in test:
        image = cv2.imread(imgpath)
        fu.showimg(image, faceboxs=[rect], landmarks=[landmark_106p])
