import pandas as pd
import cv2
import os
import numpy as np
import sys
sys.path.append("/home/hanson/facetools/lib")
import faceutils as fu
  
  
def generateBBox():
    rootdir = "/home/hanson/work/FaceLandmark_Pytorch/dataset/data/WFLW/WFLW_images"
    labelfile ="/home/hanson/work/FaceLandmark_Pytorch/dataset/data/WFLW/WFLW_images_facepp_label.csv"
    
    label_frame = pd.read_csv(labelfile, sep=" ")
    for infor  in label_frame.iterrows():
        imgpath = os.path.join(rootdir, infor[1][0])
        rect = fu.rect(infor[1][1:5].values.astype(np.int).tolist())
        landmark = infor[1][8:220].values.astype(np.int).reshape((106,2)).tolist()
        new_landmark=[]
        #print (landmark[75])
        new_landmark.append(landmark[75])
        new_landmark.append(landmark[85]) 
        new_landmark.append(landmark[54]) 
        new_landmark.append(landmark[86]) 
        new_landmark.append(landmark[91]) 

        yield imgpath, rect, new_landmark
    
if __name__ == "__main__":
    test = generateBBox( )

    for imgpath, rect, landmark in test:
        image = cv2.imread(imgpath)
        fu.showimg(image, faceboxs=[rect], landmarks=[landmark])