import pandas as pd
import cv2
import os
import numpy as np
import sys
sys.path.append("/home/hanson/facetools/lib")
import faceutils as fu

  
def generateBBox():
    rootdir = "/home/hanson/dataset/CelebA/Img/img_celeba.7z/img_celeba"
    labelfile ="label/img_celeba_facepp_label.csv"
    
    label_frame = pd.read_csv(labelfile, sep=" ")
    for infor  in label_frame.iterrows():
        imgpath = os.path.join(rootdir, infor[1][0])
        rect = fu.rect(infor[1][1:5].values.astype(np.int).tolist())
        landmark_106p = infor[1][8:220].values.astype(np.int).reshape((106,2)).tolist()
        landmark_5p = [landmark_106p[i] for i in [75,85,54,86,91] ]
        yield imgpath, rect, landmark_5p, landmark_106p
    
if __name__ == "__main__":
    test = generateBBox( )

    for imgpath, rect, landmark_5p, landmark_106p in test:
        image = cv2.imread(imgpath)
        fu.showimg(image, faceboxs=[rect], landmarks=[landmark_5p])