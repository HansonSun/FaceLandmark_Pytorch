import pandas as pd
import cv2
import os
import numpy as np
import sys
sys.path.append("/home/hanson/facetools/lib")
import faceutils as fu

  
def generateBBox():
    rootdir = "/home/hanson/dataset/CelebA/Img/img_celeba.7z/img_celeba"
    label_landmark_file ="/home/hanson/dataset/CelebA/Anno/list_landmarks_celeba.txt"
    label_rect_file ="/home/hanson/dataset/CelebA/Anno/list_bbox_celeba.txt"
    
    imgpath_list=[]
    landmark_list=[]
    rect_list=[]

    label_landmark_frame = pd.read_csv(label_landmark_file, delim_whitespace = True, header=None)
    for infor  in label_landmark_frame.iterrows():
        imgpath = os.path.join(rootdir, infor[1][0])
        imgpath_list.append(imgpath)

        landmark_5p = infor[1][1:11].values.astype(np.int).reshape((-1,2)).tolist()
        landmark_list.append(landmark_5p)
        

    label_rect_frame = pd.read_csv(label_rect_file, delim_whitespace = True, header=None)
    for infor  in label_rect_frame.iterrows():
        imgpath = os.path.join(rootdir, infor[1][0])
        rect = infor[1][1:5].values.astype(np.int).tolist()
        rect_list.append(rect)
        

    for imgpath,rect, landmark in zip(imgpath_list,rect_list,landmark_list):
        yield imgpath,fu.rect(rect),landmark, None

    
if __name__ == "__main__":
    test = generateBBox( )

    for imgpath, rect, landmark_5p, landmark_106p in test:
        image = cv2.imread(imgpath)
        fu.showimg(image, faceboxs=[rect], landmarks=[landmark_5p])