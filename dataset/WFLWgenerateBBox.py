import pandas as pd
import cv2
import os
import numpy as np
import sys
sys.path.append("/home/hanson/facetools/lib")
import faceutils as fu
  
  
def generateBBox():
    rootdir = "/home/hanson/work/FaceLandmark_Pytorch/dataset/data/WFLW/WFLW_images"
    train_labelfile ="/home/hanson/work/FaceLandmark_Pytorch/dataset/data/WFLW/WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_train.txt"
    test_labelfile ="/home/hanson/work/FaceLandmark_Pytorch/dataset/data/WFLW/WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_test.txt"
    
    imgpath_list= []
    rect_list = []
    landmark_5p_list = []
    landmark_6p_list = [] 
    landmark_98p_list = []


    label_frame = pd.read_csv(train_labelfile, sep=" ", header=None)
    for infor  in label_frame.iterrows():
        imgpath = os.path.join(rootdir, infor[1][206])
        rect = fu.p2p(infor[1][196:200].values.astype(np.int).tolist())
        landmark_98p = infor[1][0:196].values.astype(np.int).reshape((98,2)).tolist()
        landmark_5p = [landmark_98p[i] for i in [96, 97, 54, 76, 82] ]
        landmark_6p = [landmark_98p[i] for i in [96, 97, 54, 76, 82, 16] ]

        imgpath_list.append(imgpath)
        rect_list.append(rect)
        landmark_5p_list.append(landmark_5p)
        landmark_6p_list.append(landmark_6p)
        landmark_98p_list.append(landmark_98p)


    label_frame = pd.read_csv(test_labelfile, sep=" ", header=None)
    for infor  in label_frame.iterrows():
        imgpath = os.path.join(rootdir, infor[1][206])
        rect = fu.p2p(infor[1][196:200].values.astype(np.int).tolist())
        landmark_98p = infor[1][0:196].values.astype(np.int).reshape((98,2)).tolist()
        landmark_5p = [landmark_98p[i] for i in [96, 97, 54, 76, 82] ]
        landmark_6p = [landmark_98p[i] for i in [96, 97, 54, 76, 82, 16] ]

        imgpath_list.append(imgpath)
        rect_list.append(rect)
        landmark_5p_list.append(landmark_5p)
        landmark_6p_list.append(landmark_6p)
        landmark_98p_list.append(landmark_98p)

    for  imgpath, rect, landmark_5p, landmark_6p, landmark_98p in zip(imgpath_list, rect_list, landmark_5p_list, landmark_6p_list, landmark_98p_list):
        yield imgpath, rect, landmark_5p, landmark_6p, landmark_98p
 


    
if __name__ == "__main__":
    test = generateBBox( )

    for imgpath, rect, landmark_5p, landmark_6p, landmark_98p in test:
        image = cv2.imread(imgpath)
        fu.showimg(image, faceboxs=[rect], landmarks=[landmark_6p])