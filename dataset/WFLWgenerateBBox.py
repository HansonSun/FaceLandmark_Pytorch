import pandas as pd
import cv2
import os
import numpy as np

def show_landmarks(image, rect, landmarks):
    """Show image with landmarks"""
    cv2.rectangle(image, (rect[0], rect[1]), (rect[2], rect[3]), (0,255,0), 3)
    for i in landmarks:
        cv2.circle(image, (i[0], i[1]), 3, (255,0,0), -1 )
    cv2.imshow("e",image)
    cv2.waitKey(0)
  
  
def generateBBox(rootdir, labelfile):
    label_frame = pd.read_csv(labelfile, sep=" ", header=None)
    for infor  in label_frame.iterrows():
        imgpath = os.path.join(rootdir, infor[1][206])
        rect = infor[1][196:200].values.astype(np.int)
        landmark = infor[1][0:196].values.astype(np.int).reshape((98,2))
        
        yield imgpath, rect, landmark
    
if __name__ == "__main__":
    test = generateBBox("D:\\FaceLandmark_Pytorch\\dataset\\data\\WFLW\\WFLW_images",
    "D:\\FaceLandmark_Pytorch\\dataset\\data\\WFLW\\WFLW_annotations\\list_98pt_rect_attr_train_test\\list_98pt_rect_attr_train.txt")

    for imgpath, rect, landmark in test:
        image = cv2.imread(imgpath)
        show_landmarks(image, rect, landmark)
