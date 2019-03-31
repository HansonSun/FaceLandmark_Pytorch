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
  
  
def generateBBox(rootdir):
    subdirs = ["AFW", "HELEN", "IBUG", "LFPW"]
    for subdir in subdirs:
        labelpath = os.path.join(rootdir, subdir,"landmark.txt")
        label_frame = pd.read_csv(labelpath, sep=" ", header=None)
        for infor  in label_frame.iterrows():
            imgpath = os.path.join(rootdir, subdir, "picture", infor[1][0])
            landmark = infor[1][1:213].values.astype(np.int).reshape((106, 2))
            min_x = min(landmark[:, 0])
            min_y = min(landmark[:, 1])
            max_x = max(landmark[:, 0])
            max_y = max(landmark[:, 1])
            rect = (min_x, min_y, max_x, max_y )
            yield imgpath, rect, landmark
        
if __name__ == "__main__":
    test = generateBBox("D:\\FaceLandmark_Pytorch\\dataset\\data\\JD-106\\Training_data")

    for imgpath, rect, landmark in test:
        print (imgpath)
        image = cv2.imread(imgpath)
        show_landmarks(image, rect, landmark)
