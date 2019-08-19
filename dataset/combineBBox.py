import os
import sys
sys.path.append("/home/hanson/pytools/lib")
import faceutils as fu
import importlib
import cv2
from  facedetect import facedetect 


module = importlib.import_module("WFLW106"+"generateBBox")
face_fd = facedetect().get_instance("tf-ssh")

for imgpath, label_rect, label_landmark5p, label_landmark106p in module.generateBBox():
    image = cv2.imread(imgpath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    facerects, _ = face_fd.findfaces(image)
   
    target_rect_tmp = max(facerects, key = lambda x:fu.IOU(label_rect,x) ) 

    if fu.IOU(target_rect_tmp, label_rect)>0.65:
        target_rect = target_rect_tmp
    else:
        target_rect = label_rect


    roi_image = target_rect.get_roi(image)
    scale_landmark = target_rect.projectlandmark(label_landmark5p, scale=0)

    #fu.showimg(roi_image, faceboxs = [target_rect], landmarks = [label_landmark])
    #fu.showimg(roi_image, landmarks = [scale_landmark] ,landmarkscale=0)


    for i in scale_landmark:
        cv2.circle(roi_image, (i[0], i[1]), 3, (255,0,0), -1 )
    cv2.imshow("e",roi_image)
    cv2.waitKey(0)  