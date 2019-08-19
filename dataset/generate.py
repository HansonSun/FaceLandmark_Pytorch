import os
import sys
sys.path.append("/home/hanson/pytools/lib")
import faceutils as fu
import importlib
import cv2
from  facedetect import facedetect 

def generatePic(target_name):
    
    target_image_dir = target_name+"/images"
    target_106p_label_file = target_name+"/landmark106p_label.txt"
    target_5p_label_file = target_name+"/landmark5p_label.txt"
    target_6p_label_file = target_name+"/landmark6p_label.txt"

    if not os.path.exists(target_image_dir):
        os.makedirs(target_image_dir)

    module = importlib.import_module(target_name+"generateBBox")
    face_fd = facedetect("tf-ssh")
    label5p_f = open(target_5p_label_file, "w")
    label6p_f = open(target_6p_label_file, "w")
    label106p_f = open(target_106p_label_file, "w")
    
    img_cnt = 0
    for imgpath, label_rect, label_landmark5p, label_landmark6p, label_landmark106p in module.generateBBox():


        if label_rect.width<40 or label_rect.height<40 :
            continue

        print(imgpath)
        image = cv2.imread(imgpath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        facerects, _ = face_fd.findfaces(image)
       
        target_rect = label_rect
        if len(facerects)>0:
            target_rect_tmp = max(facerects, key = lambda x:fu.IOU(label_rect,x) ) 
            if fu.IOU(target_rect_tmp, label_rect)>0.65:
                target_rect = target_rect_tmp

        roi_image = target_rect.get_roi(image)
        roi_image = cv2.cvtColor(roi_image, cv2.COLOR_RGB2BGR)
        scale_landmark5p = target_rect.projectlandmark(label_landmark5p, scale=0)
        scale_landmark6p = target_rect.projectlandmark(label_landmark6p, scale=0)
        scale_landmark106p = target_rect.projectlandmark(label_landmark106p, scale=0)

        cv2.imwrite("%s/%d.jpg"%(target_image_dir,img_cnt), roi_image)

        label5p_f.write("%d.jpg"%img_cnt)
        for point in scale_landmark5p:
            label5p_f.write(" %d %d"%(point[0], point[1]) )
        label5p_f.write("\n")


        label6p_f.write("%d.jpg"%img_cnt)
        for point in scale_landmark6p:
            label6p_f.write(" %d %d"%(point[0], point[1]) )
        label6p_f.write("\n")


        label106p_f.write("%d.jpg"%img_cnt)
        for point in scale_landmark106p:
            label106p_f.write(" %d %d"%(point[0], point[1]) )
        label106p_f.write("\n")

        img_cnt +=1
    label5p_f.close()
    label6p_f.close()
    label106p_f.close()

if __name__ == "__main__":
    generatePic(target_name = "WFLW")