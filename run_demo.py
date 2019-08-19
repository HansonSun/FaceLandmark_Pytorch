from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
sys.path.append("/home/hanson/facetools/lib")
from facedetect import facedetect
import faceutils as fu
import cv2

def demo(args):
    face_fd=facedetect().get_instance("tf-ssh")

    cam=cv2.VideoCapture(0)
    cam.set(3,1280) 
    cam.set(4,720)


    while(1):
        _,img=cam.read()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rects, imgs=face_fd.findfaces(img)
        if  len(rects)==0:
            print ("find no face")
        else:
           for  index , (rect, faceimg) in enumerate( zip(rects,imgs) ) :
               cv2.imshow("1",faceimg)
               cv2.waitKey(1)


if __name__=="__main__":
    demo()
