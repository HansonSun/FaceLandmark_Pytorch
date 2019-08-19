import sys, os, argparse
sys.path.append("/home/hanson/pytools/lib")
sys.path.append("models")

import facedetect
import faceutils as fu
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F
from PIL import Image
import  utils
import importlib
import time
import resnet





def headpose(im, image_points):


    new_image_points = np.zeros((6,2),dtype=np.float32)

    new_image_points[0]= image_points[2]
    new_image_points[1]= image_points[5]
    new_image_points[2]= image_points[0]
    new_image_points[3]= image_points[1]
    new_image_points[4]= image_points[3]
    new_image_points[5]= image_points[4]

    model_points = np.array([
                                (0.0, 0.0, 0.0),             # Nose tip
                                (0.0, -330.0, -65.0),        # Chin
                                (-225.0, 170.0, -135.0),     # Left eye left corner
                                (225.0, 170.0, -135.0),      # Right eye right corne
                                (-150.0, -150.0, -125.0),    # Left Mouth corner
                                (150.0, -150.0, -125.0)      # Right mouth corner
                             
                            ])
     
         
    # Camera internals
    size=im.shape
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array(
                             [[focal_length, 0, center[0]],
                             [0, focal_length, center[1]],
                             [0, 0, 1]], dtype = "double"
                             )
     
    #print "Camera Matrix :\n {0}".format(camera_matrix)
     
    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    r_vec = np.array([[0.01891013], [0.08560084], [-3.14392813]])
    t_vec = np.array([[-14.97821226], [-10.62040383], [-2053.03596872]])


    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, new_image_points, camera_matrix, dist_coeffs, flags=cv2.CV_ITERATIVE)
     
    rmat = cv2.Rodrigues(rotation_vector)[0]
    P = np.hstack((rmat, translation_vector)) # projection matrix [R | t]
    degrees = -cv2.decomposeProjectionMatrix(P)[6]
    rx, ry, rz = degrees[:, 0]
    print (rx,ry,rz)

     
    (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
     
    for p in new_image_points:
        cv2.circle(im, (int(p[0]), int(p[1])), 3, (0,0,255), -1)


    p1 = ( int(new_image_points[0][0]), int(new_image_points[0][1]))
    p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

    cv2.line(im, p1, p2, (255,0,0), 2)

    cv2.imshow("Output", im)
    cv2.waitKey(1)




def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation using the Hopenet network.')
    parser.add_argument('--snapshot', dest='snapshot', help='Path of model snapshot.',default='', type=str)
    args = parser.parse_args()
    return args


def img_preprocess(img):
    processimg=cv2.resize(img,(96,96))
    processimg=processimg.astype(np.float32)
    processimg=np.transpose(processimg,(2,0,1))
    processimg=np.expand_dims(processimg,0)
    processimg=processimg/255.0
    processimg=(processimg-0.5)/0.5
    return processimg


if __name__ == '__main__':
    args = parse_args()
    fd_detector=facedetect.facedetect("tf-ssh")
    cudnn.enabled = True

    gpu = "cuda:0"
    snapshot_path = args.snapshot

    model=resnet.inference(10).cuda()

    saved_state_dict = torch.load(snapshot_path)
    model.load_state_dict(saved_state_dict)

    model.eval() 

    camera = cv2.VideoCapture(0)

    while True:
        ret,frame = camera.read()
        if ret == False:
            break

        cv2_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        rects, imgs = fd_detector.findfaces(cv2_frame)

        for rect, faceimg  in zip(rects, imgs):
            h,w = faceimg.shape[0:2]
            faceimg=img_preprocess(faceimg)

            img = torch.from_numpy(faceimg).cuda("cuda:0")
 
            scale_landmark = model(img)
  
            real_landmark = scale_landmark.reshape((-1,2))
            real_landmark[:,0] = rect.x+w*real_landmark[:,0]
            real_landmark[:,1] = rect.y+h*real_landmark[:,1]

            real_landmark = real_landmark.cpu().detach().numpy()

            #headpose(frame, real_landmark)
            cv2.rectangle(frame, (rect.x, rect.y), (rect.x2, rect.y2), (0,255,0), 1)
            for scale_x, scale_y in real_landmark:
               cv2.circle(frame, (int(scale_x), int(scale_y) ), 3, (255,0,0),-1 )
            cv2.imshow("f",frame)
            cv2.waitKey(1)