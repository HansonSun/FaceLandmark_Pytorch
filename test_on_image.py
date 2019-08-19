import sys, os, argparse
sys.path.append("/home/hanson/facetools/lib")
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
from models import  resnet
import glob


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

    cudnn.enabled = True
    gpu = "cuda:0"
    snapshot_path = args.snapshot

    #model = basenet.BaseNet(212)
    model=resnet.inference(10).cuda()

    saved_state_dict = torch.load(args.snapshot)
    model.load_state_dict(saved_state_dict)

    model.train()

    all_images = glob.glob("/home/hanson/work/FaceLandmark_Pytorch/images/*")

    for imgpath in all_images:
        faceimg = cv2.imread(imgpath)
        faceimg = cv2.cvtColor(faceimg,cv2.COLOR_BGR2RGB)

        h,w = faceimg.shape[0:2]
        img=img_preprocess(faceimg)

        img = torch.from_numpy(img).cuda("cuda:0")
        
        scale_landmark = model(img)
        #print(scale_landmark)
        scale_landmark = scale_landmark.cpu().detach().numpy()[0]
        print (scale_landmark)
        print (scale_landmark.shape)

        for i in range(int(len(scale_landmark)/2) ):
            print (int(w*scale_landmark[2*i]), int(h*scale_landmark[2*i+1]) )
            cv2.circle(faceimg, (int(w*scale_landmark[2*i]), int(h*scale_landmark[2*i+1]) ), 3, (255,0,0), -1 )
        cv2.imshow("f",faceimg)
        cv2.waitKey(0)