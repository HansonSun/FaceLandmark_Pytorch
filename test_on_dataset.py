import os
import shutil
import time
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from models import resnet
from dataset.FaceLandmarksDataset import *
from utils import Bar, Logger, AverageMeter,normalizedME, mkdir_p, savefig
import options 
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np

def showlandmark(testimg_output, real_landmrk_output, predict_landmark_output):
    testimg = testimg_output[0]
    real_landmrk = real_landmrk_output[0]
    predict_landmark = predict_landmark_output[0]

    predict_landmark_np = predict_landmark.cpu().detach().numpy()
    real_landmrk = real_landmrk.cpu().detach().numpy()

    predict_landmark_np.reshape((-1,2))
    real_landmrk = real_landmrk.reshape((-1,2))

    testimg = testimg.cpu().detach().numpy()
    testimg = np.transpose(testimg,(1,2,0))
    testimg = 255*(testimg*0.5+0.5)
    testimg = testimg.astype(np.uint8)

    h,w = testimg.shape[0:2]

    testimg = cv2.cvtColor(testimg, cv2.COLOR_RGB2BGR)
    for i in real_landmrk:
        cv2.circle(testimg, (int(w*i[0]), int(h*i[1]) ), 3, (255,0,0), -1 )

    cv2.imshow("d",testimg)
    cv2.waitKey(0)


def eval(model, testloader, criterion):
    print("start testing...")
    model.eval()
    all_loss = 0.0

    for batch_idx, sample in enumerate(testloader):
        inputs = sample["image"]
        inputs = inputs.to("cuda:0") 
        labels = sample["landmarks"]
        labels = labels.to("cuda:0")
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        all_loss+= float( loss.item() )
    test_loss = (all_loss/(int(batch_idx)+1))  
    print("testing loss %f"%test_loss  )
    return test_loss


def main():


    args = options.get_options()
    print (args)

    #config training dataset 
    test_transform = transforms.Compose([Resize((96,96)),
                                          ToTensor(96),
                                          Normalize([ 0.5, 0.5, 0.5 ],[ 0.5, 0.5, 0.5 ])] )

    testset = FaceLandmarksDataset([{"root_dir":"/home/hanson/work/FaceLandmark_Pytorch/dataset/WFLW/images",
                                    "label_file":"/home/hanson/work/FaceLandmark_Pytorch/dataset/WFLW/landmark5p_label.txt"},

                                    {"root_dir":"/home/hanson/work/FaceLandmark_Pytorch/dataset/JD106/images",
                                    "label_file":"/home/hanson/work/FaceLandmark_Pytorch/dataset/JD106/landmark5p_label.txt"},

                                    {"root_dir":"/home/hanson/work/FaceLandmark_Pytorch/dataset/menpoTrain106/images",
                                    "label_file":"/home/hanson/work/FaceLandmark_Pytorch/dataset/menpoTrain106/landmark5p_label.txt"},

                                    {"root_dir":"/home/hanson/work/FaceLandmark_Pytorch/dataset/menpoTest106/images",
                                    "label_file":"/home/hanson/work/FaceLandmark_Pytorch/dataset/menpoTest106/landmark5p_label.txt"}],
                                    point_num=5,
                                    transform=test_transform)

    testloader = data.DataLoader(testset, batch_size=200, num_workers=4)
    print ("test image %d"%len(testset))

    #define training model
    model=resnet.inference(10).cuda()

    if len(args.snapshot)>0:
        saved_state_dict = torch.load(args.snapshot)
        model.load_state_dict(saved_state_dict)

    criterion = nn.MSELoss().cuda()

    test_loss = eval(model, testloader, criterion)

    print (test_loss)

if __name__ == '__main__':
    main()