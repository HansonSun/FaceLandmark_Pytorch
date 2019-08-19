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
from models import resnet2
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

    min_test_loss = 999.0
    min_train_loss = 999.0
    args = options.get_options()
    print (args)

    if not os.path.exists(args.checkpoint):
        os.makedirs(args.checkpoint)


    #config training dataset 
    train_transform = transforms.Compose([Resize((112,112)),
                                          RandomCrop(96),
                                          #RandomFlip(),
                                          RandomRotate(),
                                          ToTensor(96),
                                          Normalize([ 0.5, 0.5, 0.5 ],[ 0.5, 0.5, 0.5 ])] )

    trainset = FaceLandmarksDataset([{"root_dir":"/home/hanson/work/FaceLandmark_Pytorch/dataset/WFLW/images",
                                    "label_file":"/home/hanson/work/FaceLandmark_Pytorch/dataset/WFLW/landmark5p_label.txt"},

                                    {"root_dir":"/home/hanson/work/FaceLandmark_Pytorch/dataset/JD106/images",
                                    "label_file":"/home/hanson/work/FaceLandmark_Pytorch/dataset/JD106/landmark5p_label.txt"},

                                    {"root_dir":"/home/hanson/work/FaceLandmark_Pytorch/dataset/menpoTrain106/images",
                                    "label_file":"/home/hanson/work/FaceLandmark_Pytorch/dataset/menpoTrain106/landmark5p_label.txt"},

                                    {"root_dir":"/home/hanson/work/FaceLandmark_Pytorch/dataset/menpoTest106/images",
                                    "label_file":"/home/hanson/work/FaceLandmark_Pytorch/dataset/menpoTest106/landmark5p_label.txt"}],
                                    point_num=5,
                                    transform=train_transform)

    trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=4, drop_last=True)
    print ("train image %d"%len(trainset))

    #config test dataset 
    test_transform = transforms.Compose([Resize((96,96)),
                                        ToTensor(96),
                                        Normalize([ 0.5, 0.5, 0.5 ],[ 0.5, 0.5, 0.5 ])] )

    testset = FaceLandmarksDataset([{"root_dir":"/home/hanson/work/FaceLandmark_Pytorch/dataset/menpoTrain106/images", 
                                    "label_file":"/home/hanson/work/FaceLandmark_Pytorch/dataset/menpoTrain106/landmark5p_label.txt"}],
                                    point_num=5,
                                    transform=test_transform)
    testloader = data.DataLoader(testset, batch_size=args.test_batch, num_workers=4)
    print ("test image %d"%len(testset))

    #define training model
    model=resnet2.inference(10).cuda()

    if len(args.snapshot)>0:
        saved_state_dict = torch.load(args.snapshot)
        model_dict = model.state_dict()
        snapshot = {k: v for k, v in saved_state_dict.items() if k in model_dict}
        model_dict.update(snapshot)
        model.load_state_dict(model_dict)




    criterion = nn.MSELoss().cuda()
    optimizer = optim.Adam(params=model.fc_conv1.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler=MultiStepLR(optimizer, [100,200,300], gamma=0.1, last_epoch=-1)

    # Train and val
    for epoch in range( args.epochs ):
        lr_scheduler.step()

        model.train()
        train_total_loss = 0.0
        for batch_idx, sample in enumerate(trainloader):
            inputs = sample["image"]
            inputs = inputs.to("cuda:0") 
            labels = sample["landmarks"]
            labels = labels.to("cuda:0")
            outputs = model(inputs)
            outputs = outputs.view(outputs.size(0), -1)
            #showlandmark(inputs, labels, outputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_total_loss += float(loss.item())

        train_avg_loss = train_total_loss/(int(batch_idx)+1)
        print('Epoch: [%d | %d] LR: %f loss: %f' % (epoch + 1, args.epochs, 
                optimizer.param_groups[0]['lr'], train_avg_loss) )


        if train_avg_loss < min_train_loss:
            min_train_loss = train_avg_loss
            if train_avg_loss<0.01:
                print ("save checkpoint sucessful")
                torch.save(model.state_dict(),"output/trainloss_%f.pth"%(train_avg_loss))

        # if epoch %10 == 0 and epoch>0: 
        #     test_loss = eval(model, testloader, criterion)
        #     if test_loss < min_test_loss:
        #         min_test_loss = test_loss
        #         torch.save(model.state_dict(),"output/trainloss_%f_testloss_%f.pth"%(train_avg_loss, test_loss))

if __name__ == '__main__':
    main()