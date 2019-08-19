# -*- coding: utf-8 -*-
from __future__ import print_function, division
import random
import os
import torch
import pandas as pd
#from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import cv2
from torchvision import transforms, utils
from PIL import Image


def show_landmarks(image, landmarks):
    for i in landmarks:
        cv2.circle(image, (i[0], i[1] ), 3, (255,0,0), -1 )
    cv2.imshow("e",image)
    cv2.waitKey(0)  
    

def show_landmarks2(image, landmarks):

    image = image.numpy()
    image = np.transpose(image,(1,2,0))
    image = 255*(image*0.5+0.5)
    image = image.astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    landmarks= landmarks.numpy()
    landmarks=landmarks.reshape((-1,2))

    h,w = image.shape[0:2]

    for i in landmarks:
        cv2.circle(image, (int(w*i[0]), int(h*i[1]) ), 3, (255,0,0), -1 )

    cv2.imshow("e",image)
    cv2.waitKey(0)  
    

class Resize(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        #print ((sample).keys())
        image, landmarks = sample['image'], sample['landmarks']
        h, w = image.shape[:2]

        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = cv2.resize(image, (new_h, new_w))

        landmarks = landmarks * [new_w / w, new_h / h]


        return {'image': img, 'landmarks': landmarks}



class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        landmarks = landmarks - [left, top]

        return {'image': image, 'landmarks': landmarks}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self,image_size):
        self.image_size = image_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        landmarks =landmarks.reshape(-1,1)
        landmarks =np.squeeze(landmarks)

        return {'image': torch.from_numpy(image).float().div(255),
                'landmarks': torch.from_numpy(landmarks).float().div(self.image_size)}

class RandomFlip(object):
    def __call__(self, sample):
        image = sample['image']
        landmarks = sample['landmarks']
        if random.random()<0.5:
            image = cv2.flip(image,1)
            landmarks[:,0] = image.shape[1]-landmarks[:,0]
        return {'image': image, 'landmarks': landmarks}


class RandomRotate(object):
    def __call__(self, sample):
        image = sample['image']
        landmarks = sample['landmarks']
        img_h,img_w = image.shape[0:2]
        center = (img_w//2, img_h//2)
        random_degree=np.random.uniform(-15.0, 15.0)

        rot_mat = cv2.getRotationMatrix2D(center, random_degree, 1)
        img_rotated_by_alpha = cv2.warpAffine(image, rot_mat, (img_w, img_h))

        rotated_landmark = np.asarray([(rot_mat[0][0]*x+rot_mat[0][1]*y+rot_mat[0][2],rot_mat[1][0]*x+rot_mat[1][1]*y+rot_mat[1][2]) for (x, y) in landmarks] )

        return {'image': img_rotated_by_alpha, 'landmarks': rotated_landmark}



class Normalize(object):
    def __init__(self,mean,std):
        self.mean = mean
        self.std = std
    def __call__(self, sample):
        image = sample['image']
        for t, m, s in zip(image, self.mean, self.std):
            t.sub_(m).div_(s)
        sample['image'] = image
        return sample

class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, label_dict_list, point_num=106, transform=None):

        self.images = []
        self.landmarks = []
        
        for label_dict in label_dict_list:


            label_frame = pd.read_csv(label_dict["label_file"], sep=" ", header=None)
            for infor  in label_frame.iterrows():
                imgpath = os.path.join(label_dict["root_dir"], infor[1][0])
                landmark = infor[1][1: (2*point_num+1) ].values.astype(np.int).reshape((-1,2))

                self.images.append(imgpath)
                self.landmarks.append(landmark)
                
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = cv2.imread( self.images[index] )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        landmarks = self.landmarks[index]
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample
        


if __name__=="__main__":

    test_transform = transforms.Compose([Resize((96,96)),
                                        #RandomCrop(96),
                                        RandomFlip(),
                                        #RandomRotate(),
                                        ToTensor(96),
                                        Normalize([ 0.5, 0.5, 0.5 ],[ 0.5, 0.5, 0.5 ])]
                                            )
    testset  = FaceLandmarksDataset([{"root_dir":"/home/hanson/work/FaceLandmark_Pytorch/dataset/WFLW106/images",
                                    "label_file":"/home/hanson/work/FaceLandmark_Pytorch/dataset/WFLW106/landmark106p_label.txt"}],

                                    #{"root_dir":"/home/hanson/work/FaceLandmark_Pytorch/dataset/JD106/images",
                                    #"label_file":"/home/hanson/work/FaceLandmark_Pytorch/dataset/JD106/landmark106p_label.txt"},

                                    #{"root_dir":"/home/hanson/work/FaceLandmark_Pytorch/dataset/menpoTrain106/images",
                                    #"label_file":"/home/hanson/work/FaceLandmark_Pytorch/dataset/menpoTrain106/landmark106p_label.txt"}],

                                    #{"root_dir":"/home/hanson/work/FaceLandmark_Pytorch/dataset/menpoTest106/images",
                                    #"label_file":"/home/hanson/work/FaceLandmark_Pytorch/dataset/menpoTest106/landmark106p_label.txt"}],
                                    point_num=106,
                                    transform=test_transform)

    print(len(testset))
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1, num_workers=1)

    for sample in test_loader:
 
        image = sample["image"][0]
        landmark = sample["landmarks"][0]
        print(landmark.shape)
        show_landmarks2(image, landmark)

