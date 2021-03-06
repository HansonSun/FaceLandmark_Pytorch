import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
import torch
import torchvision
import torch.utils.model_zoo as model_zoo
import time

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 32, layers[0])
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 256, layers[3], stride=2)

        self.maxpool_2 = nn.MaxPool2d(kernel_size=3)  


        self.landmark_fc1 = nn.Linear(256 * block.expansion, num_classes)
        self.headpose_fc1 = nn.Linear(256 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        #landmark branch
        landmark_x = self.conv1(x)
        landmark_x = self.bn1(landmark_x)
        landmark_x = self.relu(landmark_x)
        landmark_x = self.maxpool(landmark_x)
        landmark_x = self.layer1(landmark_x)
        landmark_x = self.layer2(landmark_x)
        landmark_x = self.layer3(landmark_x)
        landmark_x = self.layer4(landmark_x)
        landmark_x = self.maxpool_2(landmark_x)
        landmark_x.squeeze_(2)
        landmark_x.squeeze_(2)
        landmark_output = self.landmark_fc1(landmark_x)



        #headpose branch
        headpose_x = self.conv1(x)
        headpose_x = self.bn1(headpose_x)
        headpose_x = self.relu(headpose_x)
        headpose_x = self.maxpool(headpose_x)
        headpose_x = self.layer1(headpose_x)
        headpose_x = self.layer2(headpose_x)
        headpose_x = self.layer3(headpose_x)
        headpose_x = self.layer4(headpose_x)
        headpose_x = self.maxpool_2(headpose_x)
        headpose_x.squeeze_(2)
        headpose_x.squeeze_(2)
        headpose_output = self.headpose_fc1(headpose_x)


        return landmark_output, headpose_output


def inference( num_output=10 ):

    model = ResNet(BasicBlock, [2, 2, 2, 2], num_output )
    return model


if __name__ == '__main__':
    model = inference()
    model.eval()

    x = torch.randn(1, 3, 96, 96)
    output,_=model(x)
    output = output.view(output.size(0), -1)

    print (output.shape)
    print(x.device)
    s=time.time()
    numcnt=10
    for i in range(numcnt):
        output=model(x)
    e=time.time()
    print("inference use time %f ms",100.0*(e-s)*1.0/numcnt )
    torch.onnx.export(model,x,"test.onnx")


