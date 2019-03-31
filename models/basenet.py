import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch

class BaseNet(nn.Module):

    def __init__(self, num_classes=196):
        super(BaseNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  
        )
        self.fc1 = nn.Linear(2*2*512, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = F.dropout(x, inplace=True)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    
if __name__ == "__main__":
    model = BaseNet()
    input_tensor = torch.randn(1,3, 64, 64)
    output = model(input_tensor)
    print (output.shape)
    




