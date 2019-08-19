import torch
from models import resnet


if __name__ == '__main__':
    model = resnet.inference(10)
    saved_state_dict = torch.load("/home/hanson/work/FaceLandmark_Pytorch/bestmodel/conv_trainloss_0.000601.pth")
    model.load_state_dict(saved_state_dict)
    model.eval()

    x = torch.randn(1, 3, 96, 96)
    torch.onnx.export(model,x,"best_fc.onnx")



