from torchvision.models.resnet import Bottleneck, resnet50
import torch
resnet = resnet50()
a = torch.randn(5, 3, 24, 24)

conv1 = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
# print(conv1)
# print(conv1(a).size())


print(resnet)
