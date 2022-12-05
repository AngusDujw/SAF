import torch
import torchvision.models as models

model = models.__dict__['resnet50'](num_classes=1000)
model = torch.nn.DataParallel(model).cuda()

x = torch.randn(1024,3,224,224).cuda()
print('================program has been completed!=================')
while True:
    model(x)
