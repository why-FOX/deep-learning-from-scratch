import torch
import torch.nn as nn
from torch.utils import model_zoo


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152','get_model']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_plane,out_plane,stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_plane,out_plane,kernel_size=3,stride=stride,padding=1,bias=False)

def conv1x1(in_plane,out_plane,stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_plane,out_plane,kernel_size=1,stride=stride,bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self,in_planes,planes,stride=1,downsample=None,norm_layer=nn.BatchNorm2d):
        super(BasicBlock,self).__init__()
        self.conv1 = conv3x3(in_planes,planes,stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes,planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self,x):
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

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self,in_planes,planes,stride=1,downsample=None,norm_layer=nn.BatchNorm2d):
        super(Bottleneck,self).__init__()
        self.conv1 = conv1x1(in_planes,planes,stride)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes,planes,stride)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1(planes,planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self,x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)


        if self.downsample is not None:
            identity = self.downsample(x)


        out += identity
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self,block,layers,num_classes=1000,zero_init_residual=False,norm_layer=nn.BatchNorm2d):
        super(ResNet,self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3,64,kernel_size=7,stride=1,padding=3,bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.layer1 = self._make_layer(block,64,layers[0],norm_layer=norm_layer)
        self.layer2 = self._make_layer(block,128,layers[1],stride=2,norm_layer=norm_layer)
        self.layer3 = self._make_layer(block,256,layers[2],stride=2,norm_layer=norm_layer)
        self.layer4 = self._make_layer(block,512,layers[3],stride=2,norm_layer=norm_layer)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*block.expansion,num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight,0)
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight,0)

    def _make_layer(self,block,planes,num_blocks,stride=1,norm_layer=nn.BatchNorm2d):
        downsample = None
        if stride != 1 or self.in_planes != planes *block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_planes,planes * block.expansion,stride),
                norm_layer(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.in_planes,planes,stride,downsample,norm_layer))

        self.in_planes = planes * block.expansion
        for i in range(1,num_blocks):
            layers.append(block(self.in_planes,planes))

        return nn.Sequential(*layers)

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)

        return x

def resnet18(pretrained=False,**kwargs):
    model = ResNet(BasicBlock,[2,2,2,2],**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model

def resnet34(pretrained=False,**kwargs):
    model = ResNet(BasicBlock,[3,4,6,4],**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model

def resnet50(pretrained=False,**kwargs):
    model = ResNet(BasicBlock,[3,4,6,3],**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model

def resnet101(pretrained=False,**kwargs):
    model = ResNet(BasicBlock,[3,4,23,3],**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model

def resnet152(pretrained=False,**kwargs):
    model = ResNet(BasicBlock,[3,8,36,3],**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model

def get_model(model):
    if model == 'resnet18':
        return resnet18(pretrained=True)
    elif model == 'resnet34':
        return resnet34(pretrained=True)
    elif model == 'resnet50':
        return resnet50(pretrained=True)
    elif model == 'resnet101':
        return resnet101(pretrained=True)
    elif model == 'resnet152':
        return resnet152(pretrained=True)


if __name__ == '__main__':
    #model = resnet18()
    model = get_model('resnet18')
    print(model)
    img = torch.randn(1,3,224,224)
    output = model(img)
    print(output.size())

    # spile_data.py

    import os
    from shutil import copy
    import random


    def mkfile(file):
        if not os.path.exists(file):
            os.makedirs(file)


    file = 'dataset/flower_photos'
    flower_class = [cla for cla in os.listdir(file) if ".txt" not in cla]
    mkfile('dataset/flower_photos/train')
    for cla in flower_class:
        mkfile('dataset/flower_photos/train/' + cla)

    mkfile('dataset/flower_photos/val')
    for cla in flower_class:
        mkfile('dataset/flower_photos/val/' + cla)

    split_rate = 0.1
    for cla in flower_class:
        cla_path = file + '/' + cla + '/'
        images = os.listdir(cla_path)
        num = len(images)
        eval_index = random.sample(images, k=int(num * split_rate))
        for index, image in enumerate(images):
            if image in eval_index:
                image_path = cla_path + image
                new_path = 'dataset/flower_photos/val/' + cla
                copy(image_path, new_path)
            else:
                image_path = cla_path + image
                new_path = ('dataset/flower_photos/train/') + cla
                copy(image_path, new_path)
            print("\r[{}] processing [{}/{}]".format(cla, index + 1, num), end="")  # processing bar
        print()

    print("processing done!")
