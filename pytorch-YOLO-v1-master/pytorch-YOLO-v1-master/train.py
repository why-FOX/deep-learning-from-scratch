import os
import visdom
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models
from torch.autograd import Variable
from tqdm import tqdm

from net import vgg16, vgg16_bn
from resnet_yolo import resnet50, resnet18
from yoloLoss import yoloLoss
from dataset import yoloDataset

from visualize import Visualizer
import numpy as np

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    use_gpu = torch.cuda.is_available()

    file_root = 'D:\github\deep-learning-from-scratch\VOC2012\JPEGImages'
    learning_rate = 0.001
    num_epochs = 1
    batch_size = 24
    use_resnet = True
    if use_resnet:
        net = resnet50()
    else:
        net = vgg16_bn()

    print(net)
    print('load pre-trined model')
    if use_resnet:
        resnet = models.resnet50(pretrained=True)#加载预训练权重
        new_state_dict = resnet.state_dict()
        dd = net.state_dict()
        for k in new_state_dict.keys():
            if k in dd.keys() and not k.startswith('fc'):
                dd[k] = new_state_dict[k]
        net.load_state_dict(dd)#把权重复制过来我建的net里
    else:
        vgg = models.vgg16_bn(pretrained=True)
        new_state_dict = vgg.state_dict()
        dd = net.state_dict()
        for k in new_state_dict.keys():
            if k in dd.keys() and k.startswith('features'):
                dd[k] = new_state_dict[k]
        net.load_state_dict(dd)

    if False:
        net.load_state_dict(torch.load('best.pth'))
    print('cuda', torch.cuda.current_device(), torch.cuda.device_count())

    criterion = yoloLoss(7, 2, 5, 0.5)#损失函数初始化了
    if use_gpu:
        net.cuda()

    net.train()#训练模式
    params = []
    params_dict = dict(net.named_parameters())#返回网络中所有可训练参数的字典
    for key, value in params_dict.items():#把特征层和分类头分开学习率设置
        if key.startswith('features'):
            params += [{'params': [value], 'lr': learning_rate * 1}]
        else:
            params += [{'params': [value], 'lr': learning_rate}]
    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=5e-4)

    train_dataset = yoloDataset(root=file_root, list_file=['voc2012_sm.txt'], train=True, transform=[transforms.ToTensor()])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    test_dataset = yoloDataset(root=file_root, list_file='voc2012_valsm.txt', train=False, transform=[transforms.ToTensor()])
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print('the dataset has %d images' % (len(train_dataset)))
    print('the batch_size is %d' % (batch_size))
    logfile = open('log.txt', 'w')

    num_iter = 0
    vis = Visualizer(env='xiong')
    vis_test = visdom.Visdom()
    print(vis_test.check_connection())

    best_test_loss = np.inf

    for epoch in range(num_epochs):
        net.train()
        if epoch == 1:
            learning_rate = 0.0005
        if epoch == 30:
            learning_rate = 0.0001
        if epoch == 40:
            learning_rate = 0.00001

        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate

        print('\n\nStarting epoch %d / %d' % (epoch + 1, num_epochs))
        print('Learning Rate for this epoch: {}'.format(learning_rate))

        total_loss = 0.

        # 训练进度条
        for i, (images, target) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} Training")):
            images = Variable(images)
            target = Variable(target)
            if use_gpu:
                images, target = images.cuda(), target.cuda()

            pred = net(images)
            loss = criterion(pred, target)#都是7*7*30
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 5 == 0:
                print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, average_loss: %.4f'
                      % (epoch + 1, num_epochs, i + 1, len(train_loader), loss.item(), total_loss / (i + 1)))
                num_iter += 1
                vis.plot_train_val(loss_train=total_loss / (i + 1))
                print(f"Plot train loss at iter {num_iter}: {total_loss / (i + 1)}")

        # 验证进度条
        validation_loss = 0.0
        net.eval()
        with torch.no_grad():#减小计算量
            for i, (images, target) in enumerate(tqdm(test_loader, desc=f"Epoch {epoch+1} Validation")):
                images = Variable(images)
                target = Variable(target)
                if use_gpu:
                    images, target = images.cuda(), target.cuda()

                pred = net(images)
                loss = criterion(pred, target)
                validation_loss += loss.item()

        validation_loss /= len(test_loader)
        vis.plot_train_val(loss_val=validation_loss)

        if best_test_loss > validation_loss:
            best_test_loss = validation_loss
            print('get best test loss %.5f' % best_test_loss)
            torch.save(net.state_dict(), 'best.pth')
        logfile.writelines(str(epoch) + '\t' + str(validation_loss) + '\n')
        logfile.flush()
        torch.save(net.state_dict(), 'yolo.pth')

if __name__ == '__main__':
    main()
