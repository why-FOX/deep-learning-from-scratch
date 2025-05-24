#encoding:utf-8
# created by xiongzihua
import torch
from torch.autograd import Variable
import torch.nn as nn

from net import vgg16, vgg16_bn
from resnet_yolo import resnet50
import torchvision.transforms as transforms
import cv2
import numpy as np

VOC_CLASSES = (
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor'
)

Color = [[0, 0, 0],
         [128, 0, 0],
         [0, 128, 0],
         [128, 128, 0],
         [0, 0, 128],
         [128, 0, 128],
         [0, 128, 128],
         [128, 128, 128],
         [64, 0, 0],
         [192, 0, 0],
         [64, 128, 0],
         [192, 128, 0],
         [64, 0, 128],
         [192, 0, 128],
         [64, 128, 128],
         [192, 128, 128],
         [0, 64, 0],
         [128, 64, 0],
         [0, 192, 0],
         [128, 192, 0],
         [0, 64, 128]]


def decoder(pred):#在这里将四个值变成左上角和右下角的坐标而不是w、h
    grid_num = 14
    boxes = []
    cls_indexs = []
    probs = []
    cell_size = 1. / grid_num
    pred = pred.data.squeeze(0)
    contain1 = pred[:, :, 4].unsqueeze(2)
    contain2 = pred[:, :, 9].unsqueeze(2)
    contain = torch.cat((contain1, contain2), 2)#将两个置信度拼接7*7*2
    mask1 = contain > 0.1
    mask2 = (contain == contain.max())
    mask = (mask1 + mask2).gt(0)

    for i in range(grid_num):
        for j in range(grid_num):
            for b in range(2):
                if mask[i, j, b] == 1:
                    box = pred[i, j, b * 5:b * 5 + 4]#7*7*30
                    contain_prob = torch.FloatTensor([pred[i, j, b * 5 + 4]])
                    xy = torch.FloatTensor([j, i]) * cell_size
                    box[:2] = box[:2] * cell_size + xy
                    box_xy = torch.FloatTensor(box.size())
                    box_xy[:2] = box[:2] - 0.5 * box[2:]
                    box_xy[2:] = box[:2] + 0.5 * box[2:]
                    max_prob, cls_index = torch.max(pred[i, j, 10:], 0)
                    if float((contain_prob * max_prob)[0]) > 0.1:
                        boxes.append(box_xy.view(1, 4))
                        cls_indexs.append(cls_index)
                        probs.append(contain_prob * max_prob)

    if len(boxes) == 0:
        return torch.empty((0, 4)), torch.empty((0,), dtype=torch.long), torch.empty((0,))
    else:#返回合格的格子的预测框xy（boxes）以及probs置信度和最大类别的概率乘积
        boxes = torch.cat(boxes, 0)
        probs = torch.cat(probs, 0)
        cls_indexs = torch.stack(cls_indexs, 0)
    keep = nms(boxes, probs)
    return boxes[keep], cls_indexs[keep], probs[keep]


def nms(bboxes, scores, threshold=0.5):#对每个最大预测概率的box与其他所有计算iou，大于0.5的删掉其他，自己保留
    x1 = bboxes[:, 0]#所有boxes行的第一个坐标x1
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    _, order = scores.sort(0, descending=True)
    keep = []
    while order.numel() > 0:
        if order.dim() == 0:  # 0-dim tensor，单个索引取出最大的边界框
            i = order.item()
        else:
            i = order[0]
        keep.append(i)

        if order.numel() == 1:#只剩一个边界框
            break

        xx1 = x1[order[1:]].clamp(min=x1[i])#与剩余所有边界框的iou计算
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])

        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        ids = (ovr <= threshold).nonzero().squeeze()#.nonzero()返回满足条件的索引，.squeeze() 将结果压缩为一维张量
        if ids.numel() == 0:
            break
        order = order[ids + 1]#转换为原始索引，前面每次都去掉最大值所以后面一直迭代+1
    return torch.LongTensor(keep)


def predict_gpu(model, image_name, root_path=''):
    result = []
    image = cv2.imread(root_path + image_name)
    h, w, _ = image.shape
    img = cv2.resize(image, (448, 448))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mean = (123, 117, 104)
    img = img - np.array(mean, dtype=np.float32)
    #减去均值 (123, 117, 104)，这是常见的图像归一化操作，用于减少光照变化的影响。

    transform = transforms.Compose([transforms.ToTensor()])
    img = transform(img)
    img = img[None, :, :, :].cuda()#添加批次维度

    with torch.no_grad():
        pred = model(img)#1*14*14*30
        print("Model output shape:", pred.shape)
    pred = pred.cpu()
    boxes, cls_indexs, probs = decoder(pred)#解码后提取出边界框坐标、类别、置信度

    for i, box in enumerate(boxes):
        x1 = int(box[0] * w)
        x2 = int(box[2] * w)
        y1 = int(box[1] * h)
        y2 = int(box[3] * h)
        cls_index = int(cls_indexs[i])
        prob = float(probs[i])
        result.append([(x1, y1), (x2, y2), VOC_CLASSES[cls_index], image_name, prob])
    return result     #[(x1, y1), (x2, y2), 类别名称, 图像名称, 置信度分数]


if __name__ == '__main__':
    model = resnet50()
    print('Loading model...')
    model.load_state_dict(torch.load('best.pth'))
    model.eval()
    model.cuda()
    image_name = 'person.jpg'
    image = cv2.imread(image_name)
    print('Predicting...')
    result = predict_gpu(model, image_name)

    for left_up, right_bottom, class_name, _, prob in result:
        color = Color[VOC_CLASSES.index(class_name)]
        cv2.rectangle(image, left_up, right_bottom, color, 2)
        label = class_name + str(round(prob, 2))
        text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        p1 = (left_up[0], left_up[1] - text_size[1])
        cv2.rectangle(image, (p1[0] - 2 // 2, p1[1] - 2 - baseline),
                      (p1[0] + text_size[0], p1[1] + text_size[1]), color, -1)
        cv2.putText(image, label, (p1[0], p1[1] + baseline), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (255, 255, 255), 1, 8)

    cv2.imwrite('result.jpg', image)
