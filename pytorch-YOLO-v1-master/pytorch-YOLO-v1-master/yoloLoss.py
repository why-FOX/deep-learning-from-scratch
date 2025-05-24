#encoding:utf-8
#
#created by xiongzihua 2017.12.26
#
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class yoloLoss(nn.Module):
    def __init__(self, S, B, l_coord, l_noobj):
        super(yoloLoss, self).__init__()
        self.S = S
        self.B = B
        self.l_coord = l_coord# 位置损失的权重
        self.l_noobj = l_noobj# 无目标置信度损失的权重

    def compute_iou(self, box1, box2):
        '''Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
        Args:
          box1: (tensor) bounding boxes, sized [N,4].
          box2: (tensor) bounding boxes, sized [M,4].
        Return:
          (tensor) iou, sized [N,M].
        '''
        N = box1.size(0)
        M = box2.size(0)

        lt = torch.max(
            box1[:, :2].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:, :2].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        rb = torch.min(
            box1[:, 2:].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:, 2:].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        wh = rb - lt  # [N,M,2]
        wh[wh < 0] = 0  # clip at 0
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N,]
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M,]
        area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
        area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

        iou = inter / (area1 + area2 - inter)
        return iou # [N,M]

    def forward(self, pred_tensor, target_tensor):
        '''
        pred_tensor: (tensor) size(batchsize,S,S,Bx5+20=30) [x,y,w,h,c]
        target_tensor: (tensor) size(batchsize,S,S,30)
        '''
        N = pred_tensor.size(0)#7*7*30
        device = pred_tensor.device

        coo_mask = target_tensor[:, :, :, 4] > 0# (batchsize, S, S)
        noo_mask = target_tensor[:, :, :, 4] == 0
        coo_mask = coo_mask.unsqueeze(-1).expand_as(target_tensor)# (batchsize, S, S，1)-》(batchsize, S, S，30)
        noo_mask = noo_mask.unsqueeze(-1).expand_as(target_tensor)

        coo_pred = pred_tensor[coo_mask].view(-1, 30)#【X,30】只关注置信度>0的格子的30个值
        box_pred = coo_pred[:, :10].contiguous().view(-1, 5)  # box[x,y,w,h,c]【2X,5】
        class_pred = coo_pred[:, 10:]  # class preds    [X,20]

        coo_target = target_tensor[coo_mask].view(-1, 30)
        box_target = coo_target[:, :10].contiguous().view(-1, 5)
        class_target = coo_target[:, 10:]

        # compute no object loss
        noo_pred = pred_tensor[noo_mask].view(-1, 30)
        noo_target = target_tensor[noo_mask].view(-1, 30)
        noo_pred_mask = torch.zeros(noo_pred.size(), dtype=torch.bool, device=device)
        noo_pred_mask[:, 4] = True
        noo_pred_mask[:, 9] = True#只有这两个置信度值需要计算其它均为false不参与后续计算
        noo_pred_c = noo_pred[noo_pred_mask]
        noo_target_c = noo_target[noo_pred_mask]
        nooobj_loss = F.mse_loss(noo_pred_c, noo_target_c, reduction='sum')

        # compute contain object loss
        coo_response_mask = torch.zeros(box_target.size(), dtype=torch.bool, device=device)#【2X,30】
        coo_not_response_mask = torch.zeros(box_target.size(), dtype=torch.bool, device=device)
        box_target_iou = torch.zeros(box_target.size(), device=device)
        for i in range(0, box_target.size(0), 2):  # choose the best iou box
            box1 = box_pred[i:i + 2]#【2,5】
            box1_xyxy = torch.zeros_like(box1)
            box1_xyxy[:, :2] = box1[:, :2] / 14. - 0.5 * box1[:, 2:4]
            box1_xyxy[:, 2:4] = box1[:, :2] / 14. + 0.5 * box1[:, 2:4]
            box2 = box_target[i].view(-1, 5)
            box2_xyxy = torch.zeros_like(box2)
            box2_xyxy[:, :2] = box2[:, :2] / 14. - 0.5 * box2[:, 2:4]
            box2_xyxy[:, 2:4] = box2[:, :2] / 14. + 0.5 * box2[:, 2:4]
            iou = self.compute_iou(box1_xyxy[:, :4], box2_xyxy[:, :4])  # [2,1]
            max_iou, max_index = iou.max(0)
            max_index = max_index.to(device)

            coo_response_mask[i + max_index] = True
            coo_not_response_mask[i + 1 - max_index] = True

            # confidence score = iou between pred box and gt box
            box_target_iou[i + max_index, 4] = max_iou#【2X，5】

        # response loss
        box_pred_response = box_pred[coo_response_mask].view(-1, 5)
        box_target_response_iou = box_target_iou[coo_response_mask].view(-1, 5)
        box_target_response = box_target[coo_response_mask].view(-1, 5)
        contain_loss = F.mse_loss(box_pred_response[:, 4], box_target_response_iou[:, 4], reduction='sum')
        loc_loss = F.mse_loss(box_pred_response[:, :2], box_target_response[:, :2], reduction='sum') + \
                   F.mse_loss(torch.sqrt(box_pred_response[:, 2:4]), torch.sqrt(box_target_response[:, 2:4]), reduction='sum')

        # not response loss
        box_pred_not_response = box_pred[coo_not_response_mask].view(-1, 5)
        box_target_not_response = box_target[coo_not_response_mask].view(-1, 5)
        box_target_not_response[:, 4] = 0
        not_contain_loss = F.mse_loss(box_pred_not_response[:, 4], box_target_not_response[:, 4], reduction='sum')

        # class loss
        class_loss = F.mse_loss(class_pred, class_target, reduction='sum')

        loss = (self.l_coord * loc_loss + 2 * contain_loss + (not_contain_loss + self.l_noobj * nooobj_loss) + class_loss) / N
        return loss
