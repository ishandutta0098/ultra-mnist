import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.modules.loss._WeightedLoss):
    """
    Ref: https://www.kaggle.com/kimse0ha/efficientnet-train-amp-focal-loss-fmix
    """
    def __init__(self, weight=None, gamma=2,reduction='mean'):
        super(FocalLoss, self).__init__(weight,reduction=reduction)
        self.gamma = gamma
        self.weight = weight

    def forward(self, input_, target):
        ce_loss = F.cross_entropy(input_, target,reduction=self.reduction,weight=self.weight) 
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

def get_cross_entropy_loss(outputs, labels, weights, device):
    """
    Ref: https://discuss.pytorch.org/t/passing-the-weights-to-crossentropyloss-correctly/14731
    """
    
    # Weighted Cross Entropy Loss
    if weights != None:

        # Define weighted loss
        criterion_weighted = nn.CrossEntropyLoss(weight=weights.to(device))

        # Calculate weighted loss value
        loss_weighted = criterion_weighted(outputs, labels)

        return loss_weighted

    # Non Weighted Cross Entropy Loss
    else:
        return nn.CrossEntropyLoss()(outputs, labels)

def get_focal_loss(outputs, labels, weights, device):

    loss = FocalLoss()

    if weights != None:
        return loss(outputs, labels, weights.to(device))

    else:
        return loss(outputs, labels)

def fetch_loss(cfg, outputs, labels, weights, device):
    if cfg['TRAIN']['CRITERION'] == 'CrossEntropyLoss':
        return get_cross_entropy_loss(
            outputs, 
            labels,  
            weights,
            device
        )

    if cfg['TRAIN']['CRITERION'] == 'FocalLoss':
        return get_focal_loss(
            outputs, 
            labels,
            weights,
            device
        )