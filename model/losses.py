import torch
from torch import nn
from torch.nn import functional as F


class Loss(nn.Module):
    def __init__(self, full_weight, grad_weight, mask_weight=1.0,cfg=None):
        super().__init__()
        self.full_weight = full_weight
        self.grad_weight = grad_weight
        self.mask_weight = mask_weight
        self.l1_loss = nn.L1Loss(reduction='sum')
        self.mask_loss = nn.BCELoss(reduction='mean')
    
    def get_rgb_full_loss(self,rgb_values, rgb_gt):
        rgb_loss = self.l1_loss(rgb_values, rgb_gt) / float(rgb_values.shape[1])
        return rgb_loss

    def get_smooth_loss(self, diff_norm):
        if diff_norm is None or diff_norm.shape[0]==0:
            return torch.tensor(0.0).cuda().float()
        else:
            return diff_norm.mean()

    def forward(self, out_dict, rgb_gt, mask=None, mask_gt=None):
        rgb_pred = out_dict['rgb'] 
        diff_norm = out_dict['diff_norm']
        rgb_fine =out_dict.get('rgb_fine',None) 
        if self.full_weight != 0.0:
            rgb_full_loss = self.get_rgb_full_loss(rgb_pred, rgb_gt)
        else:
            rgb_full_loss = torch.tensor(0.0).cuda().float()

        if diff_norm is not None and self.grad_weight != 0.0:
            grad_loss = self.get_smooth_loss(diff_norm)
        else:
            grad_loss = torch.tensor(0.0).cuda().float()
        
        loss = self.full_weight * rgb_full_loss + \
               self.grad_weight * grad_loss
            
        loss_term = {
            'rgb_loss': rgb_full_loss,
            'grad_loss': grad_loss,
        }

        if rgb_fine is not None:
            rgb_fine_loss = self.get_rgb_full_loss(rgb_fine, rgb_gt)
            loss += self.full_weight * rgb_fine_loss
            loss_term['rgb_fine_loss'] = rgb_fine_loss

        # add mask loss
        if mask is not None and mask_gt is not None:
            lmask = self.mask_loss(mask.clamp(0,1), mask_gt)
            loss += self.mask_weight * lmask
            loss_term['mask_loss'] = lmask

        loss_term['loss'] = loss

        if torch.isnan(loss):
            breakpoint()

        return loss_term


