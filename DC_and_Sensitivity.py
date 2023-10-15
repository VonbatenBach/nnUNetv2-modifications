import torch
from nnunetv2.training.loss.sensitivity import SensitivityLoss
from nnunetv2.training.loss.dice import SoftDiceLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1
from torch import nn

class DC_and_TPR_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_TPR=1, weight_dice=1, ignore_label=None,
                 dice_class=SoftDiceLoss):
        """ nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss, TopKLoss
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_and_TPR_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_TPR = weight_TPR
        self.ignore_label = ignore_label

        self.TPR = SensitivityLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = (target != self.ignore_label).bool()
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.clone(target)
            target_dice[target == self.ignore_label] = 0
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        sensitivity_loss = self.TPR(net_output, target_dice) \
            if self.weight_TPR != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_TPR * sensitivity_loss + self.weight_dice * dc_loss
        return result

