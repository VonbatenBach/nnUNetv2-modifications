from typing import Tuple

import torch

from nnunetv2.training.dataloading.data_loader_2d import nnUNetDataLoader2D
from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.compound_losses import DC_and_topk_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper

import numpy as np


class nnUNetTrainer_probabilisticOversampling_LastLayerMask(nnUNetTrainer):
    """
    sampling of foreground happens randomly and not for the last 33% of samples in a batch
    since most trainings happen with batch size 2 and nnunet guarantees at least one fg sample, effectively this can
    be 50%
    Here we compute the actual oversampling percentage used by nnUNetTrainer in order to be as consistent as possible.
    If we switch to this oversampling then we can keep it at a constant 0.33 or whatever.
    """
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.oversample_foreground_percent = float(np.mean(
            [not sample_idx < round(self.configuration_manager.batch_size * (1 - self.oversample_foreground_percent))
             for sample_idx in range(self.configuration_manager.batch_size)]))
        self.print_to_log_file(f"self.oversample_foreground_percent {self.oversample_foreground_percent}")

    def _build_loss(self):
        assert not self.label_manager.has_regions, 'regions not supported by this trainer'
        loss = DC_and_topk_loss({'batch_dice': self.configuration_manager.batch_dice,
                                 'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
                                {'k': 10,
                                 'label_smoothing': 0.0},
                                weight_ce=1, weight_dice=1,
                                ignore_label=self.label_manager.ignore_label)

        deep_supervision_scales = self._get_deep_supervision_scales()

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
        weights[-1] = 0

        # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
        weights = weights / weights.sum()
        # now wrap the loss
        loss = DeepSupervisionWrapper(loss, weights)
        return loss

    def get_plain_dataloaders(self, initial_patch_size: Tuple[int, ...], dim: int):
        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        if dim == 2:
            dl_tr = nnUNetDataLoader2D(dataset_tr,
                                       self.batch_size,
                                       initial_patch_size,
                                       self.configuration_manager.patch_size,
                                       self.label_manager,
                                       oversample_foreground_percent=self.oversample_foreground_percent,
                                       sampling_probabilities=None, pad_sides=None, probabilistic_oversampling=True)
            dl_val = nnUNetDataLoader2D(dataset_val,
                                        self.batch_size,
                                        self.configuration_manager.patch_size,
                                        self.configuration_manager.patch_size,
                                        self.label_manager,
                                        oversample_foreground_percent=self.oversample_foreground_percent,
                                        sampling_probabilities=None, pad_sides=None, probabilistic_oversampling=True)
        else:
            dl_tr = nnUNetDataLoader3D(dataset_tr,
                                       self.batch_size,
                                       initial_patch_size,
                                       self.configuration_manager.patch_size,
                                       self.label_manager,
                                       oversample_foreground_percent=self.oversample_foreground_percent,
                                       sampling_probabilities=None, pad_sides=None, probabilistic_oversampling=True,
                                       oversample_from_last_layer_mask=True)
            dl_val = nnUNetDataLoader3D(dataset_val,
                                        self.batch_size,
                                        self.configuration_manager.patch_size,
                                        self.configuration_manager.patch_size,
                                        self.label_manager,
                                        oversample_foreground_percent=self.oversample_foreground_percent,
                                        sampling_probabilities=None, pad_sides=None, probabilistic_oversampling=True,
                                        oversample_from_last_layer_mask=True)
        return dl_tr, dl_val


