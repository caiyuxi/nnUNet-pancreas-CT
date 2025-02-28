import pydoc
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from torch import nn, autocast
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss
import numpy as np
import torch
from nnunetv2.training.loss.compound_losses import DC_and_BCE_loss, DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.model import ResidualEncoderUNetWithClassification
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p

from nnunetv2.utilities.helpers import empty_cache, dummy_context




class Combined_Loss(nn.Module):
    def __init__(self, batch_dice, ignore_label, is_ddp=False):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(Combined_Loss, self).__init__()

        self.seg_loss = DC_and_CE_loss({'batch_dice': batch_dice,
                               'smooth': 1e-5, 'do_bg': True, 'ddp': is_ddp}, {}, weight_ce=0.3, weight_dice=0.7,
                              ignore_label=ignore_label, dice_class=MemoryEfficientSoftDiceLoss)
        self.classification_loss = RobustCrossEntropyLoss()

    def forward(self, net_output, target, single_head_loss=None):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        # net_output['classification'] = list(net_output['classification'])
        if not isinstance(net_output['segmentation'], list):
            net_output['segmentation'] = [net_output['segmentation']]
        if not isinstance(target['segmentation'], list):
            target['segmentation'] = [target['segmentation']]
        if not single_head_loss:
            seg_loss = self.seg_loss(net_output['segmentation'], target['segmentation'])
            class_loss = self.classification_loss(net_output['classification'], target['classification'])
            return seg_loss + 1.5*class_loss, seg_loss, class_loss
        if single_head_loss == 'segmentation':
            return self.seg_loss(net_output['segmentation'], target['segmentation'])
        if single_head_loss == 'classification':
            return self.classification_loss(net_output['classification'], target['classification'])
        assert "Should never reach here"

class nnUNetTrainerWithClassification(nnUNetTrainer):
    def __init__(
            self,
            plans: dict,
            configuration: str,
            fold: int,
            dataset_json: dict,
            unpack_dataset: bool = True,
            device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 1e-4
        self.weight_decay = 1e-4
        self.num_epochs = 100
        self.single_head_loss = "segmentation"


    @staticmethod
    def build_network_architecture(architecture_class_name,
                                   arch_init_kwargs,
                                   arch_init_kwargs_req_import,
                                   num_input_channels,
                                   num_output_channels,
                                   enable_deep_supervision):

        architecture_kwargs = dict(arch_init_kwargs)
        for ri in arch_init_kwargs_req_import:
            if architecture_kwargs[ri] is not None:
                architecture_kwargs[ri] = pydoc.locate(architecture_kwargs[ri])
        if enable_deep_supervision:
            architecture_kwargs['deep_supervision'] = enable_deep_supervision
        model = ResidualEncoderUNetWithClassification(
            input_channels=num_input_channels,
            num_classes=num_output_channels,
            **architecture_kwargs)
        return model

    def _build_loss(self):
        loss = Combined_Loss(self.configuration_manager.batch_dice, self.label_manager.ignore_label, self.is_ddp)
        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                # very strange and stupid interaction. DDP crashes and complains about unused parameters due to
                # weights[-1] = 0. Interestingly this crash doesn't happen with torch.compile enabled. Strange stuff.
                # Anywho, the simple fix is to set a very low weight to this.
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss.seg_loss = DeepSupervisionWrapper(loss.seg_loss, weights)


        return loss

    def actual_validation_step(self, batch: dict) -> dict:
        data = batch['data']
        seg = batch['target']
        expected_classifications = torch.Tensor([int(x.split('_')[1]) for x in batch['keys']]).to(self.device)
        target = {
            'segmentation': [x.to(self.device) for x in seg],  # Segmentation labels. Move to device
            'classification': expected_classifications
        }
        data = data.to(self.device, non_blocking=True)

        if isinstance(target['segmentation'], list):
            target['segmentation'] = target['segmentation'][0]

        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            outputs = self.network(data)
            del data
            l = self.loss(outputs, target)

        return l[0].detach().cpu().numpy(), outputs["segmentation"][0].detach().cpu(), outputs["classification"].detach().cpu()

    def perform_actual_validation(self, save_probabilities: bool = False):
        self.set_deep_supervision_enabled(False)
        self.network.eval()


        validation_output_folder = join(self.output_folder, 'validation')
        maybe_mkdir_p(validation_output_folder)


        dataset_val = self.dataloader_val.generator
        dataset_val.reset()
        dataset_val.infinite = False
        dataset_val.batch_size = 1
        val_size =  len(dataset_val.indices)

        results = []
        dsc_pancreas = []
        dsc_lesion = []
        classification_pred = []
        classification_expected = []

        for _ in range(val_size):
            data = next(dataset_val)
            with torch.no_grad():
                loss, prediction, classification = self.actual_validation_step(data)

            keys = [x for x in data['keys']]
            self.print_to_log_file(f"predicting {keys}")

            self.print_to_log_file(f'{keys}, shape {data["data"].shape}, rank {self.local_rank}')
            # output_filename_truncated = join(validation_output_folder, keys)
            for i, key in enumerate(keys):
                expected_seg = data['target'][0][i]
                expected_classification = int(key.split('_')[1])

                # this needs to go into background processes
                results.append({'prediction':  prediction[i], 'classification': classification[i]})
                curr_pred = np.argmax(prediction[i], 0)
                curr_class = np.argmax(classification[i])
                lesion_score = dice_score(curr_pred, expected_seg[0])
                print(lesion_score)
                print(f'target class {expected_classification}, actual class {curr_class}')

                classification_pred.append(curr_class)
                classification_expected.append(expected_classification)
                dsc_pancreas.append(lesion_score[0])
                dsc_lesion.append(lesion_score[1])

        f1_score = f1_score_multiclass(torch.Tensor(classification_pred), torch.Tensor(classification_expected), 3)
        dsc_pancreas = np.mean(dsc_pancreas)
        dsc_lesion = np.mean(dsc_lesion)
        print(f'Model Quality: f1_score = {f1_score}, dsc_pancreas = {dsc_pancreas}, dsc_lesion = {dsc_lesion}')

def dice_score_inner(pred, target, epsilon=1e-6):
    # Compute intersection and union
    intersection = (pred * target).sum()
    pred_sum = pred.sum()
    target_sum = target.sum()

    dice_c = (2.0 * intersection + epsilon) / (pred_sum + target_sum + epsilon)

    return dice_c


def dice_score(pred, target):
    # Create binary masks for the current class c
    pred_c_12 = (pred > 0).float()
    target_c_12 = (target > 0).float()

    pancreas_dsc = dice_score_inner(pred_c_12, target_c_12)

    pred_c_2 = (pred == 2).float()
    target_c_2 = (target == 2).float()

    lesion_dsc = dice_score_inner(pred_c_2, target_c_2)

    return [pancreas_dsc, lesion_dsc]


def f1_score_multiclass(preds, targets, num_classes, ignore_index=None, eps=1e-7):
    """
    Computes the macro-averaged F1 score for multi-class classification.

    Args:
        preds (torch.Tensor): Predicted class indices, shape [N] or [batch_size, ...].
                              Values in {0, 1, ..., num_classes-1}.
        targets (torch.Tensor): Ground-truth class indices, same shape as preds.
        num_classes (int): Total number of classes.
        ignore_index (int or None): If not None, this class index is ignored in the calculation.
        eps (float): Small constant to avoid division by zero.

    Returns:
        float: Macro-averaged F1 score over all (non-ignored) classes.
    """
    preds = preds.long()
    targets = targets.long()

    f1_per_class = []

    for c in range(num_classes):
        if c == ignore_index:
            continue

        # One-vs-All: get binary masks for class c
        pred_c = (preds == c)
        target_c = (targets == c)

        TP = (pred_c & target_c).sum().float()
        FP = (pred_c & ~target_c).sum().float()
        FN = (~pred_c & target_c).sum().float()

        precision = TP / (TP + FP + eps)
        recall = TP / (TP + FN + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)

        f1_per_class.append(f1)

    if len(f1_per_class) == 0:
        return 0.0  # e.g., if we ignored all classes
    return torch.mean(torch.stack(f1_per_class)).item()