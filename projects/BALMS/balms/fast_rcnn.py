# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import math
import torch

from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Linear, ShapeSpec
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputs, FastRCNNOutputLayers

from .lvis_v0_5_categories import get_image_count_frequency

logger = logging.getLogger(__name__)


class SigmoidFastRCNNOutputs(FastRCNNOutputs):
    def get_expanded_label(self):
        """
        Expand the label to one-hot and discard the BG class
        """
        target = self.pred_class_logits.new_zeros(self.n_i, self.n_c + 1)
        target[torch.arange(self.n_i), self.gt_classes] = 1
        return target[:, :self.n_c]

    def sigmoid_loss(self):
        """
        Sigmoid baseline
        """
        self.n_i, self.n_c = self.pred_class_logits.size()
        self.target = self.get_expanded_label()
        cls_loss = F.binary_cross_entropy_with_logits(self.pred_class_logits, self.target,
                                                      reduction='none')
        return torch.sum(cls_loss) / self.n_i

    def losses(self):
        return {"loss_cls": self.sigmoid_loss(), "loss_box_reg": self.box_reg_loss()}

    def predict_probs(self):
        """
        Deprecated
        """
        probs = torch.sigmoid(self.pred_class_logits)
        # add a dummy probs, representing a fake score for background class
        n = probs.size(0)
        dummy_probs = probs.new_zeros(n, 1)
        probs = torch.cat([probs, dummy_probs], dim=1)

        return probs.split(self.num_preds_per_image, dim=0)


class BalancedSoftmaxFastRCNNOutputs(SigmoidFastRCNNOutputs):
    def __init__(self, *args, freq_info=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.freq_info = freq_info

    def balanced_softmax_loss(self):
        """
        Sigmoid variant of Balanced Softmax
        """
        self.n_i, self.n_c = self.pred_class_logits.size()
        self.target = self.get_expanded_label()

        njIn = self.freq_info.type_as(self.pred_class_logits)

        weight = (1. - njIn) / njIn     # Discard the constant 1/(k-1) to keep log(weight) mostly positive
        weight = weight.unsqueeze(0).expand(self.n_i, -1)

        fg_ind = self.gt_classes != self.n_c
        self.pred_class_logits[fg_ind] = (self.pred_class_logits - weight.log())[fg_ind]    # Only apply to  FG samples

        cls_loss = F.binary_cross_entropy_with_logits(self.pred_class_logits, self.target,
                                                      reduction='none')

        return torch.sum(cls_loss) / self.n_i

    def losses(self):
        return {"loss_cls": self.balanced_softmax_loss(), "loss_box_reg": self.box_reg_loss()}


class BALMSFastRCNNOutputs(BalancedSoftmaxFastRCNNOutputs):
    def __init__(self, *args, learner=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.learner = learner

    def balms_loss(self):
        """
        Sigmoid variant of Balanced Softmax with Meta Reweighter
        """
        self.balanced_softmax_loss()    # modify logit to Balanced Softmax

        # re-weight
        cls_loss = F.binary_cross_entropy_with_logits(self.pred_class_logits, self.target,
                                                      reduction='none')
        fg_ind = self.gt_classes != self.n_c
        cls_loss = cls_loss.sum(-1)
        cls_loss[fg_ind] = self.learner(cls_loss[fg_ind], self.target[fg_ind])

        return torch.sum(cls_loss) / self.n_i

    def losses(self):
        return {"loss_cls": self.balms_loss(), "loss_box_reg": self.box_reg_loss(), "loss_sigmoid": self.sigmoid_loss()}


class SigmoidFastRCNNOutputLayers(FastRCNNOutputLayers):
    @configurable
    def __init__(
            self,
            input_shape: ShapeSpec,
            *args,
            num_classes: int,
            prior_prob: float = 0.001,
            **kwargs
    ):
        super().__init__(
            input_shape=input_shape,
            *args,
            num_classes=num_classes,
            **kwargs
        )
        # re-init the out dimension of the last FC layer to exclude the bg class
        if isinstance(input_shape, int):  # some backward compatibility
            input_shape = ShapeSpec(channels=input_shape)
        input_size = input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)
        self.cls_score = Linear(input_size, num_classes)  # no +1 since no BG class
        nn.init.normal_(self.cls_score.weight, std=0.01)
        # init the bias with prior prob for stabler training
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_score.bias, bias_value)

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret["prior_prob"] = cfg.MODEL.ROI_HEADS.PRIOR_PROB
        return ret

    def losses(self, predictions, proposals):
        scores, proposal_deltas = predictions
        losses = SigmoidFastRCNNOutputs(
            self.box2box_transform,
            scores,
            proposal_deltas,
            proposals,
            self.smooth_l1_beta,
            self.box_reg_loss_type,
        ).losses()
        return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}

    def predict_probs(self, predictions, proposals):
        scores, _ = predictions
        num_inst_per_image = [len(p) for p in proposals]

        probs = torch.sigmoid(scores)
        # add a dummy probs, representing a fake score for background class
        n = probs.size(0)
        dummy_probs = probs.new_zeros(n, 1)
        probs = torch.cat([probs, dummy_probs], dim=1)

        return probs.split(num_inst_per_image, dim=0)


class BalancedSoftmaxFastRCNNOutputLayers(SigmoidFastRCNNOutputLayers):
    @configurable
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # load freq
        self.freq_info = torch.FloatTensor(get_image_count_frequency())

    def losses(self, predictions, proposals):
        scores, proposal_deltas = predictions
        losses = BalancedSoftmaxFastRCNNOutputs(
            self.box2box_transform,
            scores,
            proposal_deltas,
            proposals,
            self.smooth_l1_beta,
            self.box_reg_loss_type,
            freq_info=self.freq_info
        ).losses()
        return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}


class BALMSFastRCNNOutputLayers(BalancedSoftmaxFastRCNNOutputLayers):
    @configurable
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.learner = None

    def register_meta_reweigher(self, learner):
        self.learner = learner

    def losses(self, predictions, proposals):
        scores, proposal_deltas = predictions
        losses = BALMSFastRCNNOutputs(
            self.box2box_transform,
            scores,
            proposal_deltas,
            proposals,
            self.smooth_l1_beta,
            self.box_reg_loss_type,
            freq_info=self.freq_info,
            learner=self.learner
        ).losses()
        return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}
