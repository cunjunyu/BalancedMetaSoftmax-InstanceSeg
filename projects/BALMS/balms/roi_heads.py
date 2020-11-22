# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2.modeling.roi_heads.roi_heads import *
from .fast_rcnn import BalancedSoftmaxFastRCNNOutputLayers, SigmoidFastRCNNOutputLayers, BALMSFastRCNNOutputLayers


@ROI_HEADS_REGISTRY.register()
class SigmoidROIHeads(StandardROIHeads):
    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        ret = super()._init_box_head(cfg, input_shape)
        box_predictor = SigmoidFastRCNNOutputLayers(cfg, ret["box_head"].output_shape)
        ret["box_predictor"] = box_predictor
        return ret


@ROI_HEADS_REGISTRY.register()
class BalancedSoftmaxROIHeads(SigmoidROIHeads):
    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        ret = super()._init_box_head(cfg, input_shape)
        box_predictor = BalancedSoftmaxFastRCNNOutputLayers(cfg, ret["box_head"].output_shape)
        ret["box_predictor"] = box_predictor
        return ret


@ROI_HEADS_REGISTRY.register()
class BALMSROIHeads(BalancedSoftmaxROIHeads):
    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        ret = super()._init_box_head(cfg, input_shape)
        box_predictor = BALMSFastRCNNOutputLayers(cfg, ret["box_head"].output_shape)
        ret["box_predictor"] = box_predictor
        return ret
