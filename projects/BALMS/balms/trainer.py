import contextlib
import time
import higher
from torch.nn.parallel import DistributedDataParallel
import logging
import os
from collections import OrderedDict
import torch

from .build import build_detection_meta_loader
from .meta_reweighter import Learner

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.logger import setup_logger
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
)
from detectron2.modeling import GeneralizedRCNNWithTTA

try:
    _nullcontext = contextlib.nullcontext  # python 3.7+
except AttributeError:

    @contextlib.contextmanager
    def _nullcontext(enter_result=None):
        yield enter_result


class PlainTrainer(DefaultTrainer):
    """
    Trainer for both end2end training and decoupled training.
    """
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        """
        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)

        # add decoupled training setting
        if cfg.DECOUPLE_TRAINING:
            logger.info("Decouple Training. Only training the last FC.")
            for name, param in model.named_parameters():
                if 'cls_score' not in name:
                    param.requires_grad = False

        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        # For training, wrap with DDP. But don't need this for inference.
        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False,
            )
        super(DefaultTrainer, self).__init__(model, data_loader, optimizer)

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        # Assume no other objects need to be checkpointed.
        # We can later make it checkpoint the stateful hooks
        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                    ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "cityscapes_instance":
            assert (
                    torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                    torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


class MetaReweightTrainer(PlainTrainer):
    """
    Trainer for decoupled training with Meta Reweighter
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        # init Meta Reweighter
        learner = Learner(cfg.MODEL.ROI_HEADS.NUM_CLASSES)
        learner.to(torch.device(cfg.MODEL.DEVICE))
        if comm.get_world_size() > 1:
            learner = DistributedDataParallel(
                learner, device_ids=[comm.get_local_rank()], broadcast_buffers=False,
            )
        self.learner = learner
        if comm.get_world_size() > 1:
            box_predictor = self.model.module.roi_heads.box_predictor
        else:
            box_predictor = self.model.roi_heads.box_predictor
        if isinstance(box_predictor, torch.nn.ModuleList):
            for predictor in box_predictor:
                predictor.register_meta_reweigher(self.learner)
        else:
            box_predictor.register_meta_reweigher(self.learner)
        self.optimizer_meta = torch.optim.Adam(self.learner.parameters(), lr=0.01)
        meta_data_loader = self.build_meta_loader(cfg)
        self._meta_data_loader_iter = iter(meta_data_loader)

    @classmethod
    def build_meta_loader(cls, cfg):
        return build_detection_meta_loader(cfg)

    def run_step(self):
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        self.meta_forward(data)     # meta step

        loss_dict = self.model(data)

        # pop loss_sigmoid that is only used for computing meta loss
        loss_sigmoid = [k for k in loss_dict if "loss_sigmoid" in k]
        for k in loss_sigmoid:
            loss_dict.pop(k)

        losses = sum(loss_dict.values())

        self.optimizer.zero_grad()
        losses.backward()

        # use a new stream so the ops don't wait for DDP
        with torch.cuda.stream(
                torch.cuda.Stream()
        ) if losses.device.type == "cuda" else _nullcontext():
            metrics_dict = loss_dict
            metrics_dict["data_time"] = data_time
            self._write_metrics(metrics_dict)
            self._detect_anomaly(losses, loss_dict)

        self.optimizer.step()

    def meta_forward(self, data):
        self.learner.train()
        self.optimizer.zero_grad()
        self.optimizer_meta.zero_grad()
        # copy_initial_weights should be  False because learner is regarded a parameter of model
        with higher.innerloop_ctx(self.model, self.optimizer, copy_initial_weights=False) as (fmodel, diffopt):
            loss_dict = fmodel(data)
            # aggregate loss
            loss = sum([v for k, v in loss_dict.items() if 'loss_cls' in k])
            diffopt.step(loss)

            meta_data = next(self._meta_data_loader_iter)
            meta_loss_dict = fmodel(meta_data)
            # aggregate meta loss
            meta_loss = sum([v for k, v in meta_loss_dict.items() if 'loss_sigmoid' in k])
            meta_loss.backward()
            self.optimizer_meta.step()

        self.learner.eval()

        self.log_meta_info()

    def log_meta_info(self, period=20):
        # log the learned weights
        if (self.iter + 1) % period != 0:
            return
        learner = self.learner.module if isinstance(self.learner, DistributedDataParallel) else self.learner
        prob = learner.fc[0].weight.sigmoid().squeeze(0)
        num_classes = prob.shape[0]
        print_str = 'Weights: '
        for i in range(0, num_classes, num_classes // 10):
            print_str += 'class{}={:.3f},'.format(i, prob[i].item())

        logger = logging.getLogger(__name__)
        logger.info(print_str)
