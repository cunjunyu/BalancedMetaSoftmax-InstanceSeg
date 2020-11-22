from detectron2.data.build import *
import logging

from detectron2.data.common import DatasetFromList, MapDataset
from detectron2.data.dataset_mapper import DatasetMapper
from .distributed_sampler import ClassBalancedTrainingSampler


def build_detection_meta_loader(cfg, mapper=None):
    """
    build the meta set from training data with Class Balanced Sampling
    """
    dataset_dicts = get_detection_dataset_dicts(
        cfg.DATASETS.TRAIN,
        filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
        min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
        if cfg.MODEL.KEYPOINT_ON
        else 0,
        proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
    )
    dataset = DatasetFromList(dataset_dicts, copy=False)

    if mapper is None:
        mapper = DatasetMapper(cfg, True)
    dataset = MapDataset(dataset, mapper)

    logger = logging.getLogger(__name__)
    logger.info("Using training sampler Class Balanced Sampler")
    repeat_factors = ClassBalancedTrainingSampler.repeat_factors_by_inverse_category_frequency(dataset_dicts)
    sampler = ClassBalancedTrainingSampler(repeat_factors)
    return build_batch_data_loader(
        dataset,
        sampler,
        cfg.SOLVER.IMS_PER_BATCH,
        aspect_ratio_grouping=cfg.DATALOADER.ASPECT_RATIO_GROUPING,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
    )