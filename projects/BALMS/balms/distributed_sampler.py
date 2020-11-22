# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from collections import defaultdict
import torch

from detectron2.data.samplers import RepeatFactorTrainingSampler


class ClassBalancedTrainingSampler(RepeatFactorTrainingSampler):
    """
    Class Balanced Sampler. Modified from RepeatFactorTrainingSampler
    """
    @staticmethod
    def repeat_factors_by_inverse_category_frequency(dataset_dicts):
        """
        Compute (fractional) per-image repeat factors based on the inverse of the
        category frequency.
        Args:
            dataset_dicts (list[dict]): annotations in Detectron2 dataset format.
        Returns:
            torch.Tensor: the i-th element is the repeat factor for the dataset image
                at index i.
        """
        # 1. For each category c, compute the fraction of images that contain it: f(c)
        category_freq = defaultdict(int)
        for dataset_dict in dataset_dicts:  # For each image (without repeats)
            cat_ids = {ann["category_id"] for ann in dataset_dict["annotations"]}
            for cat_id in cat_ids:
                category_freq[cat_id] += 1
        num_images = len(dataset_dicts)
        for k, v in category_freq.items():
            category_freq[k] = v / num_images

        # 2. For each category c, compute the category-level repeat factor:
        #    r(c) = 1 / f(c)
        category_rep = {
            cat_id: 1. / cat_freq
            for cat_id, cat_freq in category_freq.items()
        }

        # 3. For each image I, compute the image-level repeat factor:
        #    r(I) = max_{c in I} r(c)
        rep_factors = []
        for dataset_dict in dataset_dicts:
            cat_ids = {ann["category_id"] for ann in dataset_dict["annotations"]}
            rep_factor = max({category_rep[cat_id] for cat_id in cat_ids})
            rep_factors.append(rep_factor)

        return torch.tensor(rep_factors, dtype=torch.float32)
