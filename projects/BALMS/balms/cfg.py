def add_balms_config(cfg):
    """
    Add config for BALMS.
    """
    cfg.MODEL.ROI_HEADS.PRIOR_PROB = 0.001
    cfg.DECOUPLE_TRAINING = False
    cfg.META_REWEIGHT = False
