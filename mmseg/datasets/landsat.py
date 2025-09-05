# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
# from .basesegdataset import BaseCDDataset
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class LandsatDataset(BaseSegDataset):
# class LandsatDataset(BaseCDDataset):
    """Landsat dataset.

    In segmentation map annotation for Potsdam dataset, 0 is the ignore index.
    ``reduce_zero_label`` should be set to True. The ``img_suffix`` and
    ``seg_map_suffix`` are both fixed to '.tif'.
    """
    METAINFO = dict(
        # classes=('Background', 'Cropland', 'Forest', 'Shrub', 'Grassland',
        #          'Water', 'Sonw', 'Barren', 'Impervious', 'Wetland'),
        # palette=[[0, 0, 0], [250, 227, 156], [68, 111, 51], [51, 160, 44], [171, 211, 123],
        #          [30, 105, 180], [166, 206, 227], [207, 189, 163], [226, 66, 144], [40, 155, 232]])
        classes=('Background', 'City'),
        palette=[[0, 0, 0], [255, 2555, 255]])

    def __init__(self,
                 img_suffix='.tif',
                 # img_suffix2='.TIF',
                 seg_map_suffix='.tif',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            # img_suffix2=img_suffix2,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
