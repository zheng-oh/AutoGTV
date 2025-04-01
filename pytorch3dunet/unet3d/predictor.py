import os
import time
from concurrent import futures
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from skimage import measure
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from pytorch3dunet.datasets.hdf5 import AbstractHDF5Dataset
from pytorch3dunet.datasets.utils import remove_padding
from pytorch3dunet.unet3d.model import is_model_2d
from pytorch3dunet.unet3d.utils import get_logger

logger = get_logger('UNetPredictor')


def _get_output_file(dataset: AbstractHDF5Dataset, suffix: str = '_predictions', output_dir: str = None) -> Path:
    file_path = Path(dataset.file_path)
    input_dir = file_path.parent
    if output_dir is None:
        output_dir = input_dir
    else:
        output_dir = Path(output_dir)
    output_filename = file_path.stem + suffix + '.h5'
    return Path(output_dir) / output_filename

def _load_dataset(dataset: AbstractHDF5Dataset, internal_path: str) -> np.ndarray:
    file_path = dataset.file_path
    with h5py.File(file_path, 'r') as f:
        return f[internal_path][...]

def mean_iou(pred: np.ndarray, gt: np.ndarray, n_classes: int, avg: bool = False) -> list[float] | float:
    pred = pred.astype('uint16')
    gt = gt.astype('uint16')
    assert pred.shape == gt.shape, f'Predictions and ground truth have different shapes: {pred.shape} != {gt.shape}'
    per_class_iou = []
    for c in range(1, n_classes):
        intersection = np.logical_and(gt == c, pred == c).sum()
        union = np.logical_or(gt == c, pred == c).sum()
        iou = intersection / union if union > 0 else 1.0
        per_class_iou.append(iou)
    return np.mean(per_class_iou) if avg else per_class_iou

def dice_score(pred: np.ndarray, gt: np.ndarray, avg: bool = False) -> list[float] | float:
    pred = pred.astype('uint16')
    gt = gt.astype('uint16')
    assert pred.shape == gt.shape, f'Predictions and ground truth have different shapes: {pred.shape} != {gt.shape}'
    per_class_dice = []
    for c_pred, c_gt in zip(pred, gt):
        intersection = np.logical_and(c_gt, c_pred).sum()
        union = c_gt.sum() + c_pred.sum()
        dice = 2 * intersection / union if union > 0 else 1.0
        per_class_dice.append(dice)
    return np.mean(per_class_dice) if avg else per_class_dice

class _AbstractPredictor:
    def __init__(self,
                 model: nn.Module,
                 output_dir: str,
                 out_channels: int,
                 output_dataset: str = 'predictions',
                 save_segmentation: bool = False,
                 prediction_channel: int = None,
                 performance_metric: str = None,
                 gt_internal_path: str = None,
                 **kwargs):
        self.model = model
        self.output_dir = output_dir
        self.out_channels = out_channels
        self.output_dataset = output_dataset
        self.save_segmentation = save_segmentation
        self.prediction_channel = prediction_channel
        self.performance_metric = performance_metric
        self.gt_internal_path = gt_internal_path

    def __call__(self, test_loader: DataLoader) -> Any:
        raise NotImplementedError

class StandardPredictor(_AbstractPredictor):
    def __init__(self,
                 model: nn.Module,
                 output_dir: str,
                 out_channels: int,
                 output_file: str = None,  # 新增 output_file 参数
                 output_dataset: str = 'predictions',
                 save_segmentation: bool = False,
                 prediction_channel: int = None,
                 performance_metric: str = None,
                 gt_internal_path: str = None,
                 device: str = 'cpu',
                 **kwargs):
        super().__init__(model, output_dir, out_channels, output_dataset, save_segmentation, prediction_channel,
                         performance_metric, gt_internal_path, **kwargs)
        self.device = device
        self.output_file = output_file  # 保存传入的文件路径

    def __call__(self, test_loader):
        assert isinstance(test_loader.dataset, AbstractHDF5Dataset)
        logger.info(f"Processing '{test_loader.dataset.file_path}'...")
        start = time.perf_counter()

        logger.info(f'Running inference on {len(test_loader)} batches')
        volume_shape = test_loader.dataset.volume_shape

        if self.save_segmentation:
            prediction_shape = volume_shape
        else:
            if self.prediction_channel is not None:
                prediction_shape = (1,) + volume_shape
            else:
                prediction_shape = (self.out_channels,) + volume_shape

        # 使用传入的 output_file，若未提供则使用默认逻辑
        output_file = self.output_file if self.output_file else _get_output_file(dataset=test_loader.dataset, output_dir=self.output_dir)
        with h5py.File(output_file, 'w') as h5_output_file:
            logger.info('Allocating prediction arrays...')
            prediction_array = self._allocate_prediction_array(prediction_shape, h5_output_file)

            patch_halo = test_loader.dataset.halo_shape

            self.model.eval()
            with torch.no_grad():
                for input, indices in tqdm(test_loader):
                    input = input.to(self.device, non_blocking=True)
                    prediction = self.model(input)
                    prediction = torch.softmax(prediction, dim=1)  # 添加 softmax
                    logger.info(f"Prediction shape: {prediction.shape}, Min: {prediction.min()}, Max: {prediction.max()}")
                    prediction = remove_padding(prediction, patch_halo)
                    prediction = prediction.cpu().numpy()
                    for pred, index in zip(prediction, indices):
                        if self.save_segmentation:
                            if pred.shape[0] == 1:
                                pred = pred[0] > 0.5
                            else:
                                pred = np.argmax(pred, axis=0)
                            pred = pred.astype('uint16')
                            index = tuple(index)
                        else:
                            if self.prediction_channel is None:
                                channel_slice = slice(0, self.out_channels)
                            else:
                                channel_slice = slice(0, 1)
                                pred = np.expand_dims(pred[self.prediction_channel], axis=0)
                            index = (channel_slice,) + tuple(index)

                        prediction_array[index] = pred

            logger.info(f'Finished inference in {time.perf_counter() - start:.2f} seconds')
            output_type = 'segmentation' if self.save_segmentation else 'probability maps'
            logger.info(f'Saving {output_type} to: {output_file}')
            self._create_prediction_dataset(h5_output_file, prediction_array)

            if self.performance_metric is not None:
                assert self.gt_internal_path is not None
                gt = _load_dataset(test_loader.dataset, self.gt_internal_path)
                prediction_array = prediction_array[...]
                if self.performance_metric == 'dice':
                    result = dice_score(prediction_array, gt)
                else:
                    result = mean_iou(prediction_array, gt, n_classes=self.out_channels)
                return result

    def _allocate_prediction_array(self, output_shape, output_file):
        if self.save_segmentation:
            dtype = 'uint16'
        else:
            dtype = 'float32'
        return np.zeros(output_shape, dtype=dtype)

    def _create_prediction_dataset(self, h5_output_file, prediction_array):
        h5_output_file.create_dataset(self.output_dataset, data=prediction_array, compression="gzip")


class LazyPredictor(StandardPredictor):
    """
    Applies the model on the given dataset and saves the result in the `output_file` in the H5 format.
    Predicted patches are directly saved into the H5 and they won't be stored in memory. Since this predictor
    is slower than the `StandardPredictor` it should only be used when the predicted volume does not fit into RAM.
    """

    def __init__(self,
                 model: nn.Module,
                 output_dir: str,
                 out_channels: int,
                 output_dataset: str = 'predictions',
                 save_segmentation: bool = False,
                 prediction_channel: int = None,
                 performance_metric: str = None,
                 gt_internal_path: str = None,
                 **kwargs):
        super().__init__(model, output_dir, out_channels, output_dataset, save_segmentation, prediction_channel,
                         performance_metric, gt_internal_path, **kwargs)

    def _allocate_prediction_array(self, output_shape, output_file):
        if self.save_segmentation:
            dtype = 'uint16'
        else:
            dtype = 'float32'
        # allocate datasets for probability maps
        prediction_array = output_file.create_dataset(self.output_dataset,
                                                      shape=output_shape,
                                                      dtype=dtype,
                                                      chunks=True,
                                                      compression='gzip')
        return prediction_array

    def _create_prediction_dataset(self, h5_output_file, prediction_array):
        # no need to save the prediction array, it is already saved in the H5 file
        pass


class DSB2018Predictor(_AbstractPredictor):
    def __init__(self, model, output_dir, config, save_segmentation=True, pmaps_thershold=0.5, **kwargs):
        super().__init__(model, output_dir, config, **kwargs)
        self.pmaps_threshold = pmaps_thershold
        self.save_segmentation = save_segmentation

    def _slice_from_pad(self, pad):
        if pad == 0:
            return slice(None, None)
        else:
            return slice(pad, -pad)

    def __call__(self, test_loader):
        # Sets the module in evaluation mode explicitly
        self.model.eval()
        # initial process pool for saving results to disk
        executor = futures.ProcessPoolExecutor(max_workers=32)
        # Run predictions on the entire input dataset
        with torch.no_grad():
            for img, path in test_loader:
                # send batch to gpu
                if torch.cuda.is_available():
                    img = img.cuda(non_blocking=True)
                # forward pass
                pred = self.model(img)

                executor.submit(
                    dsb_save_batch,
                    self.output_dir,
                    path
                )

        print('Waiting for all predictions to be saved to disk...')
        executor.shutdown(wait=True)


def dsb_save_batch(output_dir, path, pred, save_segmentation=True, pmaps_thershold=0.5):
    def _pmaps_to_seg(pred):
        mask = (pred > pmaps_thershold)
        return measure.label(mask).astype('uint16')

    # convert to numpy array
    for single_pred, single_path in zip(pred, path):
        logger.info(f'Processing {single_path}')
        single_pred = single_pred.squeeze()

        # save to h5 file
        out_file = os.path.splitext(single_path)[0] + '_predictions.h5'
        if output_dir is not None:
            out_file = os.path.join(output_dir, os.path.split(out_file)[1])

        with h5py.File(out_file, 'w') as f:
            # logger.info(f'Saving output to {out_file}')
            f.create_dataset('predictions', data=single_pred, compression='gzip')
            if save_segmentation:
                f.create_dataset('segmentation', data=_pmaps_to_seg(single_pred), compression='gzip')
