import pydicom
import h5py
import os
import torch
import torch.nn as nn
import numpy as np
from pytorch3dunet.unet3d import utils
from pytorch3dunet.unet3d.model import get_model
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import yaml
import logging


logger = utils.get_logger('UNet3DPredictor')
sample_name = os.getenv("SAMPLE_NAME", "default_value")

class UNet3DPredictor:
    def __init__(self, config_path="./pred_config.yaml"):
        """Initialize predictor with configuration."""
        self.config, self.config_path = self._load_config(config_path)
        self.dir_img = Path(self.config['data']['image_dir'])
        self.dir_mask = Path(self.config['data'].get('mask_dir', '')) if self.config['data'].get('mask_dir') else None
        self.target_depth = self.config['data'].get('target_depth', 100)
        self.output_dir = Path(self.config.get('output_dir', './predictions/'))
        self.threshold = self.config['visualization'].get('threshold', 0.5)
        self.model_path = Path(self.config.get('model_path', './models/best.pytorch'))
        self.device = self.config['device']
        self.slice_builder_config = self.config.get('slice_builder', {
            'name': 'SliceBuilder',
            'patch_shape': [50, 256, 256],
            'stride_shape': [50, 256, 256]
        })
        self.patch_shape = self.slice_builder_config['patch_shape']
        self.stride_shape = self.slice_builder_config['stride_shape']

    def _load_config(self, config_path):
        """Load YAML config file and set device."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Config file not found at {config_path}")
            raise

        device = config.get('device', 'cpu')
        if device == 'cpu':
            logger.warning('CPU mode forced, this may result in slow prediction')
        elif device == 'mps' and torch.backends.mps.is_available():
            config['device'] = 'mps'
        elif device == 'cuda' and torch.cuda.is_available():
            config['device'] = 'cuda'
        else:
            logger.warning('Requested device not available, using CPU')
            config['device'] = 'cpu'

        logger.info(f"Loaded config from {config_path}")
        return config, config_path

    def load_dicom_series(self, dicom_folder):
        """Load and preprocess DICOM series, retaining DICOM metadata."""
        if dicom_folder is None:
            return None, None, 0, 0, None

        dicom_folder = Path(dicom_folder)
        if not dicom_folder.exists() or not dicom_folder.is_dir():
            logger.warning(f"Directory not found or invalid: {dicom_folder}")
            return None, None, 0, 0, None

        files = [f for f in dicom_folder.glob('*.dcm')]
        if not files:
            logger.warning(f"No DICOM files found in {dicom_folder}")
            return None, None, 0, 0, None

        # 加载并按InstanceNumber排序
        slices = [pydicom.dcmread(f) for f in files]
        slices.sort(key=lambda x: float(x.InstanceNumber))
        image_3d_full = np.stack([s.pixel_array for s in slices], axis=0)

        current_depth = image_3d_full.shape[0]
        if current_depth < self.target_depth:
            pad_width = ((0, self.target_depth - current_depth), (0, 0), (0, 0))
            image_3d = np.pad(image_3d_full, pad_width, mode='constant', constant_values=0)
            start_idx, end_idx = 0, self.target_depth
        else:
            start_idx = max(0, (current_depth - self.target_depth) // 3)
            end_idx = min(current_depth, start_idx + self.target_depth)
            image_3d = image_3d_full[start_idx:end_idx]
            logger.info(f"Original depth: {current_depth}, Cropped depth: {image_3d.shape[0]}, Start: {start_idx}, End: {end_idx}")

        # 返回裁剪后的数据、完整数据、索引和DICOM slices
        return image_3d, image_3d_full, start_idx, end_idx, slices

    def load_patient_data(self):
        """Load patient data from data.image_dir and optionally data.mask_dir."""
        ct_3d, ct_3d_full, start_idx, end_idx, ct_slices = self.load_dicom_series(self.dir_img)
        if ct_3d is None:
            raise ValueError(f"Failed to load CT data from {self.dir_img}")

        mask_3d, mask_3d_full, _, _, mask_slices = self.load_dicom_series(self.dir_mask)

        logger.info(f"Loaded data: Cropped {ct_3d.shape}, Full {ct_3d_full.shape}")
        return {
            'raw': ct_3d,
            'raw_full': ct_3d_full,
            'label': mask_3d,
            'label_full': mask_3d_full,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'ct_slices': ct_slices,      # 新增：CT的DICOM slices
            'mask_slices': mask_slices   # 新增：掩码的DICOM slices（如果有）
        }

    def _extract_patches(self, volume):
        """Extract patches from volume using patch_shape and stride_shape."""
        patches = []
        z, y, x = volume.shape
        pz, py, px = self.patch_shape
        sz, sy, sx = self.stride_shape

        for k in range(0, max(1, z - pz + 1), sz):
            for j in range(0, max(1, y - py + 1), sy):
                for i in range(0, max(1, x - px + 1), sx):
                    patch = volume[k:k+pz, j:j+py, i:i+px]
                    if patch.shape == tuple(self.patch_shape):
                        patches.append(patch)
                    else:
                        pad_width = [(0, max(0, pz - patch.shape[0])), 
                                    (0, max(0, py - patch.shape[1])), 
                                    (0, max(0, px - patch.shape[2]))]
                        patches.append(np.pad(patch, pad_width, mode='constant', constant_values=0))
        return np.stack(patches, axis=0), (z, y, x)

    def _reconstruct_volume(self, patches, original_shape):
        """Reconstruct volume from patches."""
        z, y, x = original_shape
        pz, py, px = self.patch_shape
        sz, sy, sx = self.stride_shape
        reconstructed = np.zeros((self.config['model']['out_channels'], z, y, x))
        counts = np.zeros((z, y, x))

        patch_idx = 0
        for k in range(0, max(1, z - pz + 1), sz):
            for j in range(0, max(1, y - py + 1), sy):
                for i in range(0, max(1, x - px + 1), sx):
                    if patch_idx < len(patches):
                        patch = patches[patch_idx]
                        reconstructed[:, k:k+pz, j:j+py, i:i+px] += patch
                        counts[k:k+pz, j:j+py, i:i+px] += 1
                        patch_idx += 1

        reconstructed[:, counts > 0] /= counts[counts > 0]
        return reconstructed

    def load_model(self):
        """Load UNet3D model from checkpoint."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found at {self.model_path}")

        model = get_model(self.config['model'])
        logger.info(f"Loading model from {self.model_path}")
        checkpoint = torch.load(self.model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        if self.device == 'cuda' and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        return model.to(self.device)

    def save_prediction_as_dicom(self, data, prediction):
        """Save the full prediction as a DICOM series."""
        ct_slices = data['ct_slices']          # 原始CT的DICOM slices
        start_idx = data['start_idx']          # 裁剪起始索引
        end_idx = data['end_idx']              # 裁剪结束索引
        raw_full_shape = data['raw_full'].shape  # 原始CT的完整形状

        # 将预测结果转换为分割掩码
        if prediction.shape[0] > 1:  # 多通道输出，使用argmax
            pred_segmentation = np.argmax(prediction, axis=0)
        else:  # 单通道输出，使用阈值
            pred_segmentation = (prediction[0] > self.threshold).astype(np.uint8)

        # 调整预测结果到原始深度
        if end_idx - start_idx < raw_full_shape[0]:
            full_pred_segmentation = np.zeros(raw_full_shape, dtype=np.uint8)
            full_pred_segmentation[start_idx:end_idx] = pred_segmentation
        else:
            full_pred_segmentation = pred_segmentation


        # 创建保存DICOM序列的文件夹，命名为 result_ + 当前时间
        dicom_output_dir = self.output_dir / f'{sample_name}'
        dicom_output_dir.mkdir(parents=True, exist_ok=True)

        # 创建保存DICOM序列的文件夹
        # dicom_output_dir = self.output_dir / f'{self.dir_img.name}_pred_dicom'
        # dicom_output_dir.mkdir(parents=True, exist_ok=True)

        # 为每个切片生成DICOM文件
        for i, slice_ds in enumerate(ct_slices):
            new_ds = pydicom.dcmread(slice_ds.filename)  # 复制原始DICOM文件
            new_ds.PixelData = full_pred_segmentation[i].tobytes()  # 替换像素数据
            new_ds.Rows, new_ds.Columns = full_pred_segmentation[i].shape  # 更新行列信息
            new_ds.save_as(dicom_output_dir / f'slice_{i:04d}.dcm')  # 保存文件

        logger.info(f"Prediction DICOM series saved to {dicom_output_dir}")

    def predict(self, data):
        model = self.load_model()
        model.eval()

        ct_3d = data['raw']
        patches, original_shape = self._extract_patches(ct_3d)
        input_patches = torch.from_numpy(patches).float().unsqueeze(1).to(self.device)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_file = self.output_dir / f'{sample_name}.h5'

        with torch.no_grad():
            logger.info("Starting patch-based prediction...")
            predictions = []
            for i in range(0, len(patches), 2):
                batch = input_patches[i:i+2]
                pred = model(batch).cpu().numpy()
                predictions.append(pred)
            predictions = np.concatenate(predictions, axis=0)
            
            full_prediction = self._reconstruct_volume(predictions, original_shape)
            logger.info(f"Prediction shape: {full_prediction.shape}, min: {full_prediction.min()}, max: {full_prediction.max()}")
            with h5py.File(output_file, 'w') as f:
                f.create_dataset('prediction', data=full_prediction, compression='gzip')
            logger.info(f"Prediction saved to {output_file}")
        
        # 新增：保存预测结果为DICOM序列
        self.save_prediction_as_dicom(data, full_prediction)

        #self.visualize(data, full_prediction, output_file)

    def visualize(self, data, prediction, output_file):
        pred_segmentation = np.argmax(prediction, axis=0) if prediction.shape[0] > 1 else (prediction[0] > self.threshold).astype(np.uint8)
        label_data = data['label']
        raw_full_data = data['raw_full']
        start_idx, end_idx = data['start_idx'], data['end_idx']
        
        # 动态选择切片
        if label_data is None:
            max_slice_full = start_idx + np.argmax(pred_segmentation.sum(axis=(1, 2))) if pred_segmentation.sum() > 0 else raw_full_data.shape[0] // 2
        else:
            max_slice_full = start_idx + np.argmax(label_data.sum(axis=(1, 2)))
        
        logger.info(f"Visualizing slice: {max_slice_full}, pred unique: {np.unique(pred_segmentation)}")
        # 其余可视化代码保持不变
        # 根据是否有掩码数据动态调整子图数量
        if label_data is not None:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes = axes.flatten()
        else:
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes = axes.flatten()

        # 原始 CT 数据
        raw_img = axes[0].imshow(raw_full_data[max_slice_full], cmap='gray')
        axes[0].set_title(f'Original CT (Depth: {raw_full_data.shape[0]})')
        axes[0].axis('off')

        # 如果有掩码数据，显示 Ground Truth
        if label_data is not None:
            label_img = axes[1].imshow(label_data[max_slice_full - start_idx], cmap='jet')
            axes[1].set_title('Ground Truth')
            axes[1].axis('off')
            pred_img = axes[2].imshow(pred_segmentation[max_slice_full - start_idx], cmap='jet')
            axes[2].set_title(f'Prediction (Threshold {self.threshold})')
            axes[2].axis('off')
        else:
            pred_img = axes[1].imshow(pred_segmentation[max_slice_full - start_idx], cmap='jet')
            axes[1].set_title(f'Prediction (Threshold {self.threshold})')
            axes[1].axis('off')

        # 滑动条
        ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
        slider = Slider(ax_slider, 'Slice', 0, raw_full_data.shape[0] - 1, valinit=max_slice_full, valstep=1)

        def update(val):
            slice_idx = int(slider.val)
            raw_img.set_data(raw_full_data[slice_idx])
            cropped_idx = slice_idx - start_idx
            if 0 <= cropped_idx < pred_segmentation.shape[0]:
                if label_data is not None:
                    label_img.set_data(label_data[cropped_idx])
                pred_img.set_data(pred_segmentation[cropped_idx])
            else:
                if label_data is not None:
                    label_img.set_data(np.zeros_like(raw_full_data[0]))
                pred_img.set_data(np.zeros_like(raw_full_data[0]))
            fig.canvas.draw_idle()

        slider.on_changed(update)
        plt.tight_layout()
        plt.show()

    def run(self):
        """Execute prediction pipeline."""
        try:
            data = self.load_patient_data()
            self.predict(data)
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise

def main():
    predictor = UNet3DPredictor(config_path="./test_config.yaml")
    predictor.run()

if __name__ == '__main__':
    main()