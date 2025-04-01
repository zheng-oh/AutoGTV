import pydicom
import h5py
import importlib
import os
import torch
import torch.nn as nn
import numpy as np
from pytorch3dunet.unet3d import utils
from pytorch3dunet.unet3d.config import load_config
from pytorch3dunet.unet3d.model import get_model
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

logger = utils.get_logger('UNet3DPredict')

dir_img = Path('/Users/zxing/Documents/科研/课题项目比赛/比赛/生医比赛/bmedesign/training/image/')
dir_mask = Path('/Users/zxing/Documents/科研/课题项目比赛/比赛/生医比赛/bmedesign/training/gtv/')

def load_dicom_series(dicom_folder, target_depth=100):
    slices = [pydicom.dcmread(os.path.join(dicom_folder, f)) for f in os.listdir(dicom_folder) if f.endswith('.dcm')]
    slices.sort(key=lambda x: float(x.InstanceNumber))
    image_3d_full = np.stack([s.pixel_array for s in slices], axis=0)  # Full uncropped data
    
    current_depth = image_3d_full.shape[0]
    image_3d = image_3d_full.copy()  # Make a copy to crop/pad
    if current_depth < target_depth:
        pad_width = ((0, target_depth - current_depth), (0, 0), (0, 0))  # Tail padding
        image_3d = np.pad(image_3d, pad_width, mode='constant', constant_values=0)
        start_index = 0
        end_index = target_depth
    else:
        min_index = (current_depth - target_depth) // 3
        start_index = max(0, min_index)
        end_index = min(current_depth, start_index + target_depth)
        image_3d = image_3d[start_index:end_index]  # Cropped to target_depth
        print(f"Original depth: {current_depth}, Cropped depth: {image_3d.shape[0]}, Start: {start_index}, End: {end_index}")
    
    return image_3d, image_3d_full, start_index, end_index  # Return cropped, full data, and indices

def load_single_patient_data(patient_id):
    ct_3d, ct_3d_full, start_idx, end_idx = load_dicom_series(dir_img / patient_id, target_depth=100)
    mask_3d, _, _, _ = load_dicom_series(dir_mask / patient_id, target_depth=100)  # We only need cropped mask
    print(f"Loaded {patient_id}: Cropped {ct_3d.shape}, Full {ct_3d_full.shape}")
    return {'raw': ct_3d, 'raw_full': ct_3d_full, 'label': mask_3d, 'start_idx': start_idx, 'end_idx': end_idx}

def load_model_and_predict(data_in_memory):
    config, config_path = load_config()
    logger.info(f"Loaded config: {config}")
    model = get_model(config['model'])
    model_path = config.get('model_path', os.path.join(config['trainer']['checkpoint_dir'], 'best_checkpoint.pytorch'))
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
    logger.info(f'Loading model from {model_path}...')
    utils.load_checkpoint(model_path, model)

    requested_device = config['device']
    if requested_device == 'mps' and torch.backends.mps.is_available():
        try:
            test_tensor = torch.randn(1, 1, 50, 256, 256).to('mps')
            nn.MaxPool3d(2)(test_tensor)
            device = 'mps'
        except NotImplementedError:
            logger.warning("MPS does not support max_pool3d_with_indices. Falling back to CPU.")
            device = 'cpu'
    elif requested_device != 'cpu' and torch.cuda.is_available():
        device = 'cuda'
        model = nn.DataParallel(model) if torch.cuda.device_count() > 1 else model
        logger.info(f'Using {torch.cuda.device_count()} GPUs for prediction')
    else:
        device = 'cpu'
        logger.info("Using CPU for prediction")

    model = model.to(device)
    model.eval()

    ct_3d = data_in_memory['raw']  # Shape: (100, H, W)
    input_tensor = torch.from_numpy(ct_3d).float().unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 100, H, W)
    input_tensor = input_tensor.to(device)

    output_dir = config.get('output_dir', './predictions/')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'{patient_id}_pred.h5')

    predictor = get_predictor(model, output_dir, output_file, config)
    with torch.no_grad():
        logger.info("Starting prediction on in-memory data...")
        predictions = predictor.predict_in_memory(input_tensor)
        logger.info(f"Prediction completed. Results saved to {output_file}")

    visualize_slices_from_memory(data_in_memory['raw_full'], data_in_memory['label'], predictions, 
                                 data_in_memory['start_idx'], data_in_memory['end_idx'], 
                                 output_file, threshold=0.5)
    logger.info("Visualization completed.")

def get_predictor(model, output_dir, output_file, config):
    predictor_config = config.get('predictor', {})
    class_name = predictor_config.get('name', 'StandardPredictor')
    m = importlib.import_module('pytorch3dunet.unet3d.predictor')
    predictor_class = getattr(m, class_name)
    out_channels = config['model'].get('out_channels', 2)
    
    class InMemoryPredictor(predictor_class):
        def predict_in_memory(self, input_tensor):
            prediction = self.model(input_tensor)
            prediction = prediction.cpu().numpy()  # Shape: (1, C, 100, H, W)
            with h5py.File(self.output_file, 'w') as f:
                f.create_dataset('prediction', data=prediction)
            return prediction

    return InMemoryPredictor(model, output_dir, out_channels, output_file=output_file, device=config['device'], **predictor_config)

def visualize_slices_from_memory(raw_full_data, label_data, pred_data, start_idx, end_idx, output_file, threshold=0.4):
    """
    Visualize full original CT, ground truth (if available), and predictions with a slider.

    Parameters:
        raw_full_data (ndarray): Full uncropped raw image data (current_depth, H, W).
        label_data (ndarray): Ground truth labels (100, H, W) or None if not available.
        pred_data (ndarray): Predictions data (1, C, 100, H, W).
        start_idx (int): Start index of cropping.
        end_idx (int): End index of cropping.
        output_file (str): Path to save the prediction file.
        threshold (float): Threshold for binary segmentation of predictions.
    """
    # Handle prediction data
    pred_data = pred_data[0]  # Remove batch dimension, Shape: (C, 100, H, W)
    if pred_data.shape[0] > 1:  # Multi-class output
        pred_segmentation = np.argmax(pred_data, axis=0)  # Shape: (100, H, W)
    else:  # Single-channel output
        pred_segmentation = (pred_data[0] > threshold).astype(np.uint8)  # Shape: (100, H, W)

    # Ensure label_data is binary if available
    if label_data is not None:
        label_data = (label_data > 0).astype(np.uint8)  # Shape: (100, H, W)
        # Find the slice with the most foreground pixels in the ground truth
        foreground_pixels_per_slice = label_data.sum(axis=(1, 2))
        max_slice_cropped = np.argmax(foreground_pixels_per_slice)
        max_slice_full = start_idx + max_slice_cropped  # Corresponding slice in full data
        print(f"Slice with most foreground pixels in cropped data: {max_slice_cropped}, "
              f"Pixels: {foreground_pixels_per_slice[max_slice_cropped]}, Full slice: {max_slice_full}")
    else:
        max_slice_full = raw_full_data.shape[0] // 2  # Default to middle if no ground truth

    # Set up the plot with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plt.subplots_adjust(bottom=0.25)

    # Initial slice for full data
    slice_idx_full = max_slice_full

    # Full original CT
    raw_img = axes[0].imshow(raw_full_data[slice_idx_full], cmap='gray')
    axes[0].set_title(f'Original CT (Depth: {raw_full_data.shape[0]})')
    axes[0].axis('off')

    # Ground Truth (if available)
    if label_data is not None and start_idx <= slice_idx_full < end_idx:
        label_img = axes[1].imshow(label_data[slice_idx_full - start_idx], cmap='jet')
    else:
        label_img = axes[1].imshow(np.zeros_like(raw_full_data[0]), cmap='jet')  # Placeholder
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')

    # Prediction
    if start_idx <= slice_idx_full < end_idx:
        pred_img = axes[2].imshow(pred_segmentation[slice_idx_full - start_idx], cmap='jet')
    else:
        pred_img = axes[2].imshow(np.zeros_like(raw_full_data[0]), cmap='jet')  # Placeholder
    axes[2].set_title(f'Prediction (Threshold {threshold})')
    axes[2].axis('off')

    # Add slider for full data (current_depth)
    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
    slider = Slider(ax_slider, 'Slice', 0, raw_full_data.shape[0] - 1, valinit=slice_idx_full, valstep=1)

    def update(val):
        slice_idx_full = int(slider.val)
        raw_img.set_data(raw_full_data[slice_idx_full])
        if start_idx <= slice_idx_full < end_idx:
            slice_idx_cropped = slice_idx_full - start_idx
            if label_data is not None:
                label_img.set_data(label_data[slice_idx_cropped])
            pred_img.set_data(pred_segmentation[slice_idx_cropped])
        else:
            if label_data is not None:
                label_img.set_data(np.zeros_like(raw_full_data[0]))
            pred_img.set_data(np.zeros_like(raw_full_data[0]))
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.tight_layout()
    plt.show()

def main():
    global patient_id
    patient_id = "training_109"  # Replace with the desired patient folder name
    data_in_memory = load_single_patient_data(patient_id)
    load_model_and_predict(data_in_memory)

if __name__ == '__main__':
    main()