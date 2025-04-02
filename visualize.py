import os
import numpy as np
import pydicom
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import logging

logger = logging.getLogger('UNet3DVisualizer')

def load_dicom_images(dicom_folder):
    """Load DICOM images from a folder and return them as a 3D numpy array."""
    dicom_files = [f for f in os.listdir(dicom_folder) if f.endswith('.dcm')]
    dicom_files.sort()  # Make sure the files are in order of slices

    images = []
    for dicom_file in dicom_files:
        dicom_path = os.path.join(dicom_folder, dicom_file)
        dicom_data = pydicom.dcmread(dicom_path)
        images.append(dicom_data.pixel_array)

    images = np.stack(images, axis=0)
    return images

def visualize(config, dicom_folder, prediction_folder):
    """Visualize the CT, ground truth, and prediction results."""
    # Load DICOM images (raw CT data)
    raw_full_data = load_dicom_images(dicom_folder)

    # Load predictions (assuming each DICOM slice has a corresponding prediction file)
    prediction_files = [f for f in os.listdir(prediction_folder) if f.endswith('.dcm')]
    prediction_files.sort()  # Make sure the files are in order of slices

    predictions = []
    for pred_file in prediction_files:
        pred_path = os.path.join(prediction_folder, pred_file)
        pred_data = pydicom.dcmread(pred_path)
        predictions.append(pred_data.pixel_array)

    predictions = np.stack(predictions, axis=0)

    # Select the slice to visualize
    max_slice_full = raw_full_data.shape[0] // 2  # Default to the middle slice

    logger.info(f"Visualizing slice: {max_slice_full}, prediction shape: {predictions.shape}")

    # Create the figure and axes
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes = axes.flatten()

    # Plot the images
    raw_img = axes[0].imshow(raw_full_data[max_slice_full], cmap='gray')
    axes[0].set_title(f'Original CT (Slice: {max_slice_full})')
    axes[0].axis('off')

    pred_img = axes[1].imshow(predictions[max_slice_full], cmap='jet')
    axes[1].set_title(f'Prediction (Slice: {max_slice_full})')
    axes[1].axis('off')

    # Add a slider to navigate slices
    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
    slider = Slider(ax_slider, 'Slice', 0, raw_full_data.shape[0] - 1, valinit=max_slice_full, valstep=1)

    def update(val):
        slice_idx = int(slider.val)
        raw_img.set_data(raw_full_data[slice_idx])
        pred_img.set_data(predictions[slice_idx])
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.tight_layout()
    plt.show()

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Visualize CT and prediction results from DICOM files.")
    parser.add_argument('--config', type=str, required=True, help="Path to the config file.")
    parser.add_argument('--dicom', type=str, required=True, help="Path to the DICOM folder (CT data).")
    parser.add_argument('--prediction', type=str, required=True, help="Path to the DICOM folder (prediction data).")
    args = parser.parse_args()

    # You could add more code here to load the configuration if needed
    logger.setLevel(logging.INFO)

    visualize(args.config, args.dicom, args.prediction)

if __name__ == '__main__':
    main()
