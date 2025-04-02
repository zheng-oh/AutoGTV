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

def visualize(dicom_folder, prediction_folder):
    """Visualize the CT and prediction results with 8-bit workaround."""
    # Load DICOM images (raw CT data)
    raw_full_data = load_dicom_images(dicom_folder)

    # Load predictions
    prediction_files = [f for f in os.listdir(prediction_folder) if f.endswith('.dcm')]
    prediction_files.sort()

    predictions = []
    for pred_file in prediction_files:
        pred_path = os.path.join(prediction_folder, pred_file)
        try:
            pred_data = pydicom.dcmread(pred_path)
            # Check if PixelData matches 8-bit size despite 16-bit metadata
            expected_8bit_size = pred_data.Rows * pred_data.Columns * pred_data.SamplesPerPixel
            if len(pred_data.PixelData) == expected_8bit_size and pred_data.BitsAllocated == 16:
                # Reinterpret as 8-bit data
                pixel_array = np.frombuffer(pred_data.PixelData, dtype=np.uint8).reshape(pred_data.Rows, pred_data.Columns)
            else:
                pixel_array = pred_data.pixel_array  # Use default decoding if metadata matches data
            predictions.append(pixel_array)
        except Exception as e:
            logger.error(f"Failed to load {pred_file}: {e}")
            predictions.append(np.zeros((512, 512), dtype=np.uint8))  # Fallback to zero array

    if not predictions:
        logger.error("No valid prediction files loaded.")
        return

    predictions = np.stack(predictions, axis=0)

    # Check for slice consistency
    if raw_full_data.shape[0] != predictions.shape[0]:
        logger.error(f"Slice mismatch: raw data has {raw_full_data.shape[0]} slices, predictions have {predictions.shape[0]} slices")
        return

    # Visualization
    max_slice_full = raw_full_data.shape[0] // 2
    logger.info(f"Visualizing slice: {max_slice_full}, prediction shape: {predictions.shape}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes = axes.flatten()

    raw_img = axes[0].imshow(raw_full_data[max_slice_full], cmap='gray')
    axes[0].set_title(f'Original CT (Slice: {max_slice_full})')
    axes[0].axis('off')

    pred_img = axes[1].imshow(predictions[max_slice_full], cmap='jet')
    axes[1].set_title(f'Prediction (Slice: {max_slice_full})')
    axes[1].axis('off')

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
    parser.add_argument('--samplename', type=str, required=True, help="Sample name (e.g., patient001).")
    args = parser.parse_args()

    samplename = args.samplename
    dicom_folder = os.path.join('test', 'image', samplename)
    prediction_folder = os.path.join('test', 'gtv', samplename)

    logger.setLevel(logging.INFO)

    if not os.path.exists(dicom_folder):
        logger.error(f"DICOM folder not found: {dicom_folder}")
        return

    if not os.path.exists(prediction_folder):
        logger.error(f"Prediction folder not found: {prediction_folder}")
        return

    visualize(dicom_folder, prediction_folder)

if __name__ == '__main__':
    main()