import pyart
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

def radar_to_image(radar_file_path, output_dir, output_filename="processed_radar.png"):
    """
    Convert a NEXRAD radar file to an image for object detection.

    Args:
        radar_file_path (str): Path to the NEXRAD Level II radar file.
        output_dir (str): Directory to save the output image.
        output_filename (str): Name of the output image file.

    Returns:
        str: Path to the saved image.
    """
    # Load the radar file using Py-ART
    try:
        radar = pyart.io.read_nexrad_archive(radar_file_path)
    except Exception as e:
        raise ValueError(f"Failed to load radar file: {e}")

    # Extract reflectivity field (dBZ) - a key indicator of storm intensity
    try:
        reflectivity = radar.fields["reflectivity"]["data"]
    except KeyError:
        raise KeyError("Reflectivity field not found in radar file.")

    # Mask invalid values (Py-ART often uses masked arrays)
    reflectivity = np.ma.filled(reflectivity, fill_value=-9999)

    # Normalize reflectivity for visualization (e.g., between -30 and 70 dBZ)
    reflectivity = np.clip(reflectivity, -30, 70)
    reflectivity = (reflectivity + 30) / 100  # Scale to 0-1 for grayscale

    # Convert to 8-bit grayscale image (0-255)
    img = (reflectivity * 255).astype(np.uint8)

    # Apply a color map for better visualization (optional, for human interpretation)
    img_colored = cv2.applyColorMap(img, cv2.COLORMAP_JET)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)

    # Save the image
    cv2.imwrite(output_path, img_colored)
    print(f"Saved processed radar image to {output_path}")

    return output_path

if __name__ == "__main__":
    # Example usage
    radar_file = "data/sample_nexrad_file"  # Replace with actual NEXRAD file path
    output_dir = "data/processed"
    try:
        radar_to_image(radar_file, output_dir)
    except Exception as e:
        print(f"Error processing radar file: {e}")
