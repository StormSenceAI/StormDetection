import cv2
import matplotlib.pyplot as plt
import numpy as np
from src.utils.helpers import setup_logging

logger = setup_logging()

def visualize_detections(image_path, detections, output_path=None, show=True):
    """
    Visualize storm detections on a radar image by drawing bounding boxes and labels.

    Args:
        image_path (str): Path to the radar image.
        detections (list): List of detections from StormDetector.
        output_path (str, optional): Path to save the annotated image. If None, image is not saved.
        show (bool): Whether to display the image using Matplotlib.
    """
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image at {image_path}")

    # Draw each detection
    for detection in detections:
        bbox = detection["bbox"]
        confidence = detection["confidence"]
        class_name = detection["class"]

        # Extract bounding box coordinates
        x_min, y_min, x_max, y_max = map(int, bbox)

        # Draw bounding box (blue color, thickness 2)
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

        # Draw label with confidence
        label = f"{class_name}: {confidence:.2f}"
        cv2.putText(
            img,
            label,
            (x_min, y_min - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2
        )

    # Convert BGR (OpenCV) to RGB (Matplotlib)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Save the image if output_path is provided
    if output_path:
        cv2.imwrite(output_path, img)
        logger.info(f"Saved annotated image to {output_path}")

    # Display the image if show is True
    if show:
        plt.figure(figsize=(10, 10))
        plt.imshow(img_rgb)
        plt.axis("off")
        plt.title("Storm Detections")
        plt.show()

if __name__ == "__main__":
    # Example usage (mock detections for testing)
    image_path = "data/processed/processed_radar.png"
    mock_detections = [
        {
            "bbox": [100, 100, 200, 200],
            "confidence": 0.85,
            "class": "storm"
        },
        {
            "bbox": [300, 300, 400, 400],
            "confidence": 0.72,
            "class": "storm"
        }
    ]

    try:
        visualize_detections(
            image_path,
            mock_detections,
            output_path="data/processed/annotated_radar.png",
            show=True
        )
    except Exception as e:
        logger.error(f"Error during visualization: {e}")
