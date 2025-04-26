from ultralytics import YOLO
import cv2
import numpy as np

class StormDetector:
    """
    A class to handle storm detection on radar images using YOLOv8.
    """
    def __init__(self, model_path="yolov8n.pt", conf_threshold=0.5, iou_threshold=0.5):
        """
        Initialize the storm detector with a YOLOv8 model.

        Args:
            model_path (str): Path to the YOLOv8 model weights (e.g., yolov8n.pt or fine-tuned model).
            conf_threshold (float): Confidence threshold for detections.
            iou_threshold (float): IoU threshold for non-max suppression.
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def detect(self, image_path):
        """
        Detect storms in a radar image.

        Args:
            image_path (str): Path to the preprocessed radar image.

        Returns:
            list: List of detections, where each detection is a dict with keys:
                  'bbox': [x_min, y_min, x_max, y_max],
                  'confidence': float,
                  'class': str (e.g., 'storm').
        """
        # Load the image
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not load image at {image_path}")

        # Run inference
        results = self.model.predict(
            img,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )

        # Extract detections
        detections = []
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes in [x_min, y_min, x_max, y_max] format
            confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
            class_ids = result.boxes.cls.cpu().numpy()  # Class IDs

            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                detection = {
                    "bbox": box.tolist(),
                    "confidence": float(conf),
                    "class": self.model.names[int(cls_id)]  # Map class ID to name (e.g., 'storm')
                }
                detections.append(detection)

        return detections

if __name__ == "__main__":
    # Example usage
    detector = StormDetector(model_path="yolov8n.pt", conf_threshold=0.5)
    image_path = "data/processed/processed_radar.png"
    try:
        detections = detector.detect(image_path)
        print("Detections:", detections)
    except Exception as e:
        print(f"Error during detection: {e}")
