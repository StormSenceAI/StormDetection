from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from src.preprocessing.radar_to_image import radar_to_image

# Preprocess radar data
radar_file = "data/sample_nexrad_file"  # Replace with actual NEXRAD file path
output_dir = "data/processed"
try:
    img_path = radar_to_image(radar_file, output_dir, output_filename="processed_radar.png")
except Exception as e:
    print(f"Error preprocessing radar file: {e}")
    img_path = None

# Load pre-trained YOLOv8 model (replace with fine-tuned model later)
model = YOLO("yolov8n.pt")

# Load the preprocessed radar image
if img_path:
    img = cv2.imread(img_path)
else:
    raise FileNotFoundError("No preprocessed image available. Please check radar preprocessing.")

if img is None:
    raise FileNotFoundError(f"Could not load image at {img_path}.")

# Run detection
results = model(img)

# Visualize the results
annotated_img = results[0].plot()  # Overlay bounding boxes on the image
plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Storm Detection Example")
plt.show()
