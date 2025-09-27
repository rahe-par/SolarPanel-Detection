from ultralytics import YOLO

#Load your custom-trained model
# Update this path to point to your best model weights
MODEL_PATH = r"C:\Users\rahen\Documents\GUVI\SolarPanel\runs\detect\train\weights\best.pt" 
model = YOLO(MODEL_PATH)

#Run the validation
# The data argument should point to your data.yaml file
# This yaml file is inside the dataset folder you downloaded from Roboflow
DATA_YAML_PATH = r"C:\Users\rahen\Documents\GUVI\SolarPanel\SolarPanel.v1i.yolov8\data.yaml"
metrics = model.val(data=DATA_YAML_PATH)

#Print the key metrics
print("--- Model Evaluation Metrics ---")
print(f"mAP@50-95: {metrics.box.map:.4f}")    # Mean Average Precision
print(f"mAP@50: {metrics.box.map50:.4f}")      # mAP at IoU threshold of 0.5
print(f"Precision: {metrics.box.p[0]:.4f}") # Precision for the first class
print(f"Recall: {metrics.box.r[0]:.4f}")    # Recall for the first class
print("-" * 30)
print("Validation results and plots are saved in the 'runs/detect/val' folder.")