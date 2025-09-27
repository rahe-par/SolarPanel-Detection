import os
import matplotlib.pyplot as plt
import cv2 
import random

#Configuration
DATASET_PATH = r"C:\Users\rahen\Documents\GUVI\SolarPanel\Solar_Panel_Dataset"
CLASSES = os.listdir(DATASET_PATH)

#Count Images in Each Class
print("--- Image Count per Class ---")
class_counts = {}
for solar_class in CLASSES:
    class_path = os.path.join(DATASET_PATH, solar_class)
    # Check if it's a directory
    if os.path.isdir(class_path):
        count = len(os.listdir(class_path))
        class_counts[solar_class] = count
        print(f"Class '{solar_class}': {count} images")
print("-" * 30)


#Visualize Sample Images (Corrected Version)
def visualize_samples(dataset_path, classes):
    plt.figure(figsize=(15, 10))
    # Filter out any non-directory items that might be in the root folder
    class_dirs = [d for d in classes if os.path.isdir(os.path.join(dataset_path, d))]

    for i, solar_class in enumerate(class_dirs):
        class_path = os.path.join(dataset_path, solar_class)
        
        # Get a list of all images and filter for common image extensions
        all_images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not all_images:
            print(f"Warning: No images found in '{class_path}'. Skipping.")
            continue

        random_image_file = random.choice(all_images)
        image_path = os.path.join(class_path, random_image_file)

        # Read the image using OpenCV
        img = cv2.imread(image_path)
        
        # Check if the image was loaded successfully before processing
        if img is None:
            print(f"Warning: Failed to load image: {image_path}. Skipping.")
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert from BGR to RGB for matplotlib

        # Add image to the plot
        plt.subplot(2, 3, i + 1) # Create a 2x3 grid of plots
        plt.imshow(img)
        plt.title(f"Class: {solar_class}")
        plt.axis('off')

    plt.suptitle("Sample Images from Each Class", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# Run the visualization
visualize_samples(DATASET_PATH, CLASSES)