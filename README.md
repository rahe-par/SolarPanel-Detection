# SolarGuard: AI-Powered Solar Panel Defect Detection

SolarGuard is an intelligent system that uses deep learning to automate the inspection of solar panels. It leverages computer vision to classify the overall condition of a panel and uses object detection to pinpoint the exact location of specific defects like dust, cracks, bird droppings, and electrical damage.


# Features

*Image Classification:* Classifies solar panel images into six categories: Clean, Dusty, Bird-Drop, Electrical-Damage, Physical-Damage, and Snow-Covered.
*Object Detection:* Detects and localizes specific defects on the panel surface with bounding boxes.
*Interactive Web Interface:* A user-friendly Streamlit application to upload images and get instant analysis from either model.

# Script Descriptions

  *eda.py:* Performs Exploratory Data Analysis on the classification dataset. It displays sample images from each class and shows the image count per category.

  *train.py:* Trains a CNN classification model (using Transfer Learning with MobileNetV2) on the Solar_Panel_Dataset. It saves the final trained model as solarguard_classifier.h5.

  *train_yolo.py:* Connects to Roboflow to download the annotated dataset and then trains a YOLOv8 object detection model. It saves the best performing model weights (best.pt) in the "runs/detect/train/weights/" directory.

  *app2.py* The main Streamlit web application. It loads both the classifier and detector models, allowing the user to select an analysis type, upload an image, and view the prediction.

  *evaluate.py:* Evaluates the saved classification model (.h5) on a test set, providing metrics like Accuracy, Precision, Recall, and an F1-Score.

  *yoloeval.py:* Evaluates the trained YOLOv8 model (.pt) on the validation set, calculating key object detection metrics such as Precision, Recall, and Mean Average Precision.


This project is for educational purposes as part of the GUVI Artificial Intelligence and Machine Learning program.