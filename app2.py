import streamlit as st
import tensorflow as tf
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

#MODEL LOADING

# Cache the TensorFlow model for classification
@st.cache_resource
def load_classification_model():
    try:
        model = tf.keras.models.load_model("solarguard_classifier.h5")
        return model
    except Exception as e:
        st.error(f"Error loading classification model: {e}", icon="🚨")
        return None
                                                                                                            
# Cache the YOLOv8 model for object detection                                                                                                           
@st.cache_resource                                                                                                          
def load_detection_model():                                                                                                         
    # IMPORTANT: Update this path to your best.pt file                                                                                                          
    model_path = r"C:\Users\rahen\Documents\GUVI\SolarPanel\runs\detect\train\weights\best.pt"                                                                                                            
    try:                                                                                                            
        model = YOLO(model_path)                                                                                                            
        return model                                                                                                            
    except Exception as e:                                                                                                          
        st.error(f"Error loading object detection model: {e}", icon="🚨")                                                                                                           
        return None                                                                                                         
                                                                                                            
# Load both models                                                                                                          
classifier_model = load_classification_model()                                                                                                          
detector_model = load_detection_model()                                                                                                         
                                                                                                            
# Class names for the classifier (ensure this order is correct)                                                                                                         
CLASS_NAMES = ['Bird-Drop', 'Clean', 'Dusty', 'Electrical-Damage', 'Physical-Damage', 'Snow-Covered']                                                                                                           
                                                                                                            
                                                                                                            
#STREAMLIT APP LAYOUT                                                                                                        
                                                                                                            
st.set_page_config(page_title="SolarGuard AI", page_icon="☀️", layout="wide")                                                                                                           
st.title("☀️ SolarGuard: AI Defect Analysis")                                                                                                           
st.write("Choose an analysis type from the sidebar and upload a solar panel image.")                                                                                                            
                                                                                                            
# Sidebar for selecting the analysis type                                                                                                           
st.sidebar.title("Analysis Options")                                                                                                            
analysis_type = st.sidebar.radio(                                                                                                           
    "Select Analysis Type:",                                                                                                            
    ("Image Classification", "Object Detection")                                                                                                            
)                                                                                                           
                                                                                                            
# File uploader                                                                                                         
uploaded_file = st.file_uploader("Upload a Solar Panel Image", type=["jpg", "jpeg", "png"])                                                                                                         
                                                                                                            
                                                                                                            
#MAIN LOGIC
                                                                                                            
if uploaded_file is not None:                                                                                                           
    # Display the uploaded image                                                                                                            
    image = Image.open(uploaded_file)                                                                                                           
    col1, col2 = st.columns(2)                                                                                                          
    with col1:                                                                                                          
        st.image(image, caption="Uploaded Image", use_column_width=True)                                                                                                            
                                                                                                            
    # Perform analysis based on user's choice                                                                                                           
    with col2:                                                                                                          
        if analysis_type == "Image Classification":                                                                                                         
            st.subheader("Classification Result")                                                                                                           
            if classifier_model is None:                                                                                                            
                st.error("Classification model is not available.")                                                                                                          
            else:                                                                                                           
                with st.spinner("Classifying..."):                                                                                                          
                    # Preprocess for Keras model                                                                                                            
                    image_resized = image.resize((224, 224))                                                                                                            
                    image_array = np.array(image_resized)                                                                                                           
                    image_normalized = image_array / 255.0                                                                                                          
                    image_batch = np.expand_dims(image_normalized, 0)                                                                                                           
                                                                                                                                
                    # Make prediction                                                                                                           
                    prediction = classifier_model.predict(image_batch)                                                                                                          
                    predicted_class_index = np.argmax(prediction)                                                                                                           
                    predicted_class_name = CLASS_NAMES[predicted_class_index]                                                                                                           
                    confidence = np.max(prediction) * 100                                                                                                           
                                                                                                                                
                    # Display result                                                                                                            
                    st.success(f"**Prediction:** {predicted_class_name}")                                                                                                           
                    st.info(f"**Confidence:** {confidence:.2f}%")                                                                                                           
                                                                                                            
        elif analysis_type == "Object Detection":                                                                                                           
            st.subheader("Object Detection Result")                                                                                                         
            if detector_model is None:                                                                                                          
                st.error("Object detection model is not available.")                                                                                                            
            else:                                                                                                           
                with st.spinner("Detecting defects..."):                                                                                                            
                    # Convert PIL image to OpenCV format                                                                                                            
                    img_cv = np.array(image)                                                                                                            
                    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)                                                                                                            
                                                                                                                                
                    # Make prediction                                                                                                           
                    results = detector_model(img_cv)                                                                                                            
                                                                                                                                
                    # Draw bounding boxes on the image                                                                                                          
                    annotated_img = results[0].plot()                                                                                                           
                    annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)                                                                                                          
                                                                                                                                
                    # Display result                                                                                                            
                    st.image(annotated_img_rgb, caption="Detected Defects", use_column_width=True)                                                                                                          
                                                                                                            
                    # Display detection summary                                                                                                         
                    if len(results[0].boxes) > 0:                                                                                                           
                        st.write("### Detection Summary:")                                                                                                          
                        for box in results[0].boxes:                                                                                                            
                            class_name = detector_model.names[int(box.cls)]                                                                                                         
                            confidence = box.conf.item() * 100                                                                                                          
                            st.success(f"Found **{class_name}** with **{confidence:.2f}%** confidence.")                                                                                                            
                    else:                                                                                                           
                        st.info("No defects were detected.")                                                                                                            