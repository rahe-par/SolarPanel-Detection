import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os

# --- 1. SET PAGE CONFIG ---
st.set_page_config(
    page_title="SolarGuard | Defect Detection",
    page_icon="☀️",
    layout="wide"
)

# --- 2. Constants ---
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)

MODEL_PATH ="Model\SolarPanel_Project_Final\solarguard_champion_model.h5"

CLASS_NAMES = [
    'Bird-drop', 
    'Clean', 
    'Dusty', 
    'Electrical-damage', 
    'Physical-Damage', 
    'Snow-Covered'
]

# --- 3. Load The Model ---
@st.cache_resource
def load_my_model(model_path):
    """Loads the pre-trained Keras model."""
    if not os.path.exists(model_path):
        st.error(f"Model file not found at: {model_path}")
        st.error(f"Please make sure '{MODEL_PATH}' is in the same directory as this app.")
        st.stop()
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

model = load_my_model(MODEL_PATH)

# --- 4. Preprocess Image ---
def preprocess_image(image_pil):
    """
    Preprocesses the uploaded PIL image to fit model input requirements.
    
    CRITICAL: This function MUST match your training script.
    1. Resize to (224, 224)
    2. Convert to NumPy array
    3. Rescale pixels (divide by 255.0)
    4. Add batch dimension
    """
    # 1. Resize the image
    image = ImageOps.fit(image_pil, IMG_SIZE, Image.Resampling.LANCZOS)
    
    # 2. Convert image to numpy array
    img_array = np.asarray(image)
    
    # 3. Rescale the image (this matches your 'rescale=1./255')
    img_array_rescaled = img_array / 255.0
    
    # 4. Add batch dimension
    img_array_expanded = np.expand_dims(img_array_rescaled, axis=0)
    
    return img_array_expanded

# --- 5. Streamlit App UI ---
st.title("☀️ SolarGuard: Defect Detection")

st.write("Upload an image of a solar panel to classify its condition.")

# File Uploader
uploaded_file = st.file_uploader("Choose a solar panel image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns([0.6, 0.4]) 
    
    with col1:
        st.image(image, caption='Uploaded Image', use_column_width=True)
    
    with st.spinner('Analyzing image...'):
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
    
    predicted_class_index = np.argmax(prediction[0])
    predicted_class_name = CLASS_NAMES[predicted_class_index]
    confidence = 100 * np.max(prediction[0])
    
    with col2:
        st.subheader("Analysis Complete")
        
        st.write(f"**Predicted Condition:**")
        if predicted_class_name == 'Clean':
            st.success(f"**{predicted_class_name}**")
        elif 'Damage' in predicted_class_name:
            st.error(f"**{predicted_class_name}**")
        else:
            st.warning(f"**{predicted_class_name}**")
        
        st.write(f"**Confidence:**")
        st.info(f"**{confidence:.2f}%**")

        st.subheader("Full Prediction Probabilities:")
        st.dataframe(
            {"Class": CLASS_NAMES, "Probability": [f"{p*100:.2f}%" for p in prediction[0]]},
            use_container_width=True
        )
else:
    st.info("Please upload an image file to get started.")