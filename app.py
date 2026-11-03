import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

st.title("Skin Cancer Detection App")
st.write("Apni skin ki image upload karein taake model check kar sakay.")

MODEL_PATH = 'skin_cancer_detection_model.h5'

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        # Aap ki Google Drive file ka link/ID
        file_id = '1_Zn8TCgtq2ZGBAmk0RoRJDyMSh374gYE'
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, MODEL_PATH, quiet=False)
        
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

try:
    model = load_model()
except Exception as e:
    st.warning(f"Model load hone mein masla hai: {e}")
    model = None

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    
    if model is not None:
        st.write("Classification chal rahi hai...")
        
        image = image.resize((128, 128))
        img_array = np.array(image)
        
        if img_array.shape[-1] == 4:
            img_array = img_array[..., :3]
            
        img_array = np.expand_dims(img_array, axis=0)
        
        predictions = model.predict(img_array)
        class_names = ['Benign (Non-Cancerous)', 'Malignant (Cancerous)']
        
        predicted_class = class_names[np.argmax(predictions)]
        confidence = 100 * np.max(predictions)
        
        st.success(f"Result: **{predicted_class}** (Confidence: {confidence:.2f}%)")
    else:
        st.error("Model load nahi ho saka.")
