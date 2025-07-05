import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

import base64
# Function to load and encode local jpg image
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Local image filename (same folder)
image_file = 'aaa.jpg'

# Get base64 string
img_base64 = get_base64_of_bin_file(image_file)

# Inject HTML + CSS for background
page_bg_img = f"""
<style>
.stApp {{
  background-image: url("data:image/jpg;base64,{img_base64}");
  background-size: cover;
  background-repeat: no-repeat;
  background-attachment: fixed;
}}
</style>
"""

# Load CSS
st.markdown(page_bg_img, unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Project Introduction", "Predicted Tumor Type", "Creator Info"])

# -------------------------------- PAGE 1: Introduction --------------------------------
if page == "Project Introduction":
    st.title("🧠 Brain Tumor MRI Image Classification  ")
    st.write("\n")
    st.write("\n")
    st.write(""" 
    ##### This project aims to develop a deep learning-based solution for classifying brain MRI 
    ##### images into multiple categories according to tumor type. It involves building a custom 
    ##### CNN model from scratch and enhancing performance through transfer learning using 
    ##### pretrained models. The project also includes deploying a user-friendly Streamlit web 
    ##### application to enable real-time tumor type predictions from uploaded MRI images. """)
    st.write("\n")
    st.write("\n")

    st.markdown("""\n
    ### Real-time Business Use Cases:  \n
        ● AI-Assisted Medical Diagnosis \n
        ● Early Detection and Patient Triage \n
        ● Research and Clinical Trials \n
        ● Second-Opinion AI Systems \n  """)
    
    st.markdown("""
     ### Problem Domain: \n
        ● Medical Imaging — Image Classification """)

# -------------------------------- PAGE 2: Predicted Tumor Type --------------------------------
elif page == "Predicted Tumor Type":
    
    # Load model
    model = tf.keras.models.load_model("mobilenetv2_final.h5")
    
    # Class names
    class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
    
    st.title("🧠 Brain Tumor MRI Classifier")
    st.write("Upload an MRI image to classify the tumor type.")
    
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)
    
        # Preprocessing
        img = image.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
    
        # Predict
        prediction = model.predict(img_array)[0]
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction)
    
        st.markdown(f"### 🧠 Tumor Type: **{predicted_class.upper()}**")
        st.markdown(f"### 📊 Confidence: **{confidence * 100:.2f}%**")

# -------------------------------- PAGE 3: Creator Info --------------------------------

elif page == "Creator Info":
    st.title("👨‍💻 Creator of this Project")
    st.write("""
#    **Developed by:** Sudharsan M S
#    **Skills:** 
## Python,
## Deep Learning,
## Transfer Learning,
## Model Evaluation,
## Streamlit
    """)
    st.image('saa.jpg', width=150)
