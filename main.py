import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('skin_cancer_cnn.h5')



def predict_skin_cancer(image_path, model):
    img = image.load_img(image_path, target_size=(224, 224)) 
    img_array = image.img_to_array(img) / 255.0  
    img_array = np.expand_dims(img_array, axis=0) 

    # Make Prediction
    prediction = model.predict(img_array)
    class_label = "Malignant" if prediction > 0.5 else "Benign"

    return class_label, img


st.title("Skin Cancer Detection")

st.markdown("""
    This is a skin cancer detection application. Upload an image, and the model will predict whether the skin lesion is **Malignant** or **Benign**.
""")


uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:

    class_label, img = predict_skin_cancer(uploaded_image, model)

    st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)  # Show the uploaded image first
    st.write(f"**Prediction: {class_label}**")
