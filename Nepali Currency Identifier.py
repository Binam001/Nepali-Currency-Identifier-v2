import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np

# Load the trained model
model = load_model('currency_identifier_model_v2.keras')
class_labels = {0: "5", 1: "10", 2: "20", 3: "50", 4: "100", 5: "500", 6: "1000"}

def predict_image(image, img_height, img_width):
    img = img_to_array(image) / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    return np.argmax(prediction), np.max(prediction)

st.title("Nepali Currency Identifier App")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = load_img(uploaded_file, target_size=(160, 240))

    # Predict
    predicted_class, accuracy = predict_image(image, 160, 240)
    st.write(f"Money Rs. {class_labels[predicted_class]}")
    st.write(f"Prediction Accuracy: {accuracy*100:.2f}%")

    st.image(image, caption='Uploaded Image', use_container_width=True)