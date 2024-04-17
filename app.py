import h5py
import tensorflow as tf
import streamlit as st
from PIL import Image
import numpy as np

# Define the filename
filename = r"C:\Users\PRERNA\OneDrive\Desktop\B.TechProject\riceleafdisease1.h5"

# Load the model
try:
    with h5py.File(filename, 'r') as f:
        loaded_model = tf.keras.models.load_model(filename)
    st.write("Model loaded successfully.")
except FileNotFoundError:
    st.error("Model file not found.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Function to predict the label
def predict(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    return predictions

# Load and preprocess the image
def load_and_preprocess_image(image_file):
    img = Image.open(image_file)
    img = img.resize((256, 256))  # Assuming target size is 256x256
    return img

# Define class names
class_names = ['Bacterialblight', 'Blast', 'Brownspot', 'Tungro']

# Define remedies for diseases
remedies = {
    "Bacterialblight": "Chemical Pesticides: Nitrogen Fertilizers, Phosphorus Fertilizers\nBio-pesticides: Bacillus subtilis, Streptomyces spp., Baculovirus\nBotanical Pesticides: Neem Oil, Ginger Extract, Aloe Vera Extract",
    "Blast": "Chemical Pesticides: Carbendazim 50WP @ 500g/ha\nBio-pesticides: Dry seed treatment with Pseudomonas fluorescens talc formulation @10g/kg of seed.\nBotanical Pesticides: Neem Oil, Garlic Extract, Turmeric Extract",
    "Brownspot": "Chemical Pesticides: Spray Mancozeb (2.0g/lit) or Edifenphos (1ml/lit) - 2 to 3 times at 10 - 15 day intervals.\nBio-pesticides: Seed treatment with Pseudomonas fluorescens @ 10g/kg of seed followed by seedling dip\nBotanical Pesticides: Neem Oil, Papaya Leaf Extract, Aloe Vera Extract",
    "Tungro": "Chemical Pesticides: Balanced NPK Fertilizers, Zinc Sulfate\nBio-pesticides: Bacillus thuringiensis (Bt), Trichoderma spp.\nBotanicals: Neem Oil, Garlic Extract, Neem Cake (Neem Seed Kernel)"
}

# Streamlit app
st.title("Rice Disease Detection")

# File uploader
file = st.file_uploader("Please upload an image of a rice leaf", type=["jpg", "png"])

if file is not None:
    # Load and display the uploaded image
    image = load_and_preprocess_image(file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Make prediction
    predictions = predict(loaded_model, image)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_class_index]
    confidence = predictions[0][predicted_class_index]

    # Display prediction
    st.write("Predicted Class:", predicted_class)
    st.write("Confidence:", confidence)
    st.write("Remedies:", remedies[predicted_class])
