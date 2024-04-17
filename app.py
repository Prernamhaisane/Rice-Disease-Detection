import h5py
import tensorflow as tf
import streamlit as st
from PIL import Image, ImageOps
import numpy as np
from tensorflow.keras.preprocessing import image

filename = r"C:\Users\PRERNA\OneDrive\Desktop\B.TechProject\riceleafdisease1.h5"

# Open the HDF5 file
with h5py.File(filename, 'r') as f:
    # Load the model
    loaded_model = tf.keras.models.load_model(filename)

# Function to predict the label
def predict(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    return predicted_class

# Load the image
def load_and_preprocess_image(image_file):
    img = Image.open(image_file)
    img = img.resize((256, 256))  # Assuming target size is 256x256
    return img

# Define class names
class_names = ['Bacterialblight', 'Blast', 'Brownspot', 'Tungro']

# Define remedies for diseases
remedies = {
    "Bacterialblight": """ 
Chemical Pesticides : Nitrogen Fertilizers, Phosphorus Fertilizers\n 
Bio-pesticides : Bacillus subtilis, Streptomyces spp., Baculovirus  \n 
Botanical Pesticides : Neem Oil, Ginger Extract, Aloe Vera Extract\n
""",
    "Blast": """ 
Chemical Pesticides : Carbendazim 50WP @ 500g/ha\n
Bio-pesticides : Dry seed treatment with Pseudomonas fluorescens talc formulation @10g/kg of seed.\n
Botanical Pesticides : Neem Oil, Garlic Extract, Turmeric Extract\n 
""",
    "Brownspot": """ 
Chemical Pesticides : Spray Mancozeb (2.0g/lit) or Edifenphos (1ml/lit) - 2 to 3 times at 10 - 15 day intervals.\n 
Bio-pesticides : Seed treatment with Pseudomonas fluorescens @ 10g/kg of seed followed by seedling dip\n 
Botanical Pesticides : Neem Oil, Papaya Leaf Extract, Aloe Vera Extract\n
""",
    "Tungro": """ 
Chemical Pesticides : Balanced NPK Fertilizers, Zinc Sulfate\n
Bio-pesticides : Bacillus thuringiensis (Bt), Trichoderma spp.\n 
Botanicals : Neem Oil, Garlic Extract, Neem Cake (Neem Seed Kernel)\n
"""
}

# Function to predict disease
def predict_disease(image, model):
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    return predicted_class, predictions

st.write("""
    # Rice Disease Detection
    """)

file = st.file_uploader("Please upload an image of a rice leaf", type=["jpg", "png"])

if file is None:
    st.text("Please upload an image file")
else:
    image = load_and_preprocess_image(file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    predicted_class, predictions = predict_disease(image, loaded_model)
    confidence = np.max(predictions)

    st.write("Predicted Class:", predicted_class)
    st.write("Confidence:", confidence)
    st.write("Remedies:- ", remedies[predicted_class])  # Display suggestions for predicted disease
