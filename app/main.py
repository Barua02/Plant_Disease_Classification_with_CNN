import os
import json
from PIL import Image

import numpy as np
import tensorflow as tf
import streamlit as st


working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/trained_model/plant_disease_prediction_model.h5"
# Load the pre-trained model
model = tf.keras.models.load_model(model_path)

# loading the class names
class_indices = json.load(open(f"{working_dir}/class_indices.json"))


# Function to Load and Preprocess the Image using Pillow
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    # Load the image
    img = Image.open(image_path)
    # Resize the image
    img = img.resize(target_size)
    # Convert the image to a numpy array
    img_array = np.array(img)
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    # Scale the image values to [0, 1]
    img_array = img_array.astype('float32') / 255.
    return img_array


# Function to Predict the Class of an Image
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name


# Streamlit App with Improved Design
st.set_page_config(page_title="Plant Disease Classifier", page_icon="🌿", layout="wide")

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 12px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .uploadedFile {
        border: 2px dashed #4CAF50;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("🌿 About")
    st.write("This app uses a deep learning model to classify plant diseases from leaf images.")
    st.write("Upload a clear image of a plant leaf for best results.")
    st.markdown("---")
    st.write("**Supported formats:** JPG, JPEG, PNG")
    st.write("**Model:** CNN trained on PlantVillage dataset")

# Main content
st.title('🌿 Plant Disease Classifier')
st.markdown("Upload an image of a plant leaf to detect potential diseases.")

# File uploader with custom styling
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], help="Select a clear image of a plant leaf")

if uploaded_image is not None:
    try:
        image = Image.open(uploaded_image)
        
        # Display image in a nice layout
        st.markdown("### 📷 Uploaded Image")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.image(image, caption="Original Image", use_column_width=True)
        
        with col2:
            st.markdown("### 🔍 Analysis")
            if st.button('🔬 Classify Disease', key='classify'):
                with st.spinner('Analyzing image...'):
                    # Preprocess and predict
                    prediction = predict_image_class(model, uploaded_image, class_indices)
                
                # Display result with styling
                st.success(f"**Predicted Disease:** {prediction}")
                
                # Add confidence if available (assuming top prediction)
                predictions = model.predict(load_and_preprocess_image(uploaded_image))
                confidence = np.max(predictions) * 100
                st.info(f"**Confidence:** {confidence:.2f}%")
                
                # Expandable section for more details
                with st.expander("ℹ️ More Information"):
                    st.write("This prediction is based on a trained CNN model.")
                    st.write("For accurate diagnosis, consult a plant pathologist.")
    
    except Exception as e:
        st.error(f"Error processing image: {str(e)}. Please upload a valid image file.")

else:
    st.info("👆 Upload an image to get started!")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and TensorFlow")