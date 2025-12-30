import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image

# Page Configuration
st.set_page_config(page_title="AI Artistic Style Transfer", layout="wide")

st.title("🎨 Neural Style Transfer Tool")
st.write("Upload a photo and an artwork to blend them together using Deep Learning.")

# Load Model (Cached so it only loads once)
@st.cache_resource
def load_model():
    return hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

model = load_model()

def preprocess_image(image_file):
    img = Image.open(image_file).convert('RGB')
    img = np.array(img)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = img[tf.newaxis, :]
    return tf.image.resize(img, (512, 512)) # Standardized size for i7 performance

def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        tensor = tensor[0]
    return Image.fromarray(tensor)

# UI Layout: Two columns for uploads
col1, col2 = st.columns(2)

with col1:
    content_file = st.file_uploader("Choose Content Image (Your Photo)", type=['jpg', 'png', 'jpeg'])
    if content_file:
        st.image(content_file, caption="Original Photo", use_container_width=True)

with col2:
    style_file = st.file_uploader("Choose Style Image (The Artwork)", type=['jpg', 'png', 'jpeg'])
    if style_file:
        st.image(style_file, caption="Artistic Style", use_container_width=True)

# Action Button
if content_file and style_file:
    if st.button("✨ Generate Artistic Masterpiece"):
        with st.spinner("Applying AI Magic... This takes about 5-10 seconds on your i7."):
            
            # Process images
            content_img = preprocess_image(content_file)
            style_img = preprocess_image(style_file)
            
            # Run Model
            outputs = model(tf.constant(content_img), tf.constant(style_img))
            stylized_image = outputs[0]
            
            # Convert and Display
            final_img = tensor_to_image(stylized_image)
            
            st.divider()
            st.subheader("Final Result")
            st.image(final_img, caption="Stylized Image", use_container_width=True)
            
            # Download Button
            st.download_button(label="Download Image", 
                               data=open(content_file.name, "rb"), # Placeholder for logic
                               file_name="stylized_art.jpg", 
                               mime="image/jpeg")
else:
    st.info("Please upload both images to proceed.")