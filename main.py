import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def load_img(path_to_img):
    max_dim = 512 # Optimized for i7 CPU performance
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)
    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        tensor = tensor[0]
    return Image.fromarray(tensor)

# --- EXECUTION ---
print("--- Initializing Neural Style Transfer ---")

# Load Images (Ensure these files exist in your folder!)
content_image = load_img('content.jpg')
style_image = load_img('style.jpg')

# Load Pre-trained Model from TensorFlow Hub
print("Loading model (Fast Arbitrary Image Stylization)...")
model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

# Run Style Transfer
print("Applying artistic style... please wait.")
stylized_image = model(tf.constant(content_image), tf.constant(style_image))[0]

# Convert to PIL for saving
final_img = tensor_to_image(stylized_image)
final_img.save('stylized_result.jpg')

# --- DISPLAY COMPARISON FOR REPORT ---
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(np.squeeze(content_image))
plt.title('Original Content')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(np.squeeze(style_image))
plt.title('Style Reference')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(final_img)
plt.title('Final Stylized Image')
plt.axis('off')

print("Task Complete! 'stylized_result.jpg' has been saved.")
plt.show()