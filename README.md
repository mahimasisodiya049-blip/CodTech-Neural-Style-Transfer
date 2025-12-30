# Neural Style Transfer (NST) Tool
### CodTech IT Solutions - Internship Task 3

## 📌 Project Overview
This project implements a **Neural Style Transfer** (NST) model that allows users to apply artistic styles from famous paintings to their personal photographs. Built with Python and TensorFlow, the application features a modern web-based UI for an interactive experience.

## 🚀 Features
* **Web Interface:** Built with Streamlit for easy image uploading and processing.
* **Arbitrary Style Transfer:** Uses a pre-trained Magenta model from TensorFlow Hub, allowing any image to serve as a style reference.
* **CPU Optimized:** Specifically configured to run efficiently on high-performance laptops (like the Lenovo Slim Pad i7).
* **Side-by-Side Comparison:** Instant visualization of the Content, Style, and Resulting images.

## 🛠️ Technology Stack
* **Language:** Python 3.x
* **Deep Learning Framework:** TensorFlow 2.x
* **Model Source:** TensorFlow Hub (Magenta/Arbitrary Image Stylization)
* **UI Framework:** Streamlit
* **Image Processing:** PIL (Pillow), NumPy

## 📂 Project Structure
```text
CodTech_NST/
├── nst_env/            # Virtual Environment
├── app.py              # Main Streamlit UI Application
├── main.py             # Script version for direct execution
├── requirements.txt    # List of dependencies
└── README.md           # Project documentation