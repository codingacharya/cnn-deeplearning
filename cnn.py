import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

st.set_page_config(layout="wide")
st.title("🧠 CNN Filter Visualizer (Interactive)")

# ----------------------------------
# Sidebar Options
# ----------------------------------
st.sidebar.header("Filter Options")

filter_option = st.sidebar.selectbox(
    "Select Filter Type",
    [
        "Horizontal Edge",
        "Vertical Edge",
        "Diagonal Edge",
        "Blur",
        "Sharpen",
        "Emboss",
        "Box Blur",
        "Custom"
    ]
)

kernel_size = st.sidebar.slider("Kernel Size", 3, 7, 3, step=2)

# ----------------------------------
# Upload Image
# ----------------------------------
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    image = np.array(image)

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # ----------------------------------
    # Define Filters
    # ----------------------------------
    if filter_option == "Horizontal Edge":
        kernel = np.array([[-1, -1, -1],
                           [0, 0, 0],
                           [1, 1, 1]])

    elif filter_option == "Vertical Edge":
        kernel = np.array([[-1, 0, 1],
                           [-1, 0, 1],
                           [-1, 0, 1]])

    elif filter_option == "Diagonal Edge":
        kernel = np.array([[0, 1, 1],
                           [-1, 0, 1],
                           [-1, -1, 0]])

    elif filter_option == "Blur":
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)

    elif filter_option == "Box Blur":
        kernel = np.ones((3, 3)) / 9

    elif filter_option == "Sharpen":
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])

    elif filter_option == "Emboss":
        kernel = np.array([[-2, -1, 0],
                           [-1, 1, 1],
                           [0, 1, 2]])

    elif filter_option == "Custom":
        st.sidebar.write("Enter 3x3 Custom Kernel")
        kernel = np.zeros((3,3))
        for i in range(3):
            for j in range(3):
                kernel[i][j] = st.sidebar.number_input(f"K[{i}][{j}]", value=0.0)

    # ----------------------------------
    # Apply Convolution
    # ----------------------------------
    filtered = cv2.filter2D(gray, -1, kernel)

    # ----------------------------------
    # Display Results
    # ----------------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        plt.figure(figsize=(5,5))
        plt.imshow(gray, cmap="gray")
        plt.axis("off")
        st.pyplot(plt)

    with col2:
        st.subheader(f"{filter_option} Applied")
        plt.figure(figsize=(5,5))
        plt.imshow(filtered, cmap="gray")
        plt.axis("off")
        st.pyplot(plt)

    # ----------------------------------
    # Show Kernel Matrix
    # ----------------------------------
    st.subheader("Filter (Kernel) Matrix")
    st.write(kernel)

    st.markdown("""
    ### 🧠 What Happened?
    - The kernel slid across the image.
    - Each 3x3 region was multiplied with the filter.
    - Values were summed to produce the output pixel.
    - This created a feature map.
    """)

else:
    st.info("Please upload an image to start.")
