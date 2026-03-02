import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

st.set_page_config(layout="wide")
st.title("🧠 Interactive CNN Study Tool")
st.markdown("Understand Convolution, 32/64 Filters, MaxPooling & Neural Networks")

# ---------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------
st.sidebar.header("Controls")

num_filters = st.sidebar.selectbox("Number of Filters", [1, 8, 16, 32, 64])
kernel_size = st.sidebar.slider("Kernel Size", 3, 7, 3, step=2)
apply_pool = st.sidebar.checkbox("Apply MaxPooling (2x2)")
show_dense = st.sidebar.checkbox("Show Dense Layer Visualization")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

# ---------------------------------------------------
# Convolution Function
# ---------------------------------------------------
def apply_random_filters(image, num_filters, kernel_size):
    feature_maps = []
    for _ in range(num_filters):
        kernel = np.random.randn(kernel_size, kernel_size)
        kernel = kernel / np.sum(np.abs(kernel))  # Normalize
        filtered = cv2.filter2D(image, -1, kernel)
        feature_maps.append(filtered)
    return feature_maps

# ---------------------------------------------------
# MaxPooling Function
# ---------------------------------------------------
def max_pool(image):
    return cv2.resize(image, (image.shape[1]//2, image.shape[0]//2),
                      interpolation=cv2.INTER_NEAREST)

# ---------------------------------------------------
# Display Feature Maps
# ---------------------------------------------------
def display_feature_maps(feature_maps):
    cols = 8
    rows = int(np.ceil(len(feature_maps)/cols))
    fig, axes = plt.subplots(rows, cols, figsize=(15, 2*rows))
    axes = axes.flatten()

    for i in range(len(axes)):
        if i < len(feature_maps):
            axes[i].imshow(feature_maps[i], cmap='gray')
            axes[i].axis("off")
        else:
            axes[i].axis("off")

    st.pyplot(fig)

# ---------------------------------------------------
# Main App Logic
# ---------------------------------------------------
if uploaded_file:
    image = Image.open(uploaded_file)
    image = np.array(image)

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    st.subheader("Original Image")
    st.image(gray, use_column_width=True, clamp=True)

    # -------------------------------
    # Convolution Layer
    # -------------------------------
    st.subheader(f"🔹 Convolution Layer ({num_filters} Filters)")

    feature_maps = apply_random_filters(gray, num_filters, kernel_size)
    display_feature_maps(feature_maps)

    st.markdown(f"""
    - {num_filters} filters applied
    - Kernel size: {kernel_size}x{kernel_size}
    - Each filter produces one feature map
    """)

    # -------------------------------
    # MaxPooling Layer
    # -------------------------------
    if apply_pool:
        st.subheader("🔹 MaxPooling Layer (2x2)")

        pooled_maps = [max_pool(fm) for fm in feature_maps]
        display_feature_maps(pooled_maps)

        st.markdown("""
        - Image size reduced by half
        - Important features retained
        - Reduces computation
        """)

    # -------------------------------
    # Flatten Layer
    # -------------------------------
    st.subheader("🔹 Flatten Layer")

    flattened_size = gray.shape[0] * gray.shape[1]
    st.write(f"Flattened Vector Size: {flattened_size}")

    st.markdown("""
    Converts 2D feature maps into 1D vector for Dense layer.
    """)

    # -------------------------------
    # Dense Layer Visualization
    # -------------------------------
    if show_dense:
        st.subheader("🔹 Dense Neural Network")

        fig, ax = plt.subplots(figsize=(8, 6))

        # Input nodes
        for i in range(10):
            ax.scatter(0, i)

        # Hidden nodes
        for i in range(6):
            ax.scatter(2, i+2)

        # Output nodes
        for i in range(4):
            ax.scatter(4, i+3)

        # Connections
        for i in range(10):
            for j in range(6):
                ax.plot([0,2], [i, j+2], alpha=0.1)

        for i in range(6):
            for j in range(4):
                ax.plot([2,4], [i+2, j+3], alpha=0.1)

        ax.set_title("Simple Dense Neural Network")
        ax.axis("off")

        st.pyplot(fig)

        st.markdown("""
        - Fully connected layer
        - Combines extracted features
        - Final layer usually uses Softmax
        """)

else:
    st.info("Upload an image to start exploring CNN concepts.")