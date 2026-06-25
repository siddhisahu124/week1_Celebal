import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

from utils import (
    load_model,
    predict_image,
    detect_change,
    generate_heatmap
)

st.set_page_config(
    page_title="Satellite Change Detection",
    layout="wide"
)

st.title(
    "🛰️ Satellite Image Classification & Change Detection"
)

st.markdown(
    """
    Upload two satellite images to:

    - Classify land-use category
    - Detect temporal changes
    - Visualize change heatmaps
    - Compare threshold modes
    """
)

model = load_model(
    "models/best_resnet18.pt"
)

st.sidebar.header(
    "Change Detection Settings"
)

threshold_mode = st.sidebar.selectbox(
    "Threshold Mode",
    [
        "High Recall",
        "Balanced",
        "High Precision"
    ]
)

if threshold_mode == "High Recall":
    threshold = 0.75

elif threshold_mode == "Balanced":
    threshold = 0.6523

else:
    threshold = 0.55

st.sidebar.write(
    f"Current Threshold: {threshold:.4f}"
)

col1, col2 = st.columns(2)

with col1:

    image1_file = st.file_uploader(
        "Upload T1 Image",
        type=["jpg", "jpeg", "png"],
        key="img1"
    )

with col2:

    image2_file = st.file_uploader(
        "Upload T2 Image",
        type=["jpg", "jpeg", "png"],
        key="img2"
    )

if image1_file and image2_file:

    image1 = Image.open(
        image1_file
    )

    image2 = Image.open(
        image2_file
    )

    st.subheader(
        "Uploaded Images"
    )

    c1, c2 = st.columns(2)

    c1.image(
        image1,
        caption="T1 Image"
    )

    c2.image(
        image2,
        caption="T2 Image"
    )

    class1, conf1 = predict_image(
        model,
        image1
    )

    class2, conf2 = predict_image(
        model,
        image2
    )

    similarity, changed = detect_change(
        image1,
        image2,
        threshold
    )

    st.subheader(
        "Prediction Results"
    )

    st.write(
        f"**T1 Class:** {class1}"
    )

    st.write(
        f"**T1 Confidence:** {conf1:.4f}"
    )

    st.write(
        f"**T2 Class:** {class2}"
    )

    st.write(
        f"**T2 Confidence:** {conf2:.4f}"
    )

    st.write(
        f"**Similarity Score:** {similarity:.4f}"
    )

    st.write(
        f"**Threshold Used:** {threshold:.4f}"
    )

    if changed:

        st.error(
            "🚨 CHANGE DETECTED"
        )

    else:

        st.success(
            "✅ NO CHANGE DETECTED"
        )

    heatmap = generate_heatmap(
        image1,
        image2
    )

    st.subheader(
        "Change Heatmap"
    )

    fig, ax = plt.subplots(
        figsize=(5, 5)
    )

    ax.imshow(
        heatmap,
        cmap="inferno"
    )

    ax.axis("off")

    st.pyplot(fig)