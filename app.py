
import os
import zipfile
import tempfile

import streamlit as st
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

from PIL import Image
from model import load_models


# -----------------------------
# PAGE CONFIGURATION
# -----------------------------

st.set_page_config(
    page_title="ABMIL WSI Classification",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Multi-Instance Learning for Whole-Slide Image Classification")

st.write(
    "Upload a ZIP file containing patch images and its corresponding "
    "metadata CSV to classify a whole-slide image."
)


# -----------------------------
# DEVICE AND MODEL LOADING
# -----------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_DIR = "models"

@st.cache_resource
def get_models():
    return load_models(MODEL_DIR, device)


try:
    encoder, mil_model, transform = get_models()
    st.sidebar.success(f"Models loaded | Device: {device}")
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()


# -----------------------------
# PATCH PROCESSING FUNCTION
# -----------------------------

def extract_and_process(zip_file, csv_file):

    with tempfile.TemporaryDirectory() as temp_dir:

        # Extract uploaded ZIP
        zip_path = os.path.join(temp_dir, "patches.zip")

        with open(zip_path, "wb") as f:
            f.write(zip_file.getbuffer())

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(temp_dir)

        # Read metadata
        metadata = pd.read_csv(csv_file)

        required_columns = [
            "patch_path",
            "wsi_id",
            "x",
            "y"
        ]

        for col in required_columns:
            if col not in metadata.columns:
                st.error(f"Missing required column: {col}")
                return None

        # Find extracted patch images
        patch_lookup = {}

        for root, dirs, files in os.walk(temp_dir):
            for filename in files:

                if filename.lower().endswith(
                    (".png", ".jpg", ".jpeg")
                ):
                    patch_lookup[filename] = os.path.join(
                        root, filename
                    )

        metadata["filename"] = metadata["patch_path"].apply(
            lambda x: os.path.basename(str(x))
        )

        available_rows = metadata[
            metadata["filename"].isin(patch_lookup.keys())
        ].copy()

        if available_rows.empty:
            st.error("No matching patch images found.")
            return None

        # Select slide
        slide_ids = sorted(available_rows["wsi_id"].unique())

        selected_wsi = st.selectbox(
            "Select a slide",
            slide_ids
        )

        slide_data = available_rows[
            available_rows["wsi_id"] == selected_wsi
        ].copy().reset_index(drop=True)

        if slide_data.empty:
            st.error("No patches found for selected slide.")
            return None

        st.info(
            f"Processing {len(slide_data)} patches from {selected_wsi}"
        )

        # Load and transform patches
        images = []

        for filename in slide_data["filename"]:

            image_path = patch_lookup[filename]

            with Image.open(image_path) as img:
                img = img.convert("RGB")
                images.append(transform(img))

        # Extract ResNet18 features
        batch_size = 32
        feature_batches = []

        progress_bar = st.progress(0)
        total_batches = (len(images) + batch_size - 1) // batch_size

        encoder.eval()
        mil_model.eval()

        with torch.no_grad():

            for batch_number, i in enumerate(
                range(0, len(images), batch_size)
            ):

                batch = torch.stack(
                    images[i:i + batch_size]
                ).to(device)

                features = encoder(batch)

                feature_batches.append(features.cpu())

                progress_bar.progress(
                    (batch_number + 1) / total_batches
                )

        progress_bar.empty()

        slide_features = torch.cat(
            feature_batches,
            dim=0
        ).to(device)

        # MIL prediction and attention
        with torch.no_grad():

            logits, attention = mil_model(slide_features)

            probabilities = torch.softmax(logits, dim=0)

            predicted_class = int(
                torch.argmax(probabilities).item()
            )

            tumor_probability = float(
                probabilities[1].item()
            )

        slide_data["attention"] = (
            attention.cpu().numpy().flatten()
        )

        # Copy top 10 patch images into memory
        top_rows = slide_data.nlargest(
            10, "attention"
        )

        top_patch_images = []

        for _, row in top_rows.iterrows():

            filename = row["filename"]

            with Image.open(patch_lookup[filename]) as img:

                top_patch_images.append({
                    "image": img.convert("RGB").copy(),
                    "attention": float(row["attention"]),
                    "filename": filename
                })

        return (
            slide_data,
            predicted_class,
            tumor_probability,
            top_patch_images
        )


# -----------------------------
# SIDEBAR UPLOADS
# -----------------------------

st.sidebar.header("Upload Dataset")

zip_file = st.sidebar.file_uploader(
    "Upload patch ZIP file",
    type=["zip"]
)

csv_file = st.sidebar.file_uploader(
    "Upload patch metadata CSV",
    type=["csv"]
)


# -----------------------------
# ANALYZE BUTTON
# -----------------------------

if zip_file is not None and csv_file is not None:

    if st.button("Analyze Slide", type="primary"):

        with st.spinner("Analyzing slide..."):

            result = extract_and_process(
                zip_file,
                csv_file
            )

            if result is not None:

                (
                    slide_data,
                    predicted_class,
                    tumor_probability,
                    top_patch_images
                ) = result

                st.session_state["slide_data"] = slide_data
                st.session_state["predicted_class"] = predicted_class
                st.session_state["tumor_probability"] = tumor_probability
                st.session_state["top_patch_images"] = top_patch_images

                st.session_state["analysis_complete"] = True

else:
    st.info(
        "Please upload both a patch ZIP file and a metadata CSV."
    )


# -----------------------------
# DISPLAY RESULTS
# -----------------------------

if st.session_state.get("analysis_complete", False):

    slide_data = st.session_state["slide_data"]

    predicted_class = st.session_state["predicted_class"]

    tumor_probability = st.session_state["tumor_probability"]

    top_patch_images = st.session_state["top_patch_images"]

    selected_wsi = slide_data["wsi_id"].iloc[0]

    st.divider()

    st.header("Classification Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Selected Slide",
            selected_wsi
        )

    with col2:
        st.metric(
            "Predicted Class",
            "Tumor" if predicted_class == 1 else "Normal"
        )

    with col3:
        st.metric(
            "Tumor Probability",
            f"{tumor_probability * 100:.2f}%"
        )

    # -------------------------
    # TOP ATTENDED PATCHES
    # -------------------------

    st.divider()

    st.header("Top 10 Most Attended Patches")

    st.write(
        "These patches received the highest attention weights "
        "from the MIL model."
    )

    cols = st.columns(5)

    for i, item in enumerate(top_patch_images):

        with cols[i % 5]:

            st.image(
                item["image"],
                caption=f"Attention: {item['attention']:.4f}",
                use_container_width=True
            )

            st.caption(item["filename"])

    # -------------------------
    # ATTENTION HEATMAP
    # -------------------------

    st.divider()

    st.header("Spatial Attention Heatmap")

    st.write(
        "Patch coordinates are used to visualize the spatial "
        "distribution of attention across the sampled slide patches."
    )

    fig, ax = plt.subplots(figsize=(10, 7))

    scatter = ax.scatter(
        slide_data["x"],
        slide_data["y"],
        c=slide_data["attention"],
        cmap="hot",
        s=35
    )

    plt.colorbar(
        scatter,
        ax=ax,
        label="Attention Weight"
    )

    ax.set_title(
        f"Attention Heatmap - {selected_wsi}"
    )

    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")

    ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="datalim")

    st.pyplot(fig)

    plt.close(fig)

    # -------------------------
    # ATTENTION TABLE
    # -------------------------

    st.divider()

    st.header("Top Attention Scores")

    top_table = slide_data.nlargest(
        10, "attention"
    )[
        ["filename", "x", "y", "attention"]
    ].reset_index(drop=True)

    st.dataframe(
        top_table,
        use_container_width=True
    )

    # -------------------------
    # DOWNLOAD RESULTS
    # -------------------------

    st.divider()

    st.header("Download Results")

    csv_data = slide_data.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="Download Attention Results CSV",
        data=csv_data,
        file_name=f"{selected_wsi}_attention_results.csv",
        mime="text/csv"
    )

    st.success("Slide analysis completed successfully!")
