# Multi-Instance Learning for Whole-Slide Image Classification

A deep learning mini-project that classifies histopathology whole-slide images (WSIs) as **Normal** or **Tumor** using ResNet18 and Attention-Based Multiple Instance Learning (ABMIL).

## Project Overview

Whole-slide histopathology images are extremely large, making direct image processing computationally demanding.

This project uses a patch-based approach. Each slide is represented as a collection of smaller image patches. ResNet18 extracts features from each patch, while an Attention-Based MIL model combines these features to produce a slide-level classification.

## Objectives

- Classify whole-slide images into Normal and Tumor categories.
- Extract meaningful image features using ResNet18.
- Apply attention-based aggregation to patch features.
- Identify patches that contribute more to the final prediction.
- Develop a Streamlit web application for model inference and visualization.

## Dataset Details

**Dataset:** CAMELYON Preprocessed Dataset v4 (Kaggle)

| Description | Details |
|---|---|
| Total slides | 56 |
| Tumor slides | 32 |
| Normal slides | 24 |
| Patches per slide | 300 |
| Total patches | 16,800 |
| Patch dimensions | 256 × 256 |

### Dataset Split

A custom stratified split was created at the WSI level using random state 42.

| Split | Total slides | Normal | Tumor |
|---|---:|---:|---:|
| Training | 39 | 17 | 22 |
| Validation | 8 | 3 | 5 |
| Testing | 9 | 4 | 5 |

**Note:** This is a custom split, not the official CAMELYON16 split.

## Model Architecture

The project uses a two-stage pipeline.

### 1. Feature Extraction – ResNet18

- ImageNet-pretrained ResNet18.
- Frozen encoder weights.
- Final fully connected layer removed.
- Generates 512-dimensional feature vectors for each patch.

### 2. Classification – Attention-Based MIL

- Gated attention mechanism using Tanh and Sigmoid.
- Softmax attention weights.
- Attention-weighted aggregation of patch features.
- Fully connected neural network for final classification.
- Output classes: Normal and Tumor.

### Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 0.0001 |
| Weight decay | 0.001 |
| Dropout | 0.35 |
| Maximum epochs | 100 |
| Early stopping patience | 15 |

## Experimental Results

Evaluation on the held-out test set of 9 slides:

| Metric | Result |
|---|---:|
| Accuracy | 77.78% |
| Balanced Accuracy | 75.00% |
| ROC-AUC | 0.70 |
| Tumor Recall | 100% |

### Confusion Matrix

| Actual / Predicted | Normal | Tumor |
|---|---:|---:|
| Normal | 2 | 2 |
| Tumor | 0 | 5 |

The model correctly classified 7 out of 9 test slides.

**Note:** The test set is small, so these results are preliminary and require further validation.

## Web Application

A local web application was developed using Streamlit.

### Features

- Upload a ZIP file containing image patches.
- Upload the corresponding metadata CSV.
- Select a slide for analysis.
- Predict Normal or Tumor.
- Display tumor probability.
- Visualize the top 10 attention-weighted patches.
- Generate a spatial attention heatmap.
- View and download attention scores as CSV.

## Installation and Execution

### Requirements

- Python 3.10 or compatible version
- PyTorch
- Torchvision
- Streamlit
- NumPy
- Pandas
- Matplotlib
- Pillow

### Setup

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/ABMIL-Whole-Slide-Image-Classification.git
```

Navigate to the project directory:

```bash
cd ABMIL-Whole-Slide-Image-Classification
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

The application will open in your browser, usually at:

```text
http://localhost:8501
```

Make sure both trained model files are present inside the `models/` directory.

## Project Structure

```text
ABMIL-Whole-Slide-Image-Classification/
│
├── app.py
├── model.py
├── requirements.txt
├── README.md
│
└── models/
    ├── abmil_regularized_best.pth
    └── resnet18_encoder.pth
```

## Limitations

- Small dataset and test set.
- No external dataset validation.
- Application currently accepts pre-extracted patch collections rather than raw WSI TIFF files.
- Attention scores indicate model influence, not validated tumor segmentation.
- The model is not clinically validated.

## Future Improvements

- Train and evaluate using larger datasets.
- Explore pathology-specific pretrained encoders.
- Add automated patch extraction from raw WSI files.
- Perform external dataset validation.
- Improve model generalization.

## Disclaimer

This project is developed for academic and research purposes only. It is not intended for clinical diagnosis or medical decision-making.

## Technologies Used

- Python
- PyTorch
- Torchvision
- ResNet18
- Attention-Based Multiple Instance Learning
- Streamlit
- NumPy
- Pandas
- Matplotlib
