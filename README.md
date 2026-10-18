# MRI Image Quality Assessment Using Machine Learning

A machine learning pipeline for classifying simulated MRI image degradation using quantitative image features and classical ML models.

## Overview

Magnetic Resonance Imaging (MRI) is widely used for non-invasive medical imaging. Image degradation caused by noise, blur, and acquisition-related artifacts can reduce anatomical visibility and affect downstream image analysis.

This project investigates whether handcrafted image features can distinguish between different levels of simulated MRI image degradation using machine learning.

The pipeline includes image preprocessing, controlled degradation, feature extraction, model training, evaluation, and feature-importance analysis.

## Objectives

- Generate controlled MRI image degradation at different severity levels.
- Extract quantitative image-quality features.
- Compare Logistic Regression and Random Forest classifiers.
- Evaluate model performance using accuracy and weighted F1-score.
- Investigate which image features contribute to classification.
- Build a reproducible medical-imaging data science workflow.

## Dataset

The project uses brain MRI images from the publicly available [Brain Tumor MRI Dataset on Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset).

The dataset contains images categorized into glioma, meningioma, pituitary tumor, and no-tumor classes.

The selected images are used as source images for a controlled image-degradation experiment. Tumor categories are not the prediction targets.

### Simulated Quality Labels

| Label | Description |
|---|---|
| 0 — Good | Original image |
| 1 — Moderate | Gaussian noise and mild Gaussian blur |
| 2 — Poor | Stronger Gaussian noise and blur |

These labels are generated programmatically; they are not expert-assigned clinical quality ratings.

## Methodology

1. **Image loading:** Load MRI images as grayscale images using OpenCV.
2. **Image degradation:** Generate moderate and severe degradation using controlled noise and blur.
3. **Feature extraction:** Calculate quantitative image features for each image.
4. **Dataset construction:** Associate each feature vector with its simulated quality label.
5. **Train-test split:** Split by original image identifier so that different versions of the same MRI remain in the same partition.
6. **Model training:** Train Logistic Regression and Random Forest classifiers.
7. **Evaluation:** Compare accuracy and weighted F1-score.
8. **Interpretability:** Examine Random Forest feature importance and visualize classification results.

## Feature Engineering

Twelve numerical features are extracted from each grayscale image.

| Feature | Description |
|---|---|
| Mean intensity | Average normalized pixel intensity |
| Standard deviation | Variation in pixel intensities |
| Median intensity | Median normalized pixel intensity |
| Contrast | Difference between the 95th and 5th intensity percentiles |
| Shannon entropy | Information entropy of the image intensity distribution |
| Sharpness | Variance of the Laplacian |
| Edge density | Fraction of pixels detected as edges |
| Mean gradient | Average gradient magnitude |
| 10th percentile | Lower-tail intensity statistic |
| 25th percentile | Lower-quartile intensity statistic |
| 75th percentile | Upper-quartile intensity statistic |
| 90th percentile | Upper-tail intensity statistic |

These features capture image intensity, contrast, local structure, and high-frequency variation. They are general image descriptors rather than definitive MRI-specific quality metrics.

## Machine Learning Models

### Logistic Regression

Used as the baseline classifier. Standardization is applied within a scikit-learn pipeline.

### Random Forest

Used as a nonlinear classifier to capture relationships between the extracted features and simulated quality labels.

## Results

The following results were obtained in the reported experiment.

| Model | Accuracy | Weighted F1-score |
|---|---:|---:|
| Logistic Regression | 94.67% | 0.9468 |
| Random Forest | 98.67% | 0.9866 |

Random Forest achieved higher accuracy and weighted F1-score than Logistic Regression on this dataset.

These results indicate that the extracted features distinguish the experiment's simulated degradation levels effectively. They do not establish performance on real-world clinical MRI quality assessment.

## Visualizations

The `results/` directory contains the generated experimental outputs:

- `confusion_matrix.png` — classification performance by quality class.
- `feature_importance.png` — relative Random Forest feature importance.
- `model_comparison.png` — accuracy comparison between models.
- `mri_quality_features.csv` — extracted features and associated metadata, if included in the repository.

## Repository Structure

```text
MRI-Image-Quality-Assessment/
├── data/
│   └── README.md
├── notebooks/
│   └── MRI_IQA.ipynb
├── src/
│   ├── feature_extraction.py
│   └── model.py
├── results/
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   ├── model_comparison.png
│   └── mri_quality_features.csv
├── README.md
├── requirements.txt
└── .gitignore
```

The original MRI images are not included in the repository. Follow the dataset provider's terms when downloading or redistributing data.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/MRI-Image-Quality-Assessment.git
cd MRI-Image-Quality-Assessment
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Prepare the dataset

Download the [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) and extract it locally.

Update `DATA_DIR` in the notebook to point to the extracted dataset directory.

### 5. Run the notebook

Open `notebooks/MRI_IQA.ipynb` in Jupyter Notebook or VS Code and execute the cells in order.

The notebook performs image selection, degradation, feature extraction, model training, and evaluation.

## Limitations

- Quality levels are generated using synthetic Gaussian noise and blur.
- Synthetic degradation does not reproduce the full range of real MRI artifacts, such as patient motion, ringing, ghosting, or acquisition-related reconstruction errors.
- Image statistics can depend on anatomy, background pixels, intensity scaling, and preprocessing.
- Feature importance indicates model usage, not causal importance or clinical relevance.
- The experiment has not been clinically validated and should not be used to make diagnostic or patient-care decisions.

## Future Work

- Evaluate on MRI images with expert-assigned quality labels.
- Investigate MRI-specific quality metrics and realistic acquisition artifacts.
- Extend artifact simulation to motion, undersampling, ringing, and intensity non-uniformity.
- Compare handcrafted features with deep-learning approaches.
- Evaluate generalization across MRI sequences, scanners, and acquisition protocols.
- Investigate explainability methods and clinically relevant evaluation criteria.

## Technologies

Python · NumPy · Pandas · OpenCV · scikit-image · scikit-learn · Matplotlib · Seaborn · Jupyter Notebook

## Disclaimer

This project is an educational proof of concept for medical-imaging data science. It does not provide clinical image-quality certification, diagnostic predictions, or medical advice.
