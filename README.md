# VGG16-Based Skin Disease Classification

A deep learning model for classifying 17 different skin diseases using a customized VGG16 architecture.

## Author
- **Name:** Thathsara Pramodya Thalangama

## Project Overview
This project implements a skin disease classification system using transfer learning with the VGG16 architecture. The model is designed to classify 17 different skin conditions with high accuracy and recall, making it potentially valuable for medical diagnostic assistance.

## Dataset
The dataset contains images of various skin diseases categorized into 17 classes:
- Actinic
- Atopic
- Benign
- Candidiasis
- Dermatitis
- Dermatofibroma
- Melanocytic
- Melanoma
- Ringworm
- Squamous
- Tinea
- Vascular
- Carcinoma
- Cell
- Keratosis
- Lesion
- Nevus

Original dataset location: Not explicitly specified in the code, but referenced as "Skin Disease.v1i.multiclass.zip"

## Model Architecture
- **Base Model:** Pre-trained VGG16 (ImageNet weights)
- **Transfer Learning Approach:** Feature extractor layers frozen, classifier layers customized
- **Custom Classifier:**
  - Linear(25088, 1024) + ReLU + Dropout(0.5)
  - Linear(1024, 512) + ReLU + Dropout(0.4)
  - Linear(512, 256) + ReLU
  - Linear(256, 17) - Output layer

## Data Preprocessing
1. **Data Splitting:**
   - Original dataset split into training and test sets
   - 10% of training data allocated for validation

2. **Image Transformations:**
   - **Training Data:**
     - Resize to 224×224 (VGG16 input size)
     - Random horizontal flip
     - Color jitter (brightness, saturation, contrast, hue)
     - Normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
   
   - **Validation/Test Data:**
     - Resize to 224×224
     - Normalization (same parameters)

## Training Details
- **Loss Function:** CrossEntropyLoss with class weights to handle imbalance
- **Optimizer:** AdamW with learning rate 0.00001
- **Learning Rate Scheduler:** ReduceLROnPlateau (patience=3)
- **Epochs:** 20
- **Batch Size:** 8
- **Device:** GPU (CUDA) if available, otherwise CPU

## Model Performance
The model achieves impressive performance metrics, particularly with high recall values across all classes:

| Metric | Value |
|--------|-------|
| Macro-averaged F1 Score | 0.92 |
| Overall Accuracy | ~90% |
| Top-1 Accuracy | ~90% |
| Top-5 Accuracy | ~95% |

### Class-Specific Performance

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| Actinic | 0.88 | 1.00 | 0.94 | 51 |
| Atopic | 1.00 | 1.00 | 1.00 | 45 |
| Benign | 1.00 | 1.00 | 1.00 | 56 |
| Candidiasis | 0.98 | 1.00 | 0.99 | 60 |
| Dermatitis | 0.98 | 1.00 | 0.99 | 45 |
| Dermatofibroma | 0.64 | 1.00 | 0.78 | 54 |
| Melanocytic | 0.72 | 1.00 | 0.84 | 39 |
| Melanoma | 0.59 | 1.00 | 0.74 | 44 |
| Ringworm | 0.98 | 1.00 | 0.99 | 60 |
| Squamous | 0.78 | 1.00 | 0.88 | 54 |
| Tinea | 1.00 | 1.00 | 1.00 | 60 |
| Vascular | 0.96 | 1.00 | 0.98 | 51 |
| Carcinoma | 0.75 | 1.00 | 0.86 | 54 |
| Cell | 0.76 | 1.00 | 0.86 | 54 |
| Keratosis | 1.00 | 0.99 | 1.00 | 107 |
| Lesion | 0.96 | 1.00 | 0.98 | 51 |
| Nevus | 0.76 | 1.00 | 0.87 | 39 |
| **Average** | **0.87** | **1.00** | **0.92** | **924** |

## Key Findings
1. The model achieves perfect recall (1.00) for almost all classes, indicating it rarely misses positive cases.
2. Three classes (Atopic, Benign, and Tinea) achieve perfect scores across precision, recall, and F1-score.
3. Areas for improvement remain in precision for certain classes, particularly Dermatofibroma (0.64), Melanoma (0.59), and Melanocytic (0.72).
4. Class imbalance is handled effectively with weighted loss function.

## Setup and Usage
1. Clone this repository
2. Install dependencies:
   ```
   pip install torch torchvision pandas matplotlib scikit-learn tqdm pillow
   ```
3. Download the dataset and extract it to your preferred location
4. Update paths in the code if necessary
5. Run the notebook to train the model or load the pre-trained model for inference

## Future Work
- Improve precision for classes with lower performance (Dermatofibroma, Melanoma)
- Explore additional data augmentation techniques
- Implement explainability methods (like Grad-CAM) to visualize model focus areas
- Develop a user-friendly interface for clinical use

## License
Apache-2.0 license
