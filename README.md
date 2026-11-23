# 🔍 Image Forgery & Tampering Detection System

An AI-powered image forensics application that analyzes digital images to detect possible forgery and manipulation using Deep Learning.

The project combines **ResNet18 for image classification** and **U-Net for tampering localization** to identify potentially manipulated images and highlight suspicious regions.

## 🚀 Project Overview

Digital image manipulation has become increasingly common. This project explores how Deep Learning can help identify potentially forged or tampered images.

Users can upload an image through a web interface and receive:

- Image authenticity classification.
- Authentic and tampered class probabilities.
- Predicted tampering mask.
- Highlighted image showing possible manipulated regions.
- Downloadable prediction results.

## ✨ Features

- Image upload through a Streamlit web application.
- Authentic vs. Tampered image classification.
- ResNet18-based image classification.
- U-Net-based tampering localization.
- Prediction probability visualization using a donut chart.
- Original image, predicted mask and overlay comparison.
- Downloadable localization results.
- Image preprocessing and evaluation pipeline.

## 🧠 Model Architecture

### 1. ResNet18 – Image Classification

ResNet18 is used to classify uploaded images into two categories:

- Authentic
- Tampered

The model uses transfer learning with a pretrained ResNet18 architecture.

**Classification output:**
- Predicted class.
- Confidence score.
- Class probabilities.

### 2. U-Net – Tampering Localization

U-Net is used to generate a predicted mask highlighting potentially manipulated regions in an image.

The predicted mask is overlaid on the original image to make the model's localization output easier to inspect.

**Localization output:**
- Predicted binary mask.
- Highlighted image overlay.

Note: The localization model is experimental and its predictions may not accurately identify the actual manipulated regions.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| PyTorch | Deep Learning framework |
| Torchvision | Pretrained ResNet18 |
| ResNet18 | Image classification |
| U-Net | Tampering localization |
| OpenCV | Image processing |
| NumPy | Numerical operations |
| Pandas | Dataset handling |
| Matplotlib | Visualization |
| Scikit-learn | Model evaluation |
| Streamlit | Web application |
| Pillow | Image processing |

## 📂 Project Structure

```text
Image-Forgery-Tampering-Detection/
│
├── app.py
│
├── src/
│   ├── data_loader.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict_forgery.py
│   ├── segmentation_model.py
│   ├── segmentation_data_loader.py
│   ├── train_segmentation.py
│   ├── evaluate_segmentation.py
│   ├── prepare_segmentation_data.py
│   └── evaluation/
│
├── requirements.txt
│
├── .gitignore
│
└── README.md
```

## 📊 Model Evaluation Results

### ResNet18 Classification

Evaluation on the held-out test dataset:

| Metric | Result |
|---|---:|
| Test Images | 924 |
| Accuracy | 84.74% |
| Precision | 97.72% |
| Recall | 83.62% |
| F1-Score | 90.12% |

### U-Net Segmentation

Evaluation on the held-out test dataset:

| Metric | Result |
|---|---:|
| Test Images | 768 |
| Dice Score | 0.1891 |
| IoU Score | 0.1307 |
| Pixel Precision | 0.1796 |
| Pixel Recall | 0.5923 |
| Pixel Accuracy | 0.7017 |

**Evaluation note:** Classification and localization are separate tasks. The classification model achieved 84.74% test accuracy, while the U-Net localization performance remains limited and requires further improvement.

## ⚙️ Installation and Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/Sujata787/image-forgery-tampering-detection.git
```

### Step 2: Navigate to the Project

```bash
cd image-forgery-tampering-detection
```

### Step 3: Create a Virtual Environment

```bash
python -m venv venv
```

### Step 4: Activate the Virtual Environment

For Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

## 📦 Model Files and Dataset

The trained model checkpoints and dataset are not included in this GitHub repository because of their size.

To run the application, place your trained model files in the following locations:

```text
artifacts/
├── best_resnet18.pth
└── best_unet.pth
```

The dataset used for training and evaluation must also be placed in the expected project data directory.

## ▶️ Run the Application

After installing dependencies and placing the trained model checkpoints in the correct locations, run:

```bash
streamlit run app.py
```

The application will open in your browser.

Upload an image and click **Analyze Image** to view the classification and localization results.

## 🔬 Project Workflow

1. Dataset collection and verification.
2. Image preprocessing and augmentation.
3. Dataset splitting into training, validation and testing sets.
4. ResNet18 model training for image classification.
5. U-Net model training for tampering localization.
6. Model evaluation.
7. Single-image prediction.
8. Integration of trained models with Streamlit.
9. Visualization and download of prediction results.

## 🔮 Future Improvements

- Improve U-Net segmentation performance.
- Reduce false-positive and false-negative classification predictions.
- Experiment with additional image forensic techniques.
- Improve localization accuracy.
- Add more robust evaluation and visualization features.
- Explore deployment using Docker and cloud platforms.

## ⚠️ Disclaimer

This project is developed for educational and research purposes.

Model predictions are not definitive proof of image authenticity or manipulation. The application should not be used as a substitute for professional digital forensic examination.

## 👩‍💻 Author

**Sujata Sunkewar**

B.Tech – Data Science

GitHub: [Sujata787](https://github.com/Sujata787)

---

⭐ If you find this project interesting, consider giving the repository a star!