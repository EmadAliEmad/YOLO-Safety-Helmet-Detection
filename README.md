# 👷‍♂️ YOLOv8 Safety Helmet Detection

This repository contains a Computer Vision project focused on detecting safety helmets and people in workplace images. The project utilizes the YOLOv8 object detection model, fine-tuned on a custom dataset.

---

## ✨ Key Features:

*   **End-to-End Pipeline:** Covers dataset preparation, YOLOv8 model fine-tuning, comprehensive evaluation, and model export for deployment.
*   **YOLOv8 Object Detection:** Employs the cutting-edge YOLOv8s model from Ultralytics, fine-tuned for specific object detection (helmet, head, person).
*   **Custom Dataset Integration:** Demonstrates handling and preparing a custom dataset for YOLOv8 training.
*   **Professional Training:** Utilizes efficient training practices within the Ultralytics framework.
*   **Comprehensive Evaluation:** Provides detailed performance metrics (mAP50, Precision, Recall) and generates various visualizations (Confusion Matrix, F1-Curve, Sample Predictions).
*   **Model Export:** Exports the trained model to the ONNX format for wider compatibility and deployment scenarios.
*   **Robustness:** Addresses practical challenges related to environment setup and dependency management in cloud environments like Kaggle.

---

## 📚 Technologies Used:

*   **Object Detection Framework:** `Ultralytics YOLOv8`
*   **Deep Learning Frameworks:** `PyTorch` (underpins Ultralytics)
*   **Data Manipulation:** `Numpy`, `Pandas`
*   **Machine Learning Utilities:** `Scikit-learn`
*   **Computer Vision Utilities:** `Supervision`, `PyCOCOTools`
*   **Model Export:** `ONNX`
*   **Platform:** `Kaggle Notebooks`
*   **Logging & Visualization:** `Loguru`, `Rich`, `Matplotlib`
*   **Environment Management:** `Python venv`
*   **Version Control:** `Git`, `GitHub`

---

## 📂 Project Structure:

This repository contains the Jupyter Notebook source file and the generated output files from the training run.

```bash
.
├── yolo-safety-helmet-detection.ipynb # The Kaggle Notebook source file, containing all steps and outputs.
├── data.yaml                          # YOLOv8 dataset configuration file.
├── requirements.txt                   # List of all Python dependencies for the project.
├── runs/                              # Directory containing all training outputs, including model weights and plots.
│   └── detect/
│       └── train/                     # Specific run directory
│           ├── weights/               # Contains best.pt (PyTorch weights) and best.onnx (exported ONNX model)
│           ├── confusion_matrix.png   # Generated confusion matrix plot
│           ├── results.png            # Overall training results plot
│           ├── val_batch0_pred.jpg    # Example image with model predictions
│           └── ... (other plots and log files)
├── kaggle/                            # (Optional) Kaggle specific files from download.
├── yolov8s.pt                         # Base YOLOv8 model weights (pre-trained, downloaded during setup).
└── README.md                          # This project overview and documentation.


