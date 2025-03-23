# 🧠 A Comparison of Machine Learning Object Classification Approaches

## 📌 Overview

This project evaluates and compares three distinct machine learning approaches for **image classification** using the **Caltech-256 dataset**. The models tested are:

- ✅ **EfficientNet-B0 (CNN)**  
- ✅ **Support Vector Machine (SVM)** with **Bag of Features** and **SURF/SIFT descriptors**  
- ✅ **Bernoulli Naive Bayes** on simplified pixel/feature representations

> Conducted as part of a group assignment for a university course on applied machine learning.

---

## 👥 Authors

**Alex Conroy**, Jordan McCallum, Nebojsa Ajdarevic  
Group project (QUT)

**Alex’s contribution:**  
- Research and model validation  
- Troubleshooting model performance  
- Writing and editing of the final report

---

## 🧪 Objectives

The goal of this project was to explore:
- How different ML approaches handle image classification
- Tradeoffs between **deep learning**, **traditional computer vision**, and **naive statistical models**
- Reproducibility and interpretability of each pipeline

---

## 🗂 Project Structure
assignment2-object-recognition/ ├── data/ # Placeholder – Caltech-256 images ├── notebooks/ │ ├── Assignment2Training.ipynb # Main training & testing logic │ └── Bernoulli-Naive-Bayes.ipynb # Additional model comparison ├── report/ │ ├── Proposal.pdf │ └── Report-v4.docx # Final group report ├── images/ │ └── image-7.png # EfficientNet architecture ├── src/ │ └── model_pipeline.py # Placeholder for reusable ML logic ├── requirements.txt # Required Python packages └── README.md


---

## 🧠 Models Compared

| Model                   | Technique                         | Feature Type          | Accuracy | Pros                        | Cons                       |
|------------------------|-----------------------------------|------------------------|----------|-----------------------------|----------------------------|
| **EfficientNet-B0**    | CNN (Transfer Learning)           | Full Image (Tensor)    | High     | Powerful, automated feature learning | Requires GPU, longer training |
| **SVM + SURF/SIFT**    | Traditional CV + Bag of Features  | Keypoint descriptors   | Medium   | Interpretability, compact   | Requires tuning, not end-to-end |
| **Bernoulli NB**       | Statistical Classifier             | Binary input features  | Low      | Fast, simple                | Low performance, crude features |

---

## 🧪 Dataset

- **Caltech-256:**  
  30,000+ images across 256 categories.  
  [Link to download](https://data.caltech.edu/records/mzrjq-6wc02)

*Dataset not included due to size. Please download manually and extract into `./data/`.*

---

## 🔬 Future Work
Wrap pipelines into reusable functions/modules in src/

Explore additional model architectures (e.g. ResNet, ViT)

Test on more balanced or augmented datasets

Improve reporting and deployment readiness

---

## 📄 License
This project was developed as part of an academic course.
Caltech-256 dataset is publicly available for research purposes.

---

## 💡 Why This Project?
This project demonstrates:
- Clear ML modeling tradeoffs
- A working understanding of deep learning vs. classical CV
- Group-based scientific investigation
- Strong documentation and reproducibility




