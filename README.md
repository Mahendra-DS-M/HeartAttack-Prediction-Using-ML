# ❤️ Heart Disease Attack Prediction

> 🚀 A clean, reproducible ML pipeline to predict the likelihood of a heart attack using patient health metrics.

---

## 📖 Overview
This project predicts **heart disease attack risk** based on patient health data using machine learning techniques.  
It includes data preprocessing, feature engineering, model training, evaluation, and easy inference — perfect for learning or deployment.

---

## ✨ Features
- 🧹 Clean preprocessing pipeline (missing value handling, encoding, scaling)
- 🧠 Model training using Logistic Regression, Random Forest, or XGBoost
- 📊 Performance metrics — ROC AUC, accuracy, precision, recall, F1-score
- 📈 Confusion matrix and classification report visualization
- 💾 Model export (`.pkl` / `.joblib`) for easy deployment
- ⚡ Inference script for predictions on new patient data

---

## 📂 Repository Structure

---

## 🧾 Dataset
You can use the **UCI Heart Disease dataset** or your own CSV.  
Typical columns include:

| Feature | Description |
|----------|-------------|
| age | Age of the patient |
| sex | Gender (1 = male, 0 = female) |
| cp | Chest pain type |
| trestbps | Resting blood pressure |
| chol | Serum cholesterol (mg/dl) |
| fbs | Fasting blood sugar > 120 mg/dl |
| restecg | Resting electrocardiographic results |
| thalach | Maximum heart rate achieved |
| exang | Exercise-induced angina |
| oldpeak | ST depression induced by exercise |
| slope | Slope of the ST segment |
| ca | Number of major vessels (0–3) colored by fluoroscopy |
| thal | Thalassemia type |
| target | 0 = No attack, 1 = Heart attack |

---

## ⚙️ Quick Start

### 1️⃣ Create and activate virtual environment
```bash
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows
