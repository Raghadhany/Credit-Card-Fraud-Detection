# 💳 Credit Card Fraud Detection Using Machine Learning

This project aims to detect **fraudulent credit card transactions** using machine learning techniques. Given the highly imbalanced nature of the dataset, the focus is on building models that can accurately identify rare fraudulent events.

---

## 📂 Dataset

The dataset consists of **284,807 credit card transactions**, with only **492 labeled as fraudulent (0.172%)**. All features are numerical and have been **anonymized via PCA transformation** (except for `Time` and `Amount`).

**Features:**

- `Time`: Seconds elapsed since the first transaction.
- `Amount`: Transaction amount.
- `V1–V28`: PCA-transformed numerical features.
- `Class`: Binary label (0 = Legitimate, 1 = Fraudulent)

---

## 🔍 Project Approach

### 🔬 1. Data Exploration
- Analyze class imbalance
- Visualize feature distributions
- Check correlations and anomalies

### ⚙️ 2. Preprocessing
- Standardize `Amount` and `Time` features
- Split data into **training** and **testing** sets

### 🤖 3. Model Training

Three classifiers were evaluated:

- **Logistic Regression**
- **Random Forest Classifier**
- **Neural Network (MLPClassifier)**

### 📊 4. Evaluation Metrics

Each model is evaluated based on:

- ✅ Accuracy
- 🎯 Precision
- 📉 Recall
- 🧮 F1-Score
- 🧾 Confusion Matrix

---

## 📈 Results Summary

| Model             | Precision | Recall | Notes                                   |
|------------------|-----------|--------|-----------------------------------------|
| Logistic Regression | High      | Low    | Struggles with recall due to imbalance  |
| Random Forest       | Good      | Good   | Better balance between metrics          |
| Neural Network      | Better    | Better | Best performance with deeper learning   |

---

## 🚀 How to Use

### Clone the repository

```bash
git clone https://github.com/yourusername/CreditCardFraudDetection.git
cd CreditCardFraudDetection


### Install dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn  

### Run the Jupyter notebook:

  ```bash
jupyter notebook final_machine.ipynb  

