# Rock vs Mine Prediction using Logistic Regression

This machine learning project uses sonar signal data to predict whether an object is a **rock** or a **mine** based on reflected signals. It applies a **Logistic Regression** model to perform binary classification.

---

## 📊 Dataset

- **Source**: [UCI Machine Learning Repository - Sonar Dataset](https://archive.ics.uci.edu/ml/datasets/connectionist+bench+(sonar,+mines+vs.+rocks))
- **Description**: The dataset contains 208 samples, each with 60 sonar signal attributes and a label indicating whether the object is a **rock (R)** or a **mine (M)**.
- **Format**: CSV with no header row.

---

## 🧠 Problem Statement

Classify sonar signal readings to determine whether the reflected object is a **rock** or a **mine**.

---

## 🛠️ Tech Stack & Tools

- **Language**: Python
- **Libraries**:
  - `numpy`
  - `pandas`
  - `scikit-learn`

---

## 🚀 Project Workflow

### 1. Import Libraries

Load essential Python libraries such as NumPy, Pandas, and Scikit-learn.

### 2. Data Loading & Exploration

- Load dataset using `pandas.read_csv()`
- Display basic info using `head()`, `shape`, `describe()`, and `value_counts()`

### 3. Data Preparation

- Split the dataset into features (`X`) and target labels (`Y`)
- Perform train-test split (90% training, 10% testing) with stratification

### 4. Model Training

- Train a **Logistic Regression** model on the training data

### 5. Model Evaluation

- Evaluate accuracy on both **training** and **test** sets using `accuracy_score`

### 6. Prediction System

- Accepts a new sonar input (60 features)
- Predicts whether the object is a **rock** or a **mine**

---

## 🧪 Sample Input & Output

```python
# Example Input (Sonar signal features):
input_data = (0.0453, 0.0523, ..., 0.0044)

# Model Prediction:
prediction = model.predict(input_data)

# Output:
The object is a rock
