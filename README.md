# Sampling Assignment – Credit Card Dataset

## 📌 Objective
The objective of this assignment is to understand the importance of sampling techniques in handling imbalanced datasets and to analyze how different sampling strategies affect the performance of various machine learning models.

---

## 📂 Dataset
The dataset used in this project is a **Credit Card dataset**, downloaded from the following GitHub repository:

https://github.com/AnjulaMehto/Sampling_Assignment/blob/main/Creditcard_data.csv

The dataset is **highly imbalanced**, where the majority class (Class = 0) heavily outweighs the minority class (Class = 1).

---

## ⚠️ Problem Statement
In real-world applications, imbalanced datasets can significantly impact machine learning model performance.  
The task is to:
1. Convert the given dataset into a **balanced class dataset**
2. Create **five different samples**
3. Apply **five different sampling techniques** on **five different machine learning models**
4. Compare the accuracy results and determine which sampling technique works best for which model

---

## 🔁 Step 1: Handling Class Imbalance
Before applying sampling techniques, the dataset is first **balanced** using:

- **SMOTE (Synthetic Minority Oversampling Technique)**

SMOTE generates synthetic samples for the minority class so that both classes have equal representation.

This step ensures that the dataset is suitable for fair model training.

---

## 📊 Step 2: Sampling Techniques Used
After balancing the dataset, five different sampling techniques were applied to create five samples:

| Sampling Name | Technique |
|--------------|----------|
| Sampling1 | Simple Random Sampling |
| Sampling2 | Systematic Sampling |
| Sampling3 | Stratified Sampling |
| Sampling4 | Cluster Sampling |
| Sampling5 | Bootstrap Sampling |

These techniques are based on **probabilistic sampling methods** covered in the reference PPT.

---

## 🤖 Step 3: Machine Learning Models Used
Five different machine learning models were trained on each sampled dataset:

| Model | Algorithm |
|------|----------|
| M1 | Logistic Regression |
| M2 | Decision Tree Classifier |
| M3 | Random Forest Classifier |
| M4 | K-Nearest Neighbors (KNN) |
| M5 | Support Vector Machine (SVM) |

---

## 🧪 Step 4: Evaluation Metric
- **Accuracy** is used as the evaluation metric
- Each model is trained and tested on each sampled dataset
- Results are recorded in a **5 × 5 accuracy comparison table**

---

## 📈 Results
The final results are stored in:


The table compares model performance across different sampling techniques in the following format:

| Model | Sampling1 | Sampling2 | Sampling3 | Sampling4 | Sampling5 |
|------|----------|----------|----------|----------|----------|
| M1 | 91.02 | 86.93 | 94.26 | 95.95 | 97.06 |
| M2 | 97.96 | 96.08 | 96.72 | 98.65 | 99.67 |
| M3 | 99.59 | 99.35 | 99.59 | 100.0 | 100.0 |
| M4 | 85.31 | 79.74 | 83.61 | 85.14 | 89.22 |
| M5 | 65.71 | 67.32 | 69.26 | 64.86 | 70.59 |

---

## 🧠 Observation & Conclusion
- Different sampling techniques impact different models in different ways
- Stratified and Bootstrap sampling generally provide stable performance
- Tree-based models such as Random Forest perform well across multiple sampling techniques
- Cluster sampling performance depends on cluster selection and size
- The choice of sampling technique plays a crucial role in model accuracy

---

## ▶️ How to Run the Project
1. Open the project folder in **VS Code**
2. Install required libraries:
   ```bash
   pip install pandas numpy scikit-learn imbalanced-learn
   cd src
   python main.py
