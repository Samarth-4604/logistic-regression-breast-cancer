# Task 4: Logistic Regression - Binary Classification (Breast Cancer Dataset)

## 🧠 Objective
Build a binary classifier using **Logistic Regression** to predict whether a tumor is **malignant (0)** or **benign (1)** based on various features.

---

## 📚 Tools & Libraries Used
- Python
- scikit-learn
- pandas
- matplotlib
- numpy

---

## 📂 Dataset
We use the **Breast Cancer Wisconsin Dataset**, which is built into scikit-learn.

No external download or dataset path is needed:
```python
from sklearn.datasets import load_breast_cancer
```

---

## 🛠️ What the Script Does
1. Loads the dataset.
2. Splits data into training and testing sets.
3. Standardizes the features.
4. Trains a **Logistic Regression** model.
5. Evaluates the model using:
   - Confusion matrix
   - Precision, recall, F1-score
   - ROC-AUC score
   - ROC curve plot
6. Demonstrates **threshold tuning** for custom classification sensitivity.
7. Explains the **sigmoid function behavior** using code.

---

## 📊 Evaluation Metrics
- **Accuracy**  
- **Precision & Recall**  
- **F1 Score**  
- **ROC-AUC Score**  
- **ROC Curve Plot**

---

## ✅ How to Run

1. Make sure you have Python installed.
2. Install required libraries:

```bash
pip install scikit-learn pandas matplotlib
```

3. Run the Python script:

```bash
python logisticRegression.py
```

---

## 📈 Output Example

- Confusion matrix
- Classification report
- ROC-AUC score
- ROC curve (shown using `matplotlib`)
- Confusion matrix after threshold tuning

---

## 💡 What You Learn
- Binary classification with Logistic Regression
- How sigmoid maps scores to probability
- How threshold affects classification
- Importance of evaluation metrics

---

## 🔗 Dataset Info
Dataset Source (in scikit-learn): [Breast Cancer Wisconsin Diagnostic Data Set](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)

---
# logistic-regression-breast-cancer
