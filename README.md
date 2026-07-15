# Student Grade Prediction - Machine Learning Classification 🎓📊

This repository contains a supervised Machine Learning project developed for the **Machine Learning** course . The goal of this project is to predict secondary school students' final mathematics grades ("Pass" or "Fail") using demographic, social, and academic attributes from the **UCI Student Performance Dataset**.

We implemented, optimized, and compared three supervised classification algorithms to find the most accurate model for predicting student success.

---

## 🚀 Models Implemented
We explored three different machine learning approaches, applying hyperparameter tuning and optimization for each:

1. **Decision Tree Classifier**
   * Analyzed performance before and after pruning.
   * Applied **Pre & Post Pruning** techniques to control tree depth and prevent overfitting.
2. **Support Vector Machine (SVM)**
   * Tuned regularization parameters ($C$) and kernel coefficients ($\gamma$).
   * Utilized RBF kernel for optimal non-linear boundary separation.
3. **k-Nearest Neighbors (KNN)**
   * Optimized the number of neighbors using **5-Fold Cross-Validation** across a range of $K$ values (1 to 14).
   * Identified **$K = 12$** as the optimal value to balance bias and variance.

---

## 🛠️ Tech Stack & Libraries
* **Language**: Python 🐍
* **Environment**: Google Colab 🚀
* **Libraries**: 
  * `scikit-learn` (Model training, evaluation metrics, scaling, and hyperparameter tuning)
  * `pandas` & `numpy` (Data manipulation, cleaning, and matrix operations)
  * `matplotlib` & `seaborn` (Data visualization and performance plotting)

---

## 📂 Project Pipeline & Workflow

### 1. Data Preprocessing & Cleaning 🧹
* Loaded and unzipped the raw student performance data.
* Handled duplicate records, standardized string variables to lower-case, and managed missing values using forward fill.
* Applied **pd.factorize()** to encode categorical social and demographic attributes.
* Binarized the target variable `G3` into two classes: **Pass (1)** if grade $\ge 10$, and **Fail (0)** if grade $< 10$.
* Applied **MinMaxScaler** to normalize numerical features, ensuring distance-based algorithms (like KNN and SVM) perform optimally.

### 2. Dataset Splitting 📊
To ensure robust evaluation, the dataset was partitioned into:
* **70%** Training Set
* **10%** Validation Set
* **20%** Test Set

---

## 📈 Evaluation & Comparative Results
The final models were evaluated on the held-out test dataset using standard classification metrics:

| Classifier | Accuracy | Precision (Fail) | Recall (Fail) | F1-Score (Fail) | Precision (Pass) | Recall (Pass) | F1-Score (Pass) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Decision Tree (After Pruning)** | **0.90** | 0.80 | 0.92 | 0.86 | 0.96 | 0.89 | 0.92 |
| **Decision Tree (Before Pruning)** | 0.8875 | 0.81 | 0.85 | 0.83 | 0.92 | 0.91 | 0.92 |
| **SVM ($C=100$, $\gamma=0.01$)** | 0.8875 | 0.77 | 0.92 | 0.84 | 0.96 | 0.87 | 0.91 |
| **KNN ($K=12$)** | 0.85 | 0.75 | 0.81 | 0.78 | 0.90 | 0.87 | 0.89 |

### 📌 Summary of Findings
* **Pruned Decision Tree** yielded the highest overall classification accuracy of **90%**.
* **SVM** and **KNN** both demonstrated solid baseline predictive performance with high precision (up to 96% and 90% respectively for predicting student success), making them highly reliable alternatives.
