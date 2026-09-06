# Walmart Sales Prediction using Multiple Linear Regression from Scratch

This repository features an end-to-end implementation of a **Multiple Linear Regression model built entirely from scratch** using **NumPy** to predict Walmart's weekly department sales (`Weekly_Sales`). 

The core objective of this project is to explore how complex macroeconomic indicators, climate variations, and temporal factors influence large-scale retail performance while mastering the fundamental mathematics behind Machine Learning algorithms.

---

## 📌 Project Overview

The modeling is based on a historical retail dataset that presents a classic data science challenge: dealing with highly diverse variables operating on vastly different scales. 

### Features Analyzed:
*   **Store & Department Information** (Categorical indicators)
*   **Temporal Factors:** Holiday tracking (`Holiday_Flag`)
*   **Environmental Data:** External Temperature
*   **Macroeconomic Indicators:** Local Fuel Prices, regional Unemployment Rates, and the Consumer Price Index (CPI).

---

## ⚙️ Mathematical & Workflow Implementation

To process this data efficiently and optimize the model without high-level frameworks (like Scikit-Learn), the development workflow was structured into clear computational steps:

1. **Data Preprocessing & Formatting:** Transforming and aligning calendar dates into usable, continuous numerical variables using **Pandas**.
2. **Feature Scaling (Z-score Normalization):** Standardizing the feature matrix \(X\) using the mean (\(\mu\)) and standard deviation (\(\sigma\)) to ensure Gradient Descent convergence without numerical overflow.
3. **Gradient Descent Optimization:** Designing manual, fully vectorized implementations of the Mean Squared Error (MSE) cost function and partial derivative gradient updates.
4. **Evaluation:** Scaling predictions back to original retail dimensions for direct comparison against real data.

---

## 💻 Key Code Snippets

### Vectorized Gradient Descent Implementation
```python
def compute_gradient(w, b, x, y):
    m = x.shape[0]
    
    # 1. Compute predictions (Vector of size m)
    fw_b = np.dot(x, w) + b
    error = fw_b - y
    
    # 2. Vectorized partial derivatives using Matrix multiplication
    dw = (1 / m) * np.dot(x.T, error)
    db = (1 / m) * np.sum(error)
    
    return dw, db
```

### Correct Shape Initialization
```python
# Initializing weights matching the number of columns/features
num_features = X_train.shape[1]
initial_w = np.zeros(num_features)
initial_b = 0.0
```

---

## 💡 Project Conclusion & Scope

Ultimately, the Linear Regression model proved to be **insufficient** for accurately forecasting these sales. Real-world retail data exhibits highly non-linear patterns, complex seasonal behaviors, and deep feature interactions that a simple linear baseline cannot fully capture. 

However, this project was developed **entirely as a personal Machine Learning learning exercise**. Its true value lies in successfully mastering the core mathematics, calculus, and matrix operations under the hood of ML algorithms without relying on high-level libraries. It serves as a solid mathematical foundation before moving into more robust algorithms like Decision Trees, Random Forests, or XGBoost.

---

## 🛠️ How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com
   ```
2. Install the basic numeric dependencies:
   ```bash
   pip install numpy pandas matplotlib
   ```
3. Run the main script/notebook to see the Gradient Descent training loop and the final evaluation plots.
