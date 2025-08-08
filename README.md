# 📈 Bitcoin Price Movement Prediction (Logistic Regression)

This project pulls and processes nearly **50,000 rows of historical Bitcoin data**, then applies a machine learning model to predict directional movement (`target_multi`) using engineered features. The model achieves over **99% accuracy** using **Logistic Regression**, raising important questions around data validity and future model robustness.

---

## 🧠 Project Overview

- Pulled 50,000+ rows of crypto price and volume data from an API
- Cleaned and preprocessed the data (removed NaNs, scaled features)
- Trained a **Logistic Regression** model to predict upward price movement (`target_multi = 1`)
- Achieved **>99% accuracy** on the test set

---

## 📊 Technologies Used

- Python  
- Pandas & NumPy  
- Scikit-learn (LogisticRegression, train/test split, StandardScaler)  
- Jupyter Notebook  

---

## 🚨 Note on Model Performance

While the model achieves **extremely high accuracy**, this may indicate **data leakage**, class imbalance, or feature contamination. It is most likely that we see this perfomance from class imbalence due to the high volume of multi_target equalling zero, as seen in our distribution bar chart.

---

## ✅ Next Steps: Robust Model Testing

To ensure the model is genuinely predictive and not just exploiting leakage or bias, the following steps are planned:

### 1. **Leakage Check**
- Audit input features to ensure no future data (e.g., `close_price`, forward returns) is being used  
- Run correlation analysis between features and the target

### 2. **Class Balance Evaluation**
- Analyze the distribution of `target_multi`
- Re-train using class balancing methods (e.g., weighted loss, SMOTE, undersampling)

### 3. **Model Comparison**
- Establish baselines with `DummyClassifier`
- Compare results with other models such as Decision Trees, Random Forests, and SVM

### 4. **Time Series Cross-Validation**
- Use `TimeSeriesSplit` instead of random train/test splits
- Better simulate real-world prediction performance

### 5. **Out-of-Sample Testing**
- Create a true holdout dataset from a future time window  
- Use it only once for final evaluation after all model tuning is complete

---

## 📁 Files

- `btc_ml_model.ipynb` – Jupyter Notebook with data preprocessing, training, and evaluation  
- `btc_data.csv` – Cleaned dataset with 50,000+ rows  
- `README.md` – Project summary and next steps  

---

## 🚀 Future Plans

The long-term goal is to build a robust crypto price movement prediction engine by integrating additional features such as:

- Macroeconomic data  
- Social sentiment indicators  
- Technical signals (e.g., RSI, MACD)  
- Real-time data streaming  

