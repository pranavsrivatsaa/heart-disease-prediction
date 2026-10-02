# ❤️ Heart Disease Prediction

An end-to-end machine learning project that predicts whether a patient is likely to have heart disease from clinical measurements. It compares three models, evaluates them with a focus on **recall** (missing a sick patient is worse than a false alarm), and serves predictions through a Streamlit web app.

**🔗 Live demo:** [PASTE YOUR STREAMLIT LINK HERE]

> ⚠️ Educational project only. Not a medical device and not for diagnosis.

## Problem

In screening, a false negative (a sick patient told they are fine) is far more costly than a false positive (an extra check-up). Accuracy alone hides this, so this project optimizes and reports **recall, precision, and the confusion matrix**, not just accuracy.

## Dataset

[UCI Heart Disease dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Disease) (Kaggle mirror: [redwankarimsony/heart-disease-data](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data)).

- 920 patients (509 with heart disease, 411 without)
- Features: age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar, resting ECG, max heart rate, exercise-induced angina, ST depression, ST slope, number of major vessels, thalassemia
- Target: `num` converted to binary (0 = no disease, 1 = disease)
- Data is not included in this repo; download it from the links above.

## Method

1. **Missing values:** checked every column. `ca` (about 66% missing), `thal` (about 53%) and `slope` (about 34%) had the most. Impossible zero values in cholesterol and blood pressure were treated as missing.
2. **Preprocessing (scikit-learn Pipeline):** median imputation and standard scaling for numeric features; most-frequent imputation and one-hot encoding for categorical features. Using a single pipeline keeps training and the app consistent and avoids data leakage.
3. **Models:** Logistic Regression (baseline), Random Forest, XGBoost.
4. **Evaluation:** stratified 80/20 train/test split plus 5-fold cross-validation. Metrics: accuracy, precision, recall, F1, ROC-AUC, confusion matrix.
5. **Threshold tuning:** the decision threshold was lowered from 0.5 to 0.4, chosen using cross-validated predictions on the **training set only**, then checked once on the test set.

## Results

### Test-set comparison (184 patients)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.810 | 0.825 | 0.833 | 0.829 | 0.890 |
| Random Forest | 0.810 | 0.813 | 0.853 | 0.833 | 0.906 |
| XGBoost | 0.837 | 0.833 | 0.882 | 0.857 | 0.902 |

![Model comparison](screenshots/model_comparison.jpg)
![Confusion matrices](screenshots/confusion_matrices.jpg)

### 5-fold cross-validated recall

| Model | Recall (mean ± std) |
|---|---|
| Logistic Regression | 0.811 ± 0.057 |
| Random Forest | 0.859 ± 0.025 |
| XGBoost | 0.845 ± 0.035 |

![Cross-validation recall](screenshots/cv_recall.jpg)

Both tree-based models beat the Logistic Regression baseline. The two are close, and on a test set of 184 patients the gap between them is only a few patients, so I treat them as comparable. **Random Forest** was chosen for the app because it had the highest ROC-AUC and the highest, most stable cross-validated recall.

### Threshold tuning (recall-focused)

Lowering the threshold trades some precision for more sick patients caught:

![Threshold tuning](screenshots/threshold_tuning.jpg)

At a threshold of **0.4**, the held-out test set gives:

- **Recall: 94.1%** (96 of 102 patients with disease caught; 6 missed, down from 15 at the default 0.5)
- **Precision: 78.7%** (26 false alarms)
- Accuracy: 82.6%

![Final test result](screenshots/final_test_result.jpg)

## Demo

| Low risk | High risk |
|---|---|
| ![Low risk](screenshots/demo_low_risk_female.jpg) | ![High risk](screenshots/demo_high_risk_male.jpg) |

## Run locally

```bash
git clone https://github.com/YOUR-USERNAME/heart-disease-prediction.git
cd heart-disease-prediction
pip install -r requirements.txt
streamlit run app.py
```

## Project structure

```
├── app.py                              # Streamlit app
├── model.joblib                        # trained pipeline + decision threshold
├── heart_disease_prediction.ipynb      # full analysis and training notebook
├── requirements.txt
└── screenshots/
```

## Limitations

- Small dataset (920 patients); the test set has only 184, so small differences between models are within noise.
- The data pools several hospital sources and has a lot of missing values in some columns, handled by imputation.
- The model is not clinically validated and must not be used for real medical decisions.

## Tech stack

Python, pandas, NumPy, scikit-learn, XGBoost, Streamlit, Google Colab

## Author

**YOUR NAME** · [GitHub](https://github.com/pranavsrivatsaa) · [LinkedIn](PASTE YOUR LINKEDIN LINK)
