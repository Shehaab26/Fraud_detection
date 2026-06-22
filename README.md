# Telco Customer Churn Prediction

This repository contains a Telco Customer Churn analysis, model training notebook, and a Streamlit demo app that predicts the likelihood of a customer churning (leaving) based on their profile.

## Repository structure

- code/
  - app.py — Streamlit app for interactive predictions (uses the pre-trained model).
  - telco-customer-churn.ipynb — Jupyter notebook with EDA, feature engineering, model training and evaluation.
- data/
  - WA_Fn-UseC_-Telco-Customer-Churn.csv — Telco Customer Churn dataset (original CSV used in the notebook).
- xgb_model_with_meta.pkl — Pre-trained XGBoost model (pickle). The app expects this file at the repository root when run with `streamlit run code/app.py`.


## What this project does

- Explores the Telco Customer Churn dataset, performs EDA and preprocessing steps.
- Trains classification models (Logistic Regression, Random Forest, XGBoost) and evaluates their performance.
- Saves a fitted XGBoost model (and metadata) to `xgb_model_with_meta.pkl`.
- Provides a Streamlit application (`code/app.py`) to interactively predict churn risk for a single customer profile.


## Quick start

1. Clone the repository:

   git clone https://github.com/Shehaab26/Fraud_detection.git
   cd Fraud_detection

2. Install dependencies. The notebook lists the main packages used. You can install these with pip:

   pip install scikit-learn xgboost joblib pandas matplotlib seaborn pyngrok flask streamlit

(Optionally create a virtual environment first.)

3. Confirm the model file `xgb_model_with_meta.pkl` is in the repository root. The Streamlit app loads `xgb_model_with_meta.pkl` by relative filename.

4. Run the Streamlit app:

   streamlit run code/app.py

5. Use the sidebar controls to set a customer profile and click "🔮 Predict Churn Risk" to see predicted probability and a visual bar chart.


## Notes about the model file

- The repository includes `xgb_model_with_meta.pkl`. The app contains logic to handle two common pickle formats:
  - A dictionary with keys `model` and optionally `feature_columns`.
  - A bare model object (e.g., XGBClassifier). When a bare model is used, the app will attempt to read `model.feature_names_in_` to infer feature order.
- If your model was trained using a preprocessing pipeline (OneHotEncoder, StandardScaler, etc.), ensure you save and load the full pipeline (so the app can accept raw feature values). If you only save a raw classifier, you must pre-transform input data the same way as during training.


## Reproducing training

- Open `code/telco-customer-churn.ipynb` and run the cells to reproduce data cleaning, feature engineering, model training and evaluation.
- The notebook installs / references packages: scikit-learn, xgboost, joblib, pandas, matplotlib, seaborn, pyngrok, flask.


## Implementation details (from the app)

- The Streamlit app expects categorical columns to be present with the same names the model expects. If `feature_columns` metadata is present in the pickle, the app will try to align input columns to that order and will error if required columns are missing.
- For categorical variables the app will cast object-type columns to pandas `category` dtype before prediction to match XGBoost's categorical handling (if used during training).


## Known limitations and troubleshooting

- If you see an error about missing features, confirm the `xgb_model_with_meta.pkl` contains `feature_columns` or re-train/save the pipeline so the model receives preprocessed features.
- If `xgb_model_with_meta.pkl` is not found, place it at the repository root (or update `app.py` to point to the correct path).
- If you used custom preprocessing (e.g., target encoding, custom transformers), save the full pipeline (transformer + model) together.


## License

This repository does not include an explicit license. Add one (for example, MIT) if you plan to share or collaborate publicly.


## Contact

Maintainer: Shehaab26

If you want improvements (requirements file, Docker support, unit tests), open an issue or create a pull request with suggested changes.
