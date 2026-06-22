import streamlit as st
import pandas as pd
import pickle
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# Configure the Streamlit page
st.set_page_config(page_title="Telco Churn Prediction", page_icon="📉", layout="wide")

st.title("📉 Telco Customer Churn Predictor")
st.write("""
This app predicts the likelihood of a customer leaving (churning) based on their profile. 
Adjust the customer features in the sidebar to see how they impact the churn probability!
""")

# --- 1. Load the Model ---
@st.cache_resource
def load_model():
    try:
        with open("xgb_model_with_meta.pkl", "rb") as f:
            model_data = pickle.load(f)

        # Handle cases where the pickle file is a dict containing the model and metadata
        if isinstance(model_data, dict) and 'model' in model_data:
            if 'feature_columns' not in model_data and hasattr(model_data['model'], 'feature_names_in_'):
                model_data['feature_columns'] = list(model_data['model'].feature_names_in_)
            return model_data

        # Otherwise assume the loaded object is the model itself
        return {
            'model': model_data,
            'feature_columns': list(model_data.feature_names_in_) if hasattr(model_data, 'feature_names_in_') else None
        }
    except FileNotFoundError:
        st.error("Model file 'xgb_model_with_meta.pkl' not found. Please ensure it's in the same directory.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return None

model = load_model()

# --- 2. Sidebar Input Form ---
st.sidebar.header("⚙️ Customer Profile Input")

def get_user_input():
    # Demographics
    st.sidebar.subheader("Demographics")
    gender = st.sidebar.radio('Gender', ('Female', 'Male'))
    senior_citizen = st.sidebar.radio('Senior Citizen', (0, 1), format_func=lambda x: "Yes" if x==1 else "No")
    partner = st.sidebar.radio('Partner', ('Yes', 'No'))
    dependents = st.sidebar.radio('Dependents', ('Yes', 'No'))
    
    # Account Information
    st.sidebar.subheader("Account Info")
    tenure = st.sidebar.slider('Tenure (months)', min_value=0, max_value=72, value=12)
    contract = st.sidebar.selectbox('Contract', ('Month-to-month', 'One year', 'Two year'))
    paperless_billing = st.sidebar.radio('Paperless Billing', ('Yes', 'No'))
    payment_method = st.sidebar.selectbox('Payment Method', (
        'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
    ))
    monthly_charges = st.sidebar.number_input('Monthly Charges ($)', min_value=0.0, max_value=150.0, value=70.0)
    total_charges = st.sidebar.number_input('Total Charges ($)', min_value=0.0, max_value=10000.0, value=840.0)

    # Services Subscribed
    st.sidebar.subheader("Services Subscribed")
    phone_service = st.sidebar.radio('Phone Service', ('Yes', 'No'))
    multiple_lines = st.sidebar.selectbox('Multiple Lines', ('No phone service', 'No', 'Yes'))
    internet_service = st.sidebar.selectbox('Internet Service', ('DSL', 'Fiber optic', 'No'))
    
    # Advanced Services
    online_security = st.sidebar.selectbox('Online Security', ('No', 'Yes', 'No internet service'))
    online_backup = st.sidebar.selectbox('Online Backup', ('Yes', 'No', 'No internet service'))
    device_protection = st.sidebar.selectbox('Device Protection', ('No', 'Yes', 'No internet service'))
    tech_support = st.sidebar.selectbox('Tech Support', ('No', 'Yes', 'No internet service'))
    streaming_tv = st.sidebar.selectbox('Streaming TV', ('No', 'Yes', 'No internet service'))
    streaming_movies = st.sidebar.selectbox('Streaming Movies', ('No', 'Yes', 'No internet service'))

    # Store in a dictionary
    data = {
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }
    
    return pd.DataFrame(data, index=[0])

input_df = get_user_input()

# --- 3. Main Dashboard ---
st.subheader("Selected Customer Profile")
st.dataframe(input_df, hide_index=True)

st.markdown("---")

if st.button("🔮 Predict Churn Risk", type="primary", use_container_width=True):
    if model is not None:
        try:
            model_obj = model['model']
            feature_columns = model.get('feature_columns')

            predict_df = input_df.copy()
            if feature_columns is not None:
                missing_cols = [col for col in feature_columns if col not in predict_df.columns]
                if missing_cols:
                    raise ValueError(f"Missing required feature columns: {missing_cols}")
                predict_df = predict_df[feature_columns]

            # XGBoost often requires categorical columns to be explicitly typed if enable_categorical was set to True during training
            for col in predict_df.select_dtypes(include=['object']).columns:
                predict_df[col] = predict_df[col].astype('category')
                
            # Make predictions
            prediction = model_obj.predict(predict_df)
            prediction_proba = model_obj.predict_proba(predict_df)
            
            # Format outputs
            churn_risk = prediction[0]
            prob_no_churn = prediction_proba[0][0]
            prob_churn = prediction_proba[0][1]

            # Display Results Layout
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("Result")
                if churn_risk == 1:
                    st.error("⚠️ **High Risk of Churn!**")
                else:
                    st.success("✅ **Customer is Likely to Stay.**")
                    
                st.write(f"**Churn Probability:** {prob_churn:.1%}")
            
            with col2:
                st.subheader("Probability Distribution")
                
                # Create Matplotlib Figure
                fig, ax = plt.subplots(figsize=(6, 3))
                sns.barplot(
                    x=['Stay (No Churn)', 'Leave (Churn)'], 
                    y=[prob_no_churn, prob_churn], 
                    palette=['#2ca02c', '#d62728'], 
                    ax=ax
                )
                
                # Customize the chart
                ax.set_ylabel('Probability', fontweight='bold')
                ax.set_ylim(0, 1.1)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
                # Add percentage labels on top of the bars
                for p in ax.patches:
                    ax.annotate(f'{p.get_height():.1%}', 
                                (p.get_x() + p.get_width() / 2., p.get_height()), 
                                ha='center', va='bottom', fontsize=12, fontweight='bold', 
                                xytext=(0, 5), textcoords='offset points')
                                
                st.pyplot(fig)
                
        except Exception as e:
            st.error(f"**Prediction Failed.**")
            st.warning(f"""
            If your model was trained using an scikit-learn pipeline (e.g. `OneHotEncoder`, `StandardScaler`), 
            you need to ensure the raw dataframe is transformed first.
            
            *Detailed Error Message:* {e}
            """)