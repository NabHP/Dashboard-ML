import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import recall_score, confusion_matrix, classification_report

# Set Streamlit page configuration (must be the first Streamlit command)
st.set_page_config(page_title='Bank Deposit Prediction', layout='wide')

# Add a title and description in the sidebar
st.sidebar.title('Welcome!')
st.sidebar.markdown('''
This dashboard allows you to predict whether a customer will subscribe to a bank deposit product based on their features. 
It includes a comparison between control and treatment groups, confusion matrix, revenue uplift calculations, and an interactive 
feature prediction tool.
''')

# Load model and data
model_path = 'kingsman_model_bank_deposit_lgbm_tuned.sav'
final_model = joblib.load(open(model_path, 'rb'))
X_new = pd.read_csv('X_new_for_inference.csv')
y_new = pd.read_csv('y_new_actual.csv')
treatment_group_sample = pd.read_csv('treatment_group_sample1000.csv')
control_group_sample = pd.read_csv('control_group_sample1000.csv')

# Predict probabilities
y_proba_new = final_model.predict_proba(X_new)[:, 1]  # Get the probability for the positive class (deposit)
X_new['predicted_proba'] = y_proba_new

# Sort Treatment Group by predicted probability (descending order)
treatment_group = X_new.sort_values(by='predicted_proba', ascending=False).iloc[:len(X_new)//2]
control_group = X_new.drop(treatment_group.index)

# Simulating actual outcomes for the Treatment and Control Groups
treatment_group_sample['actual_deposit'] = y_new.loc[treatment_group_sample.index]
control_group_sample['actual_deposit'] = y_new.loc[control_group_sample.index]

# Add predicted labels based on a threshold (e.g., 0.5)
treatment_group_sample['predicted'] = (treatment_group_sample['predicted_proba'] >= 0.5).astype(int)
control_group_sample['predicted'] = (control_group_sample['predicted_proba'] >= 0.5).astype(int)

# Calculate conversion rates
treatment_conversion_rate_sample = treatment_group_sample['actual_deposit'].mean()
control_conversion_rate_sample = control_group_sample['actual_deposit'].mean()
uplift = treatment_conversion_rate_sample - control_conversion_rate_sample

# Cost and Revenue Calculations
deposit_amount = 31.75
marketing_cost = 1.7228  # Cost per customer

# Gross Revenue Calculation
control_revenue = np.sum(control_group_sample['predicted']) * deposit_amount
treatment_revenue = np.sum(treatment_group_sample['predicted']) * deposit_amount

# Marketing Costs
control_cost = len(control_group_sample) * marketing_cost
treatment_cost = len(treatment_group_sample) * marketing_cost

# Net Revenue after Marketing Cost
control_net_revenue = control_revenue - control_cost
treatment_net_revenue = treatment_revenue - treatment_cost
uplift_net_revenue = treatment_net_revenue - control_net_revenue

# Classification Report
report = classification_report(y_new, (y_proba_new >= 0.5).astype(int), output_dict=True)
report_df = pd.DataFrame(report).transpose()

# Streamlit UI Layout with Tabs
st.title("Kingsman Bank Deposit Prediction Dashboard")

# Tabs
tab1, tab2, tab3 = st.tabs(["Control vs Treatment", "Confusion Matrix & Revenue Uplift", "Interactive Feature Prediction"])

# First Tab: Control vs. Treatment Dataset Comparison
with tab1:
    st.header("Control vs. Treatment Dataset Comparison")
    st.markdown('''These tables allow you to compare the control and treatment groups based on their predicted probabilities, actual outcomes, and conversion rates. The treatment group consists of customers who are more likely to subscribe to a deposit product, while the control group includes those less likely.''')
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Control Group")
        st.write(control_group_sample[['actual_deposit', 'predicted', 'predicted_proba']].head())
        st.write(f"Control Group Conversion Rate: **{control_conversion_rate_sample:.2%}**")

    with col2:
        st.subheader("Treatment Group")
        st.write(treatment_group_sample[['actual_deposit', 'predicted', 'predicted_proba']].head())
        st.write(f"Treatment Group Conversion Rate: **{treatment_conversion_rate_sample:.2%}**")

# Second Tab: Confusion Matrix and Revenue Uplift
with tab2:
    st.header("Confusion Matrix and Revenue Uplift Calculation")
    st.markdown('''This section shows the accuracy of our models with confusion matrices for treatment groups, followed by the net revenue uplift from the control and treatment groups, which measures the financial impact of the treatment compared to the control.''')
    st.markdown("---")

    # First Row: Confusion Matrices 
    st.subheader("Confusion Matrix - Treatment Group")
    cm_treatment = confusion_matrix(treatment_group_sample['actual_deposit'], treatment_group_sample['predicted'])
    fig_cm_treatment, ax_cm_treatment = plt.subplots()
    sns.heatmap(cm_treatment, annot=True, fmt="d", cmap="Blues", ax=ax_cm_treatment)
    ax_cm_treatment.set_xlabel('Predicted labels')
    ax_cm_treatment.set_ylabel('True labels')
    st.pyplot(fig_cm_treatment)
        

    
    # Second Row: Net Revenue Uplift and Bar Chart 
    st.subheader("Net Revenue Uplift Calculation (After Marketing Costs)")
    col3, col4 = st.columns([1, 1])

    with col3:
        st.write(f"Control Group Gross Revenue: **€{control_revenue:,.2f}**")
        st.write(f"Treatment Group Gross Revenue: **€{treatment_revenue:,.2f}**")
        st.write(f"Control Group Marketing Cost: **€{control_cost:,.2f}**")
        st.write(f"Treatment Group Marketing Cost: **€{treatment_cost:,.2f}**")
        st.write(f"Control Group Net Revenue: **€{control_net_revenue:,.2f}**")
        st.write(f"Treatment Group Net Revenue: **€{treatment_net_revenue:,.2f}**")
        st.write(f"Net Revenue Uplift: **€{uplift_net_revenue:,.2f}**")

    with col4:
        fig_revenue, ax_revenue = plt.subplots(figsize=(6,4))
        ax_revenue.bar(['Control Group Net Revenue', 'Treatment Group Net Revenue'], [control_net_revenue, treatment_net_revenue], color=['green', 'red'])
        ax_revenue.set_ylabel('Net Revenue (€)')
        st.pyplot(fig_revenue)

# Third Tab: Interactive Feature Prediction
with tab3:
    st.header("Interactive Feature Prediction")
    st.markdown('''Feel free to adjust various features and see how they affect the prediction of whether a customer will subscribe to a bank deposit product. This can help you observe and better understand the factors that influence customer decisions.''')
    st.markdown("---")

    user_input = {}

    # Subtitle for Numerical Features
    st.subheader("Numerical Features")
    
    # Create three columns layout
    col1, col2, col3 = st.columns(3)

    # Sliders for numerical features (10 inputs)
    numeric_features = X_new.select_dtypes(include=[np.number]).columns
    for idx, feature in enumerate(numeric_features):
        if idx % 3 == 0:
            with col1:
                min_value = float(X_new[feature].min())
                max_value = float(X_new[feature].max())
                mean_value = float(X_new[feature].mean())
                user_input[feature] = st.slider(f"{feature}", min_value, max_value, mean_value)
        elif idx % 3 == 1:
            with col2:
                min_value = float(X_new[feature].min())
                max_value = float(X_new[feature].max())
                mean_value = float(X_new[feature].mean())
                user_input[feature] = st.slider(f"{feature}", min_value, max_value, mean_value)
        elif idx % 3 == 2:
            with col3:
                min_value = float(X_new[feature].min())
                max_value = float(X_new[feature].max())
                mean_value = float(X_new[feature].mean())
                user_input[feature] = st.slider(f"{feature}", min_value, max_value, mean_value)

    # Subtitle for Categorical Features
    st.subheader("Categorical Features")
    
    # Create three columns layout for categorical features
    col1, col2, col3 = st.columns(3)

    # Dropdowns for categorical features (9 inputs)
    categorical_features = X_new.select_dtypes(exclude=[np.number]).columns
    for idx, feature in enumerate(categorical_features):
        if idx % 3 == 0:
            with col1:
                unique_values = X_new[feature].unique()
                user_input[feature] = st.selectbox(f"{feature}", unique_values)
        elif idx % 3 == 1:
            with col2:
                unique_values = X_new[feature].unique()
                user_input[feature] = st.selectbox(f"{feature}", unique_values)
        elif idx % 3 == 2:
            with col3:
                unique_values = X_new[feature].unique()
                user_input[feature] = st.selectbox(f"{feature}", unique_values)

    # Convert user input to DataFrame and make prediction
    input_df = pd.DataFrame([user_input])
    user_proba = final_model.predict_proba(input_df)[:, 1]
    user_prediction = (user_proba >= 0.5).astype(int)

    st.subheader("Prediction Results")
    st.write(f"Successful Rate of Deposit: **{user_proba[0] * 100:.2f}%**")
    st.write(f"Predicted Deposit: **{'Yes' if user_prediction[0] == 1 else 'No'}**")

    # Classification Report
    st.subheader("Classification Report")
    st.dataframe(report_df)
