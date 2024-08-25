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
treatment_group_sample = pd.read_csv('treatment_group_samples.csv')
control_group_sample = pd.read_csv('control_group_samples.csv')

# Predict probabilities for the entire dataset
y_proba_new = final_model.predict_proba(X_new)[:, 1]  # Get the probability for the positive class (deposit)
X_new['predicted_proba'] = y_proba_new

# Add predicted labels based on a threshold (e.g., 0.5)
treatment_group_sample['predicted'] = (treatment_group_sample['predicted_proba'] >= 0.5).astype(int)
control_group_sample['predicted'] = (control_group_sample['predicted_proba'] >= 0.5).astype(int)

# Calculate conversion rates using the actual outcomes
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

# Define tabs
tab1, tab2, tab3 = st.tabs(["Control vs Treatment", "Confusion Matrix & Revenue Uplift", "Interactive Feature Prediction"])

# Tab 1: Control vs Treatment Dataset Comparison
with tab1:
    st.subheader("Control vs Treatment Dataset Comparison")
    st.markdown('''These tables allow you to compare the control and treatment groups based on their predicted probabilities, actual outcomes, and conversion rates. The treatment group consists of customers who are more likely to subscribe to a deposit product, while the control group includes those less likely.''')
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Control Group")
        st.write(f"Total Data: **{len(control_group_sample)}**")
        st.write(control_group_sample[['actual_deposit', 'predicted', 'predicted_proba']].head())
        st.write(f"Control Group Conversion Rate: **{control_conversion_rate_sample:.2%}**")

    with col2:
        st.subheader("Treatment Group")
        st.write(f"Total Data: **{len(treatment_group_sample)}**")
        st.write(treatment_group_sample[['actual_deposit', 'predicted', 'predicted_proba']].head())
        st.write(f"Treatment Group Conversion Rate: **{treatment_conversion_rate_sample:.2%}**")

# Tab 2: Confusion Matrix, Revenue Uplift
with tab2:
    st.header("Revenue Uplift Calculation and Confusion Matrix")
    st.markdown('''This section shows the net revenue uplift from the control and treatment groups, measuring the financial impact of the treatment compared to the control. It also presents the accuracy of our models with a confusion matrix for the treatment group.''')
    st.markdown("---")

    # First Row: Net Revenue Uplift and Bar Chart 
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
        
    # Second Row: Confusion Matrix
    st.subheader("Confusion Matrix - Treatment Group")
    col5, col6 = st.columns([1, 1])

    with col5:
        cm_treatment = confusion_matrix(treatment_group_sample['actual_deposit'], treatment_group_sample['predicted'])
        fig_cm_treatment, ax_cm_treatment = plt.subplots(figsize=(5, 3))
        sns.heatmap(cm_treatment, annot=True, fmt="d", cmap="Blues", ax=ax_cm_treatment)
        ax_cm_treatment.set_xlabel('Predicted labels')
        ax_cm_treatment.set_ylabel('True labels')
        st.pyplot(fig_cm_treatment)

# Tab 3: Interactive Feature Prediction
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
    numeric_features = [feature for feature in numeric_features if feature != 'predicted_proba']
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

   # Add slider for threshold adjustment
    st.subheader("Adjust Prediction Threshold")
    threshold = st.slider("Select Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

    # Convert user input to DataFrame and make prediction with the selected threshold
    input_df = pd.DataFrame([user_input])
    user_proba = final_model.predict_proba(input_df)[:, 1]
    user_prediction = (user_proba >= threshold).astype(int)

    st.subheader("Prediction Results")
    st.write(f"Predicted Deposit: **{'Yes' if user_prediction[0] == 1 else 'No'}** at a threshold of {threshold:.2f}")

    # Classification Report
    st.subheader("Classification Report")
    st.dataframe(report_df)
