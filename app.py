import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import shap
import lime
import lime.lime_tabular
import time
import pickle
import warnings
warnings.filterwarnings('ignore')

# Title of the web app
st.title("Loan Status Prediction and Analysis")

# File uploader to upload dataset
uploaded_file = st.file_uploader("Upload Your CSV File", type=["csv"])

if uploaded_file is not None:
    # Loading the dataset
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Loaded Successfully!")
    st.write(df.head())

    # Exploratory Data Analysis and Preprocessing
    st.subheader("Data Preprocessing")
    
    # Check for missing values
    st.write("Missing Values:")
    st.write(df.isnull().sum())

    # Impute missing values if any
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].median(), inplace=True)

    # Outlier Removal
    df = df[df['ApplicantIncome'] < 25000]
    df = df[df['LoanAmount'] < 400000]

    # Feature Engineering
    df['Income_to_Loan_Ratio'] = df['ApplicantIncome'] / df['LoanAmount']
    df['Log_ApplicantIncome'] = np.log1p(df['ApplicantIncome'])
    df['Log_LoanAmount'] = np.log1p(df['LoanAmount'])
    df['ApplicantIncome_squared'] = df['ApplicantIncome'] ** 2  # Adding squared feature for complexity

    # Label Encoding for categorical variables
    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col])

    # Display Correlation Heatmap
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sb.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    # Splitting Data
    X = df.drop('Loan_Status', axis=1)
    y = df['Loan_Status']

    # Balancing Data using SMOTE
    smote = SMOTE(random_state=42)
    X_smote, y_smote = smote.fit_resample(X, y)

    # Standard Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_smote)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_smote, test_size=0.2, random_state=42)

    # Model Selection Dropdown
    st.subheader("Select Model for Training")
    model_selection = st.selectbox("Choose Model", ["Logistic Regression", "Random Forest", "SVC", "Voting Classifier"])

    # Hyperparameter Tuning for Random Forest or other models
    model = None
    if model_selection == "Random Forest":
        st.subheader("Hyperparameter Tuning for Random Forest")
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }
        grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        st.write(f"Best Parameters: {best_params}")
        st.write(f"Best Cross-Validation Score: {best_score:.4f}")
        model = grid_search.best_estimator_

    elif model_selection == "Logistic Regression":
        model = LogisticRegression()

    elif model_selection == "SVC":
        model = SVC(kernel='rbf', probability=True)

    elif model_selection == "Voting Classifier":
        rf = RandomForestClassifier()
        log_reg = LogisticRegression()
        svc = SVC(kernel='rbf', probability=True)
        model = VotingClassifier(estimators=[('rf', rf), ('lr', log_reg), ('svc', svc)], voting='soft')

    # Model Training and Performance
    st.subheader("Model Performance")
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    y_pred = model.predict(X_test)
    auc_score = roc_auc_score(y_test, y_pred)
    st.write(f"Training Time: {training_time:.4f} seconds")
    st.write(f"AUC Score: {auc_score:.4f}")

    # Cross-Validation Score
    st.subheader("Cross-Validation Score")
    cross_val = cross_val_score(model, X_scaled, y_smote, cv=5, scoring='accuracy')
    st.write(f"Cross-Validation Mean Score: {np.mean(cross_val):.4f}")

    # SHAP Interpretability
    st.subheader("SHAP Values Summary Plot")
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test, check_additivity=False)
    shap.summary_plot(shap_values, X_test, feature_names=X.columns)

    # LIME Interpretability (for individual predictions)
    if model_selection != "Voting Classifier":
        st.subheader("LIME Explanation for Random Instance")
        explainer_lime = lime.lime_tabular.LimeTabularExplainer(X_train, training_labels=y_train, mode='classification', training_mode='regression', feature_names=X.columns)
        idx = 1  # Example instance to explain
        exp = explainer_lime.explain_instance(X_test[idx], model.predict_proba, num_features=5)
        exp.show_in_notebook()

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    sb.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
    st.pyplot(fig)

    # ROC Curve
    st.subheader("ROC Curve")
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc='lower right')
    st.pyplot(fig)

    # Model Download Button
    st.subheader("Download Trained Model")
    save_model = st.button("Save Model")
    if save_model:
        with open("loan_status_model.pkl", "wb") as model_file:
            pickle.dump(model, model_file)
        st.write("Model saved as 'loan_status_model.pkl'")

    # User Input for Prediction
    st.subheader("Enter Your Details to Check Loan Eligibility")

    # User Inputs
    loan_id = st.text_input("Loan ID")
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Marital Status", ["Married", "Unmarried"])
    applicant_income = st.number_input("Applicant Income", min_value=0, max_value=100000, value=5000)
    loan_amount = st.number_input("Loan Amount", min_value=0, max_value=1000000, value=150000)

    # Prepare Input Data for Prediction
    user_input = np.array([[gender, married, applicant_income, loan_amount]])
    user_input = pd.DataFrame(user_input, columns=['Gender', 'Married', 'ApplicantIncome', 'LoanAmount'])
    
    # Apply preprocessing steps to the user input
    user_input['Gender'] = le.transform(user_input['Gender'])  # Encode gender
    user_input['Married'] = le.transform(user_input['Married'])  # Encode marital status
    user_input_scaled = scaler.transform(user_input)  # Standardize

    # Model Prediction
    if st.button("Check Loan Eligibility"):
        prediction = model.predict(user_input_scaled)
        result = "Approved" if prediction[0] == 1 else "Rejected"
        st.write(f"Loan Status: {result}")

    # Feature Importance (for Random Forest)
    if model_selection == "Random Forest":
        st.subheader("Feature Importance (Random Forest)")
        feature_importances = model.feature_importances_
        feature_names = X.columns
        feature_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importances
        }).sort_values(by='Importance', ascending=False)
        st.write(feature_df)

        # Feature Importance Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        sb.barplot(x='Importance', y='Feature', data=feature_df, ax=ax)
        st.pyplot(fig)

    # Download Results
    st.subheader("Download Results")
    if st.button("Download Confusion Matrix as CSV"):
        cm_df = pd.DataFrame(cm)
        cm_df.to_csv("confusion_matrix.csv", index=False)
        st.write("Confusion Matrix CSV downloaded.")
