import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import shap
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

    # Impute missing values if any (Example: Fill with median for numerical and mode for categorical)
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

    # Model Comparison
    models = {
        'Logistic Regression': LogisticRegression(),
        'Random Forest': RandomForestClassifier(random_state=42),
        'SVC': SVC(kernel='rbf', probability=True)
    }

    results = {}
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        auc_score = roc_auc_score(y_test, y_pred)
        results[model_name] = auc_score

    # Display Model Comparison Results
    st.subheader("Model Comparison")
    for model, score in results.items():
        st.write(f"{model}: AUC = {score:.4f}")

    # Best Model Selection and Interpretation
    best_model_name = max(results, key=results.get)
    best_model = models[best_model_name]

    # SHAP Interpretability
    st.subheader("SHAP Values Summary Plot")
    explainer = shap.Explainer(best_model, X_train)
    shap_values = explainer(X_test, check_additivity=False)
    shap.summary_plot(shap_values, X_test, feature_names=X.columns)

    # Confusion Matrix
    st.subheader(f"{best_model_name} Confusion Matrix")
    y_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    sb.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
    st.pyplot(fig)

    # ROC Curve
    st.subheader(f"{best_model_name} ROC Curve")
    y_proba = best_model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'{best_model_name} - ROC Curve')
    ax.legend(loc='lower right')
    st.pyplot(fig)
