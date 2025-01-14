
# Loan Status Prediction and Analysis

## Project Overview

This project predicts loan eligibility based on various applicant features such as gender, marital status, income, and loan amount. It leverages machine learning techniques to build a predictive model and provides detailed interpretability using SHAP and LIME.

### Key Features:
- **Data Preprocessing:** Handles missing values, feature engineering, and outlier removal.
- **Model Selection:** Implements multiple models, including Logistic Regression, Random Forest, SVC, and a Voting Classifier.
- **Performance Metrics:** Evaluates models based on AUC score, confusion matrix, and ROC curve.
- **Model Interpretability:** Provides explanations using SHAP and LIME.
- **User Input:** Allows users to input personal details to predict loan eligibility.
- **Model Export:** Saves the trained model for later use.

## Technologies Used
- **Streamlit:** For the interactive web app interface.
- **Pandas:** For data manipulation and preprocessing.
- **Scikit-learn:** For machine learning models, metrics, and data preprocessing.
- **Imbalanced-learn (SMOTE):** For handling class imbalance.
- **SHAP and LIME:** For model interpretability.
- **Matplotlib and Seaborn:** For visualization (e.g., correlation heatmap, ROC curve, confusion matrix).
- **Pickle:** For saving the trained model.

## Requirements

To run this project, you need to install the following dependencies:

- streamlit
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- imbalanced-learn
- shap
- lime

You can install these dependencies by running:

```
pip install -r requirements.txt
```

## How to Run

1. **Clone the repository:**

   ```
   git clone https://github.com/yourusername/loan-status-prediction.git
   ```

2. **Navigate to the project directory:**

   ```
   cd loan-status-prediction
   ```

3. **Run the Streamlit app:**

   ```
   streamlit run app.py
   ```

4. The web app should now be running locally. You can interact with the app by uploading your dataset and selecting the machine learning model for training.

## Features

1. **Upload CSV File:**
   - Upload a CSV file with loan-related data for analysis.
   
2. **Data Preprocessing:**
   - Missing values are imputed based on data type.
   - Outliers are removed from the `ApplicantIncome` and `LoanAmount` columns.
   - Feature engineering is performed to create new features such as `Income_to_Loan_Ratio`, `Log_ApplicantIncome`, and `Log_LoanAmount`.
   
3. **Model Training:**
   - Train models such as Logistic Regression, Random Forest, SVC, or Voting Classifier.
   - Hyperparameter tuning is performed for Random Forest.
   - The model performance is evaluated with metrics like AUC score, cross-validation, and training time.

4. **Interpretability:**
   - SHAP (SHapley Additive exPlanations) values are used to visualize the impact of each feature on the model's predictions.
   - LIME (Local Interpretable Model-agnostic Explanations) is used for model explainability, providing local interpretations for individual predictions.

5. **Confusion Matrix & ROC Curve:**
   - Visualize the confusion matrix and ROC curve for evaluating classification performance.

6. **Download Model:**
   - After training the model, you can save it as a `.pkl` file for later use.

7. **Predict Loan Status:**
   - Input personal details such as gender, marital status, income, and loan amount, and predict loan eligibility.

## Example

### Input:
- Gender: Male
- Marital Status: Married
- Applicant Income: 5000
- Loan Amount: 150000

### Output:
- Loan Status: Approved/Rejected

## Future Improvements

- **Model Tuning:** Further fine-tune the model with more sophisticated algorithms (e.g., XGBoost, LightGBM).
- **Additional Features:** Include more features, such as credit history, property area, etc.
- **Deployment:** Deploy the app on a cloud platform such as AWS or Heroku.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
