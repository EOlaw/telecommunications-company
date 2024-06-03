# Import Libraries and Load Data
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('telecom_customer_data.csv')

# Data Proprocessing
def preprocess_data(df):
    # Handle missing values
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].mean(), inplace=True)
    # Encode categorical variables
    categorical_columns = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
                           'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                           'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
                           'PaperlessBilling', 'PaymentMethod']
    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    # Convert 'Churn' column to numeric
    df ['Churn'] = df['Churn'].apply(lambda x: 1 if x =='Yes' else 0)
    return df

df = preprocess_data(df)

# Exploratory Data Analysis (EDA)
def perform_eda(df):
    # Statistical summary
    print(df.describe())

    # Correlation matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
    plt.show()

    # Distribution of target variable
    sns.countplot(x='Churn', data=df)
    plt.title('Churn Distribution')
    plt.show()

    # Pairplot for selected features
    selected_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']
    sns.pairplot(df[selected_features], hue='Churn', palette='coolwarm')
    plt.show()

ds = perform_eda(df)

# Feature Engineering
def feature_engineering(df):
    # Create a new feature: 'MonthlyChargesPerTenure'
    df['MonthlyChargesPerTenure'] = df['MonthlyCharges'] / (df['tenure'] + 1)

    # Feature selection (Simple example)
    features = df.drop(['customerID', 'Churn'], axis=1)
    target = df['Churn']

    return features, target

features, target = feature_engineering(df)

# Model Building and Evaluation
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

def build_and_evaluate_model(features, target):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

    # Train models
    log_reg = LogisticRegression(max_iter=1000)
    rf_clf = RandomForestClassifier(n_estimators=100)

    log_reg.fit(X_train, y_train)
    rf_clf.fit(X_train, y_train)

    # Predictions
    log_reg_pred = log_reg.predict(X_test)
    rf_clf_pred = rf_clf.predict(X_test)

    # Evaluation
    print("Logistic Regression Report:")
    print(classification_report(y_test, log_reg_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, log_reg_pred)}")

    print("Random Forest Report:")
    print(classification_report(y_test, rf_clf_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, rf_clf_pred)}")

    return log_reg, rf_clf

log_reg, rf_clf = build_and_evaluate_model(features, target)

# Visualization
def visualize_model_performance(log_reg, rf_clf, X_test, y_test):
    # Predict probabilities
    log_reg_probs = log_reg.predict_proba(X_test)[:, 1]
    rf_clf_probs = rf_clf.predict_proba(X_test)[:, 1]
    
    # ROC Curve
    from sklearn.metrics import roc_curve
    
    log_reg_fpr, log_reg_tpr, _ = roc_curve(y_test, log_reg_probs)
    rf_clf_fpr, rf_clf_tpr, _ = roc_curve(y_test, rf_clf_probs)
    
    plt.figure(figsize=(10, 6))
    plt.plot(log_reg_fpr, log_reg_tpr, label='Logistic Regression')
    plt.plot(rf_clf_fpr, rf_clf_tpr, label='Random Forest')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

visualize_model_performance(log_reg, rf_clf, features, target)
