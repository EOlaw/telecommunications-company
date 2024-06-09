# Import Libraries and Load Data
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

# Load data
df = pd.read_csv('telecom_customer_data.csv')

# Data Preprocessing
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
    df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
    return df

df = preprocess_data(df)

# Save plots instead of showing them
def perform_eda(df):
    # Statistical summary
    stats_summary = df.describe().to_html()

    # Correlation matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
    plt.savefig('correlation_matrix.png')
    plt.close()

    # Distribution of target variable
    sns.countplot(x='Churn', data=df)
    plt.title('Churn Distribution')
    plt.savefig('churn_distribution.png')
    plt.close()

    # Pairplot for selected features
    selected_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']
    sns.pairplot(df[selected_features], hue='Churn', palette='coolwarm')
    plt.savefig('pairplot.png')
    plt.close()

    return stats_summary

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
    log_reg_report = classification_report(y_test, log_reg_pred)
    rf_clf_report = classification_report(y_test, rf_clf_pred)
    
    print("Logistic Regression Report:")
    print(log_reg_report)
    print(f"ROC-AUC Score: {roc_auc_score(y_test, log_reg_pred)}")

    print("Random Forest Report:")
    print(rf_clf_report)
    print(f"ROC-AUC Score: {roc_auc_score(y_test, rf_clf_pred)}")

    return log_reg, rf_clf, log_reg_report, rf_clf_report, X_test, y_test

log_reg, rf_clf, log_reg_report, rf_clf_report, X_test, y_test = build_and_evaluate_model(features, target)

def visualize_model_performance(log_reg, rf_clf, X_test, y_test):
    # Predict probabilities
    log_reg_probs = log_reg.predict_proba(X_test)[:, 1]
    rf_clf_probs = rf_clf.predict_proba(X_test)[:, 1]
    
    # ROC Curve
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
    plt.savefig('roc_curve.png')
    plt.close()

visualize_model_performance(log_reg, rf_clf, X_test, y_test)

def generate_html_report(stats_summary, log_reg_report, rf_clf_report):
    html_content = f"""
    <html>
    <head>
        <title>Telecom Customer Churn Analysis Report</title>
    </head>
    <body>
        <h1>Telecom Customer Churn Analysis Report</h1>
        
        <h2>Data Preprocessing</h2>
        <p>Missing values in 'TotalCharges' were filled with the mean of the column. Categorical variables were encoded using one-hot encoding.</p>
        
        <h2>Exploratory Data Analysis</h2>
        <h3>Statistical Summary</h3>
        {stats_summary}
        
        <h3>Correlation Matrix</h3>
        <img src="correlation_matrix.png" alt="Correlation Matrix">
        
        <h3>Churn Distribution</h3>
        <img src="churn_distribution.png" alt="Churn Distribution">
        
        <h3>Pairplot of Selected Features</h3>
        <img src="pairplot.png" alt="Pairplot of Selected Features">
        
        <h2>Model Building and Evaluation</h2>
        <h3>Logistic Regression Report</h3>
        <pre>
        {log_reg_report}
        </pre>
        <h3>Random Forest Report</h3>
        <pre>
        {rf_clf_report}
        </pre>
        
        <h3>ROC Curve</h3>
        <img src="roc_curve.png" alt="ROC Curve">
        
    </body>
    </html>
    """
    with open("report.html", "w") as file:
        file.write(html_content)

stats_summary = perform_eda(df)
generate_html_report(stats_summary, log_reg_report, rf_clf_report)
