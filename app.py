import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import roc_auc_score
from sklearn.svm import LinearSVC
from sklearn.impute import SimpleImputer

# Load the dataset (replace 'creditcard.csv' with your actual path)
data = pd.read_csv('creditcard.csv')

# Preprocessing and model training (same as your notebook code)
# ... (your existing code for data loading, preprocessing, and model training) ...

# Create a Streamlit app
def main():
    st.title("Credit Card Fraud Detection")

    # Display dataset information
    st.header("Dataset Information")
    st.write(data.head())
    st.write(data.info())

    # Pie chart of target variable
    labels = data['Class'].dropna().unique()
    sizes = data.Class.value_counts().reindex(labels).values
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.3f%%')
    ax.set_title('Target Variable Value Counts')
    st.pyplot(fig)

    # Correlation plot
    corr_values = data.corr()['Class'].drop('Class')
    fig, ax = plt.subplots(figsize=(10, 6))  # Adjust figsize as needed
    ax.barh(corr_values.index, corr_values.values)
    ax.set_title('Correlation with Target Variable')
    st.pyplot(fig)

    # Decision tree visualization
    st.subheader("Decision Tree Plot")
    fig, ax = plt.subplots(figsize=(12, 8))  # Adjust figsize as needed
    plot_tree(dt, filled=True, feature_names=X.columns, class_names=['Not Fraud', 'Fraud'])
    st.pyplot(fig)

    # Display model performance metrics (ROC-AUC scores)
    st.header("Model Performance")
    st.write(f'Decision Tree ROC-AUC score: {roc_auc_dt:.3f}')
    st.write(f'SVM ROC-AUC score: {roc_auc_svm:.3f}')


# Run the Streamlit app
if __name__ == "__main__":
    main()