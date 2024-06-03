import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Add logo to the top right corner
logo_url = "logo.png"  # Replace with your logo URL
st.sidebar.image(logo_url, use_column_width=True)

# Load data
@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Preprocess data
def preprocess_data(data):
    X = data[['Air temperature [K]', 'Process temperature [K]', 'Torque [Nm]']]
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(data['Failure Type'])

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

# Function to evaluate model performance
def evaluate_model(model, X_train, X_test, y_train, y_test):
    # Train the model
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Calculate accuracy and F1 score
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Concatenate train and test data for cross-validation
    X = pd.concat([X_train, X_test])
    y = np.append(y_train, y_test)

    # Cross-validation score
    cv_scores = cross_val_score(model, X, y, cv=5)
    mean_cv_score = np.mean(cv_scores)

    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    return accuracy, f1, mean_cv_score, conf_matrix, y_pred

# Function to map predicted labels to failure types, status, and action
def map_predicted_labels(y_pred):
    failure_types = [
        "Tidak Ada Kegagalan",
        "Kegagalan Pembuangan Panas",
        "Kegagalan Daya",
        "Kegagalan Overstrain",
        "Kegagalan Keausan Alat"
    ]
    repair_status = []
    repair_action = []

    for label in y_pred:
        if label < len(failure_types):
            if failure_types[label] == "Tidak Ada Kegagalan":
                repair_status.append("Tidak Butuh Perbaikan")
                repair_action.append("Santai Saja")
            else:
                repair_status.append("Butuh Perbaikan")
                repair_action.append("Perbaiki Segera")
        else:
            repair_status.append("Butuh Perbaikan")
            repair_action.append("Perbaiki Segera")

    return repair_status, repair_action

# Function to display evaluation results with status and action
def show_results(accuracy, f1, mean_cv_score, conf_matrix, predicted_failure, repair_status, repair_action):
    st.write("Evaluation Results:")
    st.write(f"Accuracy: {accuracy}")
    st.write(f"F1 Score: {f1}")
    st.write(f"Cross-Validation Score: {mean_cv_score}")

    # Plot confusion matrix
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    st.pyplot(fig)

    # Show predicted failure, repair status, and repair action
    st.write("Predicted Failure, Repair Status, and Repair Action:")
    results_df = pd.DataFrame({
        'Predicted Failure': predicted_failure,
        'Repair Status': repair_status,
        'Repair Action': repair_action
    })
    st.write(results_df)

# Streamlit app
st.title("PREDIKSI AKURASI DAN KEGAGALAN")
st.success("PILIH MODEL YANG TELAH DISEDIAKAN PROGRAM")

# File uploader for data input
uploaded_file = st.sidebar.file_uploader("Upload file CSV", type=["csv"])

# Sidebar for Model Selection
st.sidebar.title("Pemilihan Model")
model_name = st.sidebar.selectbox("Pilih Model", ["Random Forest", "Gradient Boosting", "SVM Classifier", "Neural Networks"])

# Button for model evaluation
button_clicked = st.sidebar.button("Evaluasi")

if uploaded_file is not None:
    data = load_data(uploaded_file)

    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(data)

    if button_clicked:
        if model_name == "Random Forest":
            model = RandomForestClassifier()
        elif model_name == "Gradient Boosting":
            model = GradientBoostingClassifier()
        elif model_name == "SVM Classifier":
            model = SVC()
        elif model_name == "Neural Networks":
            model = MLPClassifier()

        # Evaluate model
        accuracy, f1, mean_cv_score, conf_matrix, y_pred = evaluate_model(model, X_train, X_test, y_train, y_test)

        # Map predicted labels to repair status and repair action
        repair_status, repair_action = map_predicted_labels(y_pred)

        # Show results
        show_results(accuracy, f1, mean_cv_score, conf_matrix, y_pred, repair_status, repair_action)
