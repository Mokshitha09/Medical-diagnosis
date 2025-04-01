import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load datasets
diabetes_data = pd.read_csv('diabetes.csv')
heart_data = pd.read_csv('heart.csv')
kidney_data = pd.read_csv('kidney_disease.csv')

# Preprocessing function
def preprocess_data(df, target_column):
    df = df.dropna()
    # Encode categorical values if present
    for col in df.columns:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
    X = df.drop(columns=[target_column])
    y = df[target_column]
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y

# Train models
def train_models():
    models = {}
    # Diabetes
    X, y = preprocess_data(diabetes_data, 'Outcome')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    diabetes_model = RandomForestClassifier().fit(X_train, y_train)
    models['diabetes'] = diabetes_model

    # Heart Disease
    X, y = preprocess_data(heart_data, 'target')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    heart_model = RandomForestClassifier().fit(X_train, y_train)
    models['heart'] = heart_model

    # Kidney Disease
    X, y = preprocess_data(kidney_data, 'classification')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    kidney_model = RandomForestClassifier().fit(X_train, y_train)
    models['kidney'] = kidney_model

    # Save models
    for name, model in models.items():
        pickle.dump(model, open(f'{name}_model.pkl', 'wb'))

train_models()

# Streamlit Interface
st.title('Multi-Disease Diagnosis System')

# Input for Disease Selection
disease_option = st.selectbox("Select Disease to Diagnose", ('Diabetes', 'Heart Disease', 'Kidney Disease'))

# Input Fields for Patient Data
def input_features(features):
    return [st.number_input(f) for f in features]

# Prediction Function
def predict_disease(model_path, features):
    model = pickle.load(open(model_path, 'rb'))
    prediction = model.predict([features])
    return 'Positive' if prediction[0] == 1 else 'Negative'

# Feature Inputs Based on Selection
if disease_option == 'Diabetes':
    features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    input_data = input_features(features)
    if st.button('Diagnose'):
        result = predict_disease('diabetes_model.pkl', input_data)
        st.success(f'Diabetes Diagnosis: {result}')

elif disease_option == 'Heart Disease':
    features = ['Age', 'Sex', 'CP', 'Trestbps', 'Chol', 'Fbs', 'Restecg', 'Thalach', 'Exang', 'Oldpeak', 'Slope', 'Ca', 'Thal']
    input_data = input_features(features)
    if st.button('Diagnose'):
        result = predict_disease('heart_model.pkl', input_data)
        st.success(f'Heart Disease Diagnosis: {result}')

elif disease_option == 'Kidney Disease':
    features = ['Age', 'Bp', 'Sg', 'Al', 'Su', 'Rbc', 'Pc', 'Pcc', 'Ba', 'Bgr', 'Bu', 'Sc', 'Sod', 'Pot', 'Hemo', 'Pcv', 'Wbcc', 'Rbcc', 'Htn', 'Dm', 'Cad', 'Appet', 'Pe', 'Ane']
    input_data = input_features(features)
    if st.button('Diagnose'):
        result = predict_disease('kidney_model.pkl', input_data)
        st.success(f'Kidney Disease Diagnosis: {result}')
