# app.py
import streamlit as st
import joblib
import numpy as np
from sklearn.datasets import load_iris

# Load trained model
model = joblib.load("iris_model.pkl")
iris = load_iris()

st.title("🌸 Iris Flower Classifier")
st.write("This app predicts the type of Iris flower based on features.")

# User input
sepal_length = st.slider("Sepal length (cm)", float(iris.data[:,0].min()), float(iris.data[:,0].max()))
sepal_width  = st.slider("Sepal width (cm)", float(iris.data[:,1].min()), float(iris.data[:,1].max()))
petal_length = st.slider("Petal length (cm)", float(iris.data[:,2].min()), float(iris.data[:,2].max()))
petal_width  = st.slider("Petal width (cm)", float(iris.data[:,3].min()), float(iris.data[:,3].max()))

# Prediction
if st.button("Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Iris type: **{iris.target_names[prediction]}**")
