import streamlit as st
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib 
import numpy as np

# Page configuration
st.set_page_config(page_title="Iris Species Predictor", layout="wide")

# Title and description
st.markdown("<h1 style='text-align: center; color: darkblue;'>🌸 Iris Species Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter flower measurements to predict the Iris species</p>", unsafe_allow_html=True)

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X, y)

# Create input section
st.subheader("🌼 Input Flower Measurements")
col1, col2 = st.columns(2)

with col1:
    sepal_length = st.slider(f"{feature_names[0]}", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
    sepal_width  = st.slider(f"{feature_names[1]}", min_value=0.0, max_value=10.0, value=3.5, step=0.1)

with col2:
    petal_length = st.slider(f"{feature_names[2]}", min_value=0.0, max_value=10.0, value=1.4, step=0.1)
    petal_width  = st.slider(f"{feature_names[3]}", min_value=0.0, max_value=10.0, value=0.2, step=0.1)

# Button for prediction
st.markdown("<br>", unsafe_allow_html=True)
if st.button("🌟 Predict Iris Species 🌟"):
    input_features = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = rf_model.predict(input_features)
    species = target_names[prediction[0]]
    
    st.success(f"✅ The predicted Iris species is: **{species}**")
    
    # Optional: show input features in a table
    input_df = pd.DataFrame(input_features, columns=feature_names)
    st.subheader("Your Input Measurements")
    st.table(input_df)

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Made with ❤️ using Streamlit & Random Forest</p>", unsafe_allow_html=True)
