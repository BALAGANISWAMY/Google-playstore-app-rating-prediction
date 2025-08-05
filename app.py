import streamlit as st
import pickle
import numpy as np

# Load model and PCA
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("pca.pkl", "rb") as f:
    pca = pickle.load(f)

st.title("Google Play Store App Rating Prediction")

st.write("Enter app features below:")

# Only include 9 fields (as used in training)
category = st.number_input("Category (numeric)", min_value=0.0, step=1.0)
reviews = st.number_input("Reviews", min_value=0.0, step=1.0)
size = st.number_input("Size (MB)", min_value=0.0, step=0.1)
installs = st.number_input("Installs", min_value=0.0, step=1000.0)
type_ = st.number_input("Type (Free=0, Paid=1)", min_value=0.0, max_value=1.0, step=1.0)
price = st.number_input("Price", min_value=0.0, step=0.01)
content_rating = st.number_input("Content Rating (numeric)", min_value=0.0, step=1.0)
genres = st.number_input("Genres (numeric)", min_value=0.0, step=1.0)
android_ver = st.number_input("Android Version", min_value=0.0, step=0.1)

if st.button("Predict Rating"):
    try:
        # Only pass 9 features
        input_features = np.array([[category, reviews, size, installs, type_,
                                    price, content_rating, genres, android_ver]])
        
        input_pca = pca.transform(input_features)
        prediction = model.predict(input_pca)[0]
        st.success(f"Predicted Rating: {round(prediction, 2)}")

    except Exception as e:
        st.error(f"Error: {e}")
