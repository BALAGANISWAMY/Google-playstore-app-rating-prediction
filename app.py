import streamlit as st
import pickle
import numpy as np

# Load the model and PCA
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("pca.pkl", "rb") as pca_file:
    pca = pickle.load(pca_file)

# Streamlit UI
st.title("📱 Google Play Store App Rating Prediction")

st.write("🔢 Enter the following features below. Only numeric values should be used.")

# Only include 9 inputs as PCA was trained on 9 features
category = st.number_input("Category (encoded)", step=1.0)
reviews = st.number_input("Number of Reviews", step=100.0)
size = st.number_input("App Size (in MB)", step=0.1)
installs = st.number_input("Installs (total downloads)", step=1000.0)
type_ = st.number_input("Type (Free=0, Paid=1)", min_value=0.0, max_value=1.0, step=1.0)
price = st.number_input("Price (₹)", step=0.1)
content_rating = st.number_input("Content Rating (encoded)", step=1.0)
genres = st.number_input("Genres (encoded)", step=1.0)
android_ver = st.number_input("Minimum Android Version (e.g. 4.1)", step=0.1)

# Predict button
if st.button("Predict Rating"):
    try:
        # Create input array
        input_data = np.array([[category, reviews, size, installs, type_,
                                price, content_rating, genres, android_ver]])

        # Apply PCA
        transformed_data = pca.transform(input_data)

        # Predict
        prediction = model.predict(transformed_data)[0]

        st.success(f"🌟 Predicted App Rating: {round(prediction, 2)}")

    except Exception as e:
        st.error(f"❌ Error: {e}")
