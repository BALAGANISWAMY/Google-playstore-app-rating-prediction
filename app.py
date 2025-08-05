import streamlit as st
import numpy as np
import pickle

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load PCA
with open("pca.pkl", "rb") as f:
    pca = pickle.load(f)

st.title("Google Playstore App Rating Predictor")

st.write(
    """
    This app predicts the rating of a Google Play Store app based on its attributes.
    """
)

# Example features you might have trained on:
# please update these to match your model
category = st.number_input("Category (as number encoded)", min_value=0.0)
reviews = st.number_input("Reviews")
size = st.number_input("Size (in MB)")
installs = st.number_input("Installs")
type_ = st.number_input("Type (0=Free, 1=Paid)")
price = st.number_input("Price (0 if free)")
content_rating = st.number_input("Content Rating (encoded)")
genres = st.number_input("Genres (encoded)")
last_updated = st.number_input("Last Updated (numerical)")
current_ver = st.number_input("Current Version (numerical)")
android_ver = st.number_input("Android Version (numerical)")

if st.button("Predict Rating"):
    try:
        features = np.array(
            [
                category,
                reviews,
                size,
                installs,
                type_,
                price,
                content_rating,
                genres,
                last_updated,
                current_ver,
                android_ver,
            ]
        ).reshape(1, -1)

        features_pca = pca.transform(features)
        prediction = model.predict(features_pca)[0]
        st.success(f"Predicted App Rating: {round(prediction, 2)} ⭐️")

    except Exception as e:
        st.error(f"Error: {str(e)}")
