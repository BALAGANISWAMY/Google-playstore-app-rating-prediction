from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model and PCA
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('pca.pkl', 'rb') as f:
    pca = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')  # Optional: use this for browser input

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data (from form or JSON)
        data = request.form if request.form else request.get_json()

        # Example expected keys: ['Category', 'Reviews', 'Size', 'Installs', 'Type', 'Price', 'Content_Rating', 'Genres', 'Last_Updated', 'Current_Ver', 'Android_Ver']
        # Assume frontend or preprocessor converts these to numerical values
        feature_keys = ['Category', 'Reviews', 'Size', 'Installs', 'Type', 'Price', 'Content_Rating', 
                        'Genres', 'Last_Updated', 'Current_Ver', 'Android_Ver']  # Update based on your model

        # Convert input to float in correct order
        features = [float(data[key]) for key in feature_keys]
        input_array = np.array([features])

        # Apply PCA transformation
        input_pca = pca.transform(input_array)

        # Predict rating
        predicted_rating = model.predict(input_pca)[0]

        return jsonify({'predicted_rating': round(predicted_rating, 2)})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
