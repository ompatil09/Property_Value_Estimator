from flask import Flask, render_template, request
import pandas as pd
import pickle
import os

app = Flask(__name__)

# Load the dataset and model outside the routes to avoid reloading them on every request
data_path = 'final_dataset.csv'
model_path = 'RidgeModel.pkl'

# Error handling for loading the dataset and model
try:
    if os.path.exists(data_path):
        data = pd.read_csv(data_path)
    else:
        raise FileNotFoundError(f"File '{data_path}' not found.")
except Exception as e:
    print(f"Error loading dataset: {e}")
    data = None

try:
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            pipe = pickle.load(f)
    else:
        raise FileNotFoundError(f"File '{model_path}' not found.")
except Exception as e:
    print(f"Error loading model: {e}")
    pipe = None

@app.route('/')
def index():
    if data is not None:
        bedrooms = sorted(data['beds'].unique())
        bathrooms = sorted(data['baths'].unique())
        sizes = sorted(data['size'].unique())
        zip_codes = sorted(data['zip_code'].unique())
    else:
        bedrooms, bathrooms, sizes, zip_codes = [], [], [], []

    return render_template('index.html', bedrooms=bedrooms, bathrooms=bathrooms, sizes=sizes, zip_codes=zip_codes)

@app.route('/predict', methods=['POST'])
def predict():
    if pipe is not None:
        # Get form data
        bedrooms = request.form.get('beds')
        bathrooms = request.form.get('baths')
        size = request.form.get('size')
        zipcode = request.form.get('zip_code')

        # Create a DataFrame with the input data
        input_data = pd.DataFrame([[bedrooms, bathrooms, size, zipcode]],
                                   columns=['beds', 'baths', 'size', 'zip_code'])

        # Convert 'baths' column to numeric with errors='coerce'
        input_data['baths'] = pd.to_numeric(input_data['baths'], errors='coerce')

        # Convert input data to numeric types
        input_data = input_data.astype({'beds': int, 'baths': float, 'size': float, 'zip_code': int})

        # Handle unknown categories in the input data
        for column in input_data.columns:
            unknown_categories = set(input_data[column]) - set(data[column].unique())
            if unknown_categories:
                # Handle unknown categories (e.g., replace with a default value)
                input_data[column] = input_data[column].replace(unknown_categories, data[column].mode()[0])

        # Predict the price
        prediction = pipe.predict(input_data)[0]
        usd_to_inr_conversion_rate = 75.0
        prediction_inr = prediction * usd_to_inr_conversion_rate
        return str(prediction_inr)
    else:
        return "Model not available"

if __name__ == "__main__":
    app.run(debug=True, port=5000)
