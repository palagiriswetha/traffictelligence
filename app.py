from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the ML model and encoder
model = pickle.load(open('model.pkl', 'rb'))
encoder = pickle.load(open('encoder.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Collect form data
        date_time = request.form['date_time']
        temp = float(request.form['temp'])
        rain = float(request.form['rain'])
        snow = float(request.form['snow'])
        clouds = int(request.form['clouds'])
        weather_main = request.form['weather_main']
        weather_description = request.form['weather_description']

        # Preprocess categorical data
        weather_main_enc = encoder.transform([weather_main])[0]
        weather_desc_enc = encoder.transform([weather_description])[0]

        # Feature array
        features = np.array([[temp, rain, snow, clouds, weather_main_enc, weather_desc_enc]])

        # Prediction
        prediction = model.predict(features)
        output = round(prediction[0], 2)

        return render_template('result.html', prediction_text=f'Estimated Traffic Volume: {output}')

if __name__ == "__main__":
    app.run(debug=True)