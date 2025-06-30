from flask import Flask, render_template, request
import joblib 
import numpy as np
import pandas as pd

app = Flask(__name__)


model = joblib.load('model.pkl')
encoder = joblib.load('encoder.pkl')

print("Supported labels (encoder.classes_):")
print(encoder.classes_)

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

        # Optional: validate weather values
        if weather_main not in encoder.classes_ or weather_description not in encoder.classes_:
            return f"❌ Error: Please enter valid weather values. Supported: {list(encoder.classes_)}"

        # Preprocess categorical data
        weather_main_enc = encoder.transform([weather_main])[0]
        weather_desc_enc = encoder.transform([weather_description])[0]

        # Feature array
        features = np.array([[temp, rain, snow, clouds]])

        # Prediction
        prediction = model.predict(features)
        output = round(prediction[0], 2)

        return render_template('result.html', prediction_text=f'Estimated Traffic Volume is : {output}')

if __name__ == "__main__":
    app.run(debug=True)