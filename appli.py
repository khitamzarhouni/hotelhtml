from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Charger le modèle et le scaler
model = pickle.load(open('random_forest_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [
        float(request.form['lead_time']),
        float(request.form['stays_in_weekend_nights']),
        float(request.form['stays_in_week_nights']),
        float(request.form['adults']),
        float(request.form['children']),
        float(request.form['babies']),
        float(request.form['adr']),
        float(request.form['total_of_special_requests'])
    ]
    
    # Mise à l'échelle
    features_scaled = scaler.transform([features])

    # Prédiction
    prediction = model.predict(features_scaled)[0]
    result = "❌ Annulée" if prediction == 1 else "✅ Confirmée"

    return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
