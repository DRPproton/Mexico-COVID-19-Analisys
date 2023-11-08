from flask import Flask, request, jsonify
import pickle
import pandas as pd

with open('xgb_model.bin', 'rb') as f:
    model = pickle.load(f)

app = Flask('midterm')

@app.route('/predict', methods=['POST'])
def predict():
    client = request.get_json()

    X = pd.DataFrame([client])
    y_pred = model.predict_proba(X)[0,1]
    died = y_pred>= 0.5

    result = {
        "Probability of dying from COVID-19": round(float(y_pred), 3),
        "Patient died": bool(died)
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)