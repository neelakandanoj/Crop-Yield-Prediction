from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle

app = Flask(__name__)
CORS(app)

# Load Model + Scaler
with open("..\Crop Predicion\Models\yield_model_oj.pkl", "rb") as f:
    model = pickle.load(f)

with open("..\Crop Predicion\Models\yield_scaler_oj.pkl", "rb") as f:
    scaler = pickle.load(f)

@app.route("/api/predict", methods=["POST"])
def predict():

    data = request.json

    try:
        features = [
            float(data["optical"]["NDVI"]),
            float(data["optical"]["NDRE"]),
            float(data["optical"]["EVI"]),
            float(data["optical"]["SAVI"]),
            float(data["optical"]["FAPAR"]),
            float(data["optical"]["LAI"]),

            float(data["sar"]["VH"]),
            float(data["sar"]["VV"]),
            float(data["sar"]["Entropy"]),
            float(data["sar"]["Alpha"]),
            float(data["sar"]["RVI"]),

            float(data["soil"]["Clay Content"]),
            float(data["soil"]["Nitrogen"]),
            float(data["soil"]["Organic Carbon"]),
            float(data["soil"]["pH"]),

            float(data["weather"]["Solar Radiation"]),
            float(data["weather"]["GDD"]),
            float(data["weather"]["Humidity"]),
            float(data["weather"]["Rainfall"])
        ]

    except Exception as e:
        return jsonify({"error": "Invalid input", "details": str(e)}), 400

    arr = np.array([features])
    arr_scaled = scaler.transform(arr)

    # ---------- Model Prediction ----------
    # Main prediction
    prediction = model.predict(arr_scaled)[0]

    # Confidence estimation from tree variance
    tree_preds = np.array([tree.predict(arr_scaled)[0] for tree in model.estimators_])

    mean_pred = tree_preds.mean()
    std_pred = tree_preds.std()

    if mean_pred != 0:
        confidence = 1 - (std_pred / abs(mean_pred))
    else:
        confidence = 0.5

    # Clamp 0–1
    confidence = max(0, min(confidence, 1))

    # ---------- API Response ----------
    return jsonify({
        "predictedYield": float(prediction),
        "unit": "kgs/hectare",
        "confidence": float(confidence)
    })



if __name__ == "__main__":
    app.run(port=5000, debug=True)
