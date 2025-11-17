from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import tifffile
import os

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model("model.h5")

def load_and_stack_tiffs(file_paths, img_size=(224, 224)):
    bands = []

    for fp in file_paths:
        img = tifffile.imread(fp)  # load TIFF
        img = cv2.resize(img, img_size)  # resize
        img = img / 255.0  # normalize
        bands.append(img)

    # Stack to (H, W, 19)
    stacked = np.stack(bands, axis=-1)

    # Add batch dimension → (1, H, W, 19)
    return np.expand_dims(stacked, axis=0)


@app.route("/api/predict", methods=["POST"])
def predict():
    if "files" not in request.files:
        return jsonify({"error": "Upload 19 TIFF files with key 'files'"}), 400

    files = request.files.getlist("files")

    if len(files) != 19:
        return jsonify({"error": "Exactly 19 TIFF files required"}), 400

    # Save temporary files
    file_paths = []
    for file in files:
        path = os.path.join("temp", file.filename)
        file.save(path)
        file_paths.append(path)

    try:
        # Preprocess (stack 19 bands)
        input_tensor = load_and_stack_tiffs(file_paths)

        # Predict
        preds = model.predict(input_tensor)[0]

        predicted = float(preds[0])  # yield
        confidence = float(np.max(preds))  # probability

        return jsonify({
            "predictedYield": predicted,
            "confidence": confidence,
            "unit": "kg/ha"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        # Clean up temp files
        for p in file_paths:
            os.remove(p)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
