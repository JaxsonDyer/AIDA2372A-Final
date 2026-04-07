from flask import Flask, request, jsonify
import mlflow.pyfunc
import pandas as pd
import os

app = Flask(__name__)

# Assuming the model is saved in a local folder during docker build
# Or loaded from MLflow if MLFLOW_TRACKING_URI is set
# For containerization, it's best to package the model inside
model_path = os.getenv("MODEL_PATH", "model/")

model = None
try:
    if os.path.exists(model_path):
        model = mlflow.pyfunc.load_model(model_path)
        print(f"Model loaded from {model_path}")
    else:
        print(f"Warning: Model path {model_path} does not exist.")
except Exception as e:
    print(f"Failed to load model: {e}")


@app.route('/health', methods=['GET'])
def health():
    if model is not None:
        return jsonify({"status": "healthy", "model_loaded": True}), 200
    return jsonify({"status": "unhealthy", "model_loaded": False}), 503


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model is not loaded"}), 503

    try:
        data = request.get_json(force=True)
        # Convert incoming JSON into a DataFrame
        df = pd.DataFrame(data)

        # Predict using loaded MLflow model
        predictions = model.predict(df)

        # Determine probability if available
        # pyfunc abstracts predict, typically returning the class directly
        # for classification, we just return the prediction.
        return jsonify({
            "predictions": predictions.tolist()
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
