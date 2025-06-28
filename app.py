from flask import Flask, request, jsonify, send_from_directory
import numpy as np
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

# Base directory where app.py and npy files are located
base_path = os.path.dirname(os.path.abspath(__file__))

# Load model weights and biases
W1 = np.load(os.path.join(base_path, "W1.npy"))
b1 = np.load(os.path.join(base_path, "b1.npy"))
W2 = np.load(os.path.join(base_path, "W2.npy"))
b2 = np.load(os.path.join(base_path, "b2.npy"))
W3 = np.load(os.path.join(base_path, "W3.npy"))
b3 = np.load(os.path.join(base_path, "b3.npy"))

print("✅ W1 loaded, shape:", W1.shape, "Sample:", W1[0][:5])

# ReLU and softmax activation functions
def relu(Z):
    return np.maximum(0, Z)

def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

# Neural network forward pass
def forward_pass(X):
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = relu(Z2)
    Z3 = np.dot(A2, W3) + b3
    A3 = softmax(Z3)
    return A3

# Serve index.html at root
@app.route("/")
def serve_homepage():
    return send_from_directory(base_path, "index.html")

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json.get("image", [])
        if not data or len(data) != 784:
            return jsonify({"error": "Invalid image data"}), 400

        X = np.array(data).reshape(1, -1)
        prediction = forward_pass(X)
        predicted_digit = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        print("🧪 Input sum:", np.sum(X))
        print("🔎 First 10 values of input:", X[0][:10])
        print("📈 Softmax output:", prediction)
        print("🎯 Predicted:", predicted_digit)

        return jsonify({
            "digit": predicted_digit,
            "confidence": round(confidence * 100, 2),
            "probabilities": prediction.tolist()[0]
        })

    except Exception as e:
        print("❌ Error:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
