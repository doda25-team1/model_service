"""
Flask API of the SMS Spam detection model model.
"""
import joblib
import os
import requests
import time
import logging
from functools import wraps
from flask import Flask, jsonify, request, g
from flasgger import Swagger
import pandas as pd

from text_preprocessing import prepare, _extract_message_len, _text_process

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def timed_request(f):
    """Decorator to log request timing and details."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        start_time = time.time()
        response = f(*args, **kwargs)
        duration_ms = (time.time() - start_time) * 1000
        
        logger.info(
            f"{request.method} {request.path} - "
            f"Status: {response[1] if isinstance(response, tuple) else 200} - "
            f"Duration: {duration_ms:.2f}ms"
        )
        return response
    return decorated_function


# ========== Load Training Model ========== #

# load env values
MODEL_DIR = "output"
MODEL_VERSION = os.getenv("MODEL_VERSION")
MODEL_BASE_URL = os.getenv("MODEL_BASE_URL", "")

os.makedirs(MODEL_DIR, exist_ok=True)

if MODEL_VERSION:
    MODEL_FILENAME = f"model-{MODEL_VERSION}.joblib"
else:
    raise RuntimeError(f"MODEL_VERSION not specified.")

MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor.joblib")

def _download_file(url, dest):
    if os.path.exists(dest):
        return
    if not url:
        raise RuntimeError(f"File {dest} not found and no download URL configured.")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    with open(dest, "wb") as f:
        f.write(resp.content)

def _ensure_model_files():
    if os.path.exists(MODEL_PATH) and os.path.exists(PREPROCESSOR_PATH):
        return
    
    if not MODEL_VERSION or not MODEL_BASE_URL:
        raise RuntimeError(
            "Model files missing and MODEL_VERSION / MODEL_BASE_URL are not set. "
            "Either mount the files into output/ or configure download url."
        )
    
    tag = f"model-v{MODEL_VERSION}"
    model_url = f"{MODEL_BASE_URL}/{tag}/{MODEL_FILENAME}"
    preproc_url = f"{MODEL_BASE_URL}/{tag}/preprocessor.joblib"

    _download_file(model_url, MODEL_PATH)
    _download_file(preproc_url, PREPROCESSOR_PATH)

# download into /app/output if needed; then load model
_ensure_model_files()
model = joblib.load(MODEL_PATH)

# ========== Create App ========== #
app = Flask(__name__)
swagger = Swagger(app)

@app.route('/health', methods=['GET'])
@timed_request
def health():
    """
    Liveness probe endpoint.
    Returns 200 OK if the application is running.
    ---
    responses:
      200:
        description: Service is alive
    """
    return jsonify({"status": "UP", "service": "model-service"}), 200


@app.route('/ready', methods=['GET'])
@timed_request
def ready():
    """
    Readiness probe endpoint.
    Returns 200 OK if the model is loaded and ready to serve predictions.
    ---
    responses:
      200:
        description: Service is ready
      503:
        description: Service is not ready (model not loaded)
    """
    # Check if model files are loaded
    if os.path.exists(MODEL_PATH) and os.path.exists(PREPROCESSOR_PATH):
        return jsonify({
            "status": "READY",
            "model": "loaded",
            "version": MODEL_VERSION,
            "modelPath": MODEL_PATH
        }), 200
    else:
        return jsonify({
            "status": "NOT_READY",
            "model": "not loaded",
            "version": MODEL_VERSION,
            "modelPath": MODEL_PATH
        }), 503


@app.route('/predict', methods=['POST'])
@timed_request
def predict():
    """
    Predict whether an SMS is Spam.
    ---
    consumes:
      - application/json
    parameters:
        - name: input_data
          in: body
          description: message to be classified.
          required: True
          schema:
            type: object
            required: sms
            properties:
                sms:
                    type: string
                    example: This is an example of an SMS.
    responses:
      200:
        description: "The result of the classification: 'spam' or 'ham'."
    """
    input_data = request.get_json()
    sms = input_data.get('sms')
    processed_sms = prepare(sms)
    prediction = model.predict(processed_sms)[0]

    res = {
        "result": prediction,
        "classifier": "decision tree",
        "sms": sms
    }
    logger.info(f"Prediction: '{sms[:50]}...' -> {prediction}")
    return jsonify(res)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8081))
    app.run(host="0.0.0.0", port=port, debug=True)
