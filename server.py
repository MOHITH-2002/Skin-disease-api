from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model, model_from_json
from tensorflow.keras.optimizers import Adam

import cv2
import numpy as np
import json
import requests
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

# Custom optimizer to handle deprecated 'lr' parameter in older models
def custom_optimizer(lr=0.001):
    return Adam(learning_rate=lr)

# Load the skin disease prediction model with custom optimizer handling
disease_model = load_model('final_vgg1920epochs.h5', custom_objects={'Adam': lambda **kwargs: custom_optimizer(**kwargs)})

# Load human detection model from JSON
with open('model.json', 'r') as file:
    loaded_json_model = file.read()
human_detection_model = model_from_json(loaded_json_model)
human_detection_model.load_weights('model.h5')

# Load disease data from JSON
with open('dat.json', 'r') as file:
    disease_data = json.load(file)
disease_keys = list(disease_data.keys())

@app.route('/predict', methods=['POST'])
def predict():
    image_url = request.json.get('url')
    if not image_url:
        return jsonify({'classified': 'No image detected', 'result': 'No'}), 200
    
    response = requests.get(image_url)
    if response.status_code != 200:
        return jsonify({'classified': 'No image detected', 'result': 'No'}), 200
    
    npimg = np.frombuffer(response.content, np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({'classified': 'No image detected', 'result': 'No'}), 200

    # Resize and prepare the image for human detection
    img_for_human_detection = cv2.resize(image, (224, 224))
    img_for_human_detection = img_for_human_detection.reshape((1, 224, 224, 3)) / 255.0

    human_prediction = human_detection_model.predict(img_for_human_detection)
    human_pred_class = np.argmax(human_prediction)

    if human_pred_class == 2:  # Assuming class '2' is 'no human skin'
        return jsonify({'classified': 'No human skin detected', 'result': 'No'}), 200

    # If human skin is detected, proceed with disease prediction
    img_for_disease_detection = cv2.resize(image, (32, 32)) / 255.0
    prediction = disease_model.predict(np.array([img_for_disease_detection]))
    max_index = prediction.argmax()
    disease_name = disease_keys[max_index]
    details = disease_data.get(disease_name, {})

    # Return disease details
    return jsonify({
        'name': disease_name,
        'description': details.get('description', 'No description available'),
        'symptoms': details.get('symptoms', 'No symptoms available'),
        'causes': details.get('causes', 'No causes available'),
        'treatment': details.get('treatment-1', 'No treatment available'),
        'additional_treatment': details.get('treatment-2', 'No additional treatment available')
    })

if __name__ == '__main__':
    app.run(debug=True)
