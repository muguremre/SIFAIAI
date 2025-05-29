from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
import numpy as np
import xgboost as xgb
import os
import pickle
import json

app = Flask(__name__)

mr_model = xgb.XGBClassifier()
mr_model.load_model('xgboost_model.json')

anamnez_model = xgb.XGBClassifier()
anamnez_model.load_model('anamnez_model.json')

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
categories = list(label_encoder.classes_)

def process_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def format_confidence(value):
    return "approximately 99%" if value >= 0.99 else f"{value * 100:.0f}%"

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    file_path = os.path.join('temp', file.filename)
    os.makedirs('temp', exist_ok=True)
    file.save(file_path)

    try:
        processed_image = process_image(file_path)
        features = base_model.predict(processed_image).flatten().reshape(1, -1)

        mr_pred_proba = mr_model.predict_proba(features)[0]
        mr_top_idx = int(np.argmax(mr_pred_proba))
        mr_top_label = categories[mr_top_idx]
        mr_confidence = float(mr_pred_proba[mr_top_idx])

        anamnez_json = request.form.get("anamnez_data")
        if anamnez_json:
            anamnez_data = json.loads(anamnez_json)
            anamnez_features = [
                int(anamnez_data["epilepsy"]),
                int(anamnez_data["worsening_headache"]),
                int(anamnez_data["morning_headache"]),
                int(anamnez_data["vision_loss"]),
                int(anamnez_data["hormonal_issues"]),
                int(anamnez_data["family_history"]),
                int(anamnez_data["age"]),
                1 if anamnez_data["gender"].lower() == "male" else 0
            ]
            anamnez_proba = anamnez_model.predict_proba([anamnez_features])[0]
            anamnez_top_idx = int(np.argmax(anamnez_proba))
            anamnez_top_label = categories[anamnez_top_idx]
            anamnez_confidence = float(anamnez_proba[anamnez_top_idx])
        else:
            anamnez_proba = np.ones(len(categories)) / len(categories)
            anamnez_top_label = "Unknown"
            anamnez_confidence = 0.25

        combined_proba = (mr_pred_proba + anamnez_proba) / 2
        final_idx = int(np.argmax(combined_proba))
        final_label = categories[final_idx]
        combined_confidence = float(combined_proba[final_idx])

        explanation = f"The image analysis model suggests a {mr_top_label} diagnosis with {format_confidence(mr_confidence)} confidence. "
        if anamnez_json:
            explanation += f"The anamnesis evaluation supports a {anamnez_top_label} diagnosis with {format_confidence(anamnez_confidence)} confidence. "
        else:
            explanation += "No anamnesis data provided, only image-based diagnosis was considered. "

        explanation += f"Overall, **{final_label}** is the most likely diagnosis with {format_confidence(combined_confidence)} confidence."
        if combined_confidence < 0.6:
            explanation += " However, this prediction has low confidence. Further tests are recommended."

        os.remove(file_path)

        return jsonify({
            "tumor_type": final_label,
            "güven": f"{combined_confidence * 100:.0f}%",
            "mr_tahmini": mr_top_label,
            "mr_güven": f"{mr_confidence * 100:.0f}%",
            "anamnez_tahmini": anamnez_top_label,
            "anamnez_güven": f"{anamnez_confidence * 100:.0f}%",
            "yorum": explanation
        }), 200

    except Exception as e:
        os.remove(file_path)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
