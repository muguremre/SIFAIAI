from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
import tensorflow.keras.backend as K
import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
import json
import os
import cv2
import base64
from PIL import Image
from flask_cors import CORS
import traceback

app = Flask(__name__)
CORS(app)

os.makedirs("temp", exist_ok=True)

# === Ortak ===
def format_confidence(value):
    return "approximately 99%" if value >= 0.99 else f"{value * 100:.0f}%"

# ============ BEYİN TÜMÖRÜ MODELLERİ ============

def tversky(y_true, y_pred, alpha=0.7, beta=0.3):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    TP = K.sum(y_true_f * y_pred_f)
    FN = K.sum(y_true_f * (1 - y_pred_f))
    FP = K.sum((1 - y_true_f) * y_pred_f)
    return (TP + 1e-6) / (TP + alpha * FN + beta * FP + 1e-6)

def focal_tversky(y_true, y_pred, alpha=0.7, beta=0.3, gamma=0.75):
    t_index = tversky(y_true, y_pred, alpha, beta)
    return K.pow((1 - t_index), gamma)

mr_model = xgb.XGBClassifier()
mr_model.load_model('xgboost_model.json')

anamnez_model_brain = xgb.XGBClassifier()
anamnez_model_brain.load_model('anamnez_model.json')

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)
categories_brain = list(label_encoder.classes_)

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

seg_model = load_model("seg_model.h5", custom_objects={
    "focal_tversky": focal_tversky,
    "tversky": tversky
})

def process_mr_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

def segment_image(img_path):
    img = cv2.imread(img_path)
    resized = cv2.resize(img, (256, 256))
    norm_img = resized / 255.0
    input_img = np.expand_dims(norm_img, axis=0)

    mask = seg_model.predict(input_img)[0]
    mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))
    mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255

    overlay = img.copy()
    overlay[mask_binary == 255] = [0, 255, 0]

    _, buffer = cv2.imencode('.png', overlay)
    return base64.b64encode(buffer).decode('utf-8')

@app.route('/predict/brain', methods=['POST'])
def predict_brain():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    file_path = os.path.join('temp', file.filename)
    file.save(file_path)

    try:
        features = base_model.predict(process_mr_image(file_path)).flatten().reshape(1, -1)
        mr_pred_proba = mr_model.predict_proba(features)[0]

        mr_top_idx = int(np.argmax(mr_pred_proba))
        mr_top_label = categories_brain[mr_top_idx]
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
            anamnez_proba = anamnez_model_brain.predict_proba([anamnez_features])[0]
            anamnez_top_idx = int(np.argmax(anamnez_proba))
            anamnez_top_label = categories_brain[anamnez_top_idx]
            anamnez_confidence = float(anamnez_proba[anamnez_top_idx])
        else:
            anamnez_proba = np.ones(len(categories_brain)) / len(categories_brain)
            anamnez_top_label = "Unknown"
            anamnez_confidence = 0.25

        combined_proba = (mr_pred_proba + anamnez_proba) / 2
        final_idx = int(np.argmax(combined_proba))
        final_label = categories_brain[final_idx]
        combined_confidence = float(combined_proba[final_idx])

        explanation = f"The image analysis model suggests a {mr_top_label} diagnosis with {format_confidence(mr_confidence)} confidence. "
        if anamnez_json:
            explanation += f"The anamnesis evaluation supports a {anamnez_top_label} diagnosis with {format_confidence(anamnez_confidence)} confidence. "
        else:
            explanation += "No anamnesis data provided. "
        explanation += f"Final diagnosis: **{final_label}** with {format_confidence(combined_confidence)} confidence."
        if combined_confidence < 0.6:
            explanation += " However, this prediction has low confidence. Further tests are recommended."

        segmentation_image_base64 = segment_image(file_path)
        os.remove(file_path)

        return jsonify({
            "tumor_type": final_label,
            "güven": f"{combined_confidence * 100:.0f}%",
            "mr_tahmini": mr_top_label,
            "mr_güven": f"{mr_confidence * 100:.0f}%",
            "anamnez_tahmini": anamnez_top_label,
            "anamnez_güven": f"{anamnez_confidence * 100:.0f}%",
            "yorum": explanation,
            "segmentation_image_base64": segmentation_image_base64
        }), 200

    except Exception as e:
        traceback.print_exc()
        os.remove(file_path)
        return jsonify({'error': str(e)}), 500

# ============ CİLT KANSERİ MODELLERİ ============

CNN_MODEL = load_model("improved_skin_cancer_model.h5")
with open("anamnez_modeli.pkl", "rb") as f:
    ANAMNEZ_MODEL_SKIN = pickle.load(f)

IMG_SIZE = (180, 180)

def hazirla_gorsel(img_path):
    img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
    img_array = np.asarray(img).astype(np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)
def parse_swagger_anamnez(form):
    return {
        "kasinti_kanama": int(form.get("Anamnez.kasinti_kanama", 0)),
        "ailede_kanser": int(form.get("Anamnez.ailede_kanser", 0)),
        "kenar_duzensiz": int(form.get("Anamnez.kenar_duzensiz", 0)),
        "gunes_maruz": int(form.get("Anamnez.gunes_maruz", 0)),
        "tedavi_alindi": int(form.get("Anamnez.tedavi_alindi", 0)),
        "renk_degisti": int(form.get("Anamnez.renk_degisti", 0)),
        "fark_suresi": int(form.get("Anamnez.fark_suresi", 0)),
        "lezyon_kabuk": int(form.get("Anamnez.lezyon_kabuk", 0)),
        "ten_rengi": int(form.get("Anamnez.ten_rengi", 0)),
        "bolge": int(form.get("Anamnez.bolge", 0)),
        "travma_sonrasi": int(form.get("Anamnez.travma_sonrasi", 0))
    }

def ensemble_raporu(skor):
    if skor < 0.3:
        return "Düşük Risk", "Risk düşüktür. Cilt değişikliklerini izlemeye devam edin."
    elif skor < 0.6:
        return "Orta Risk", "Risk orta seviyededir. Uzman görüşü alınmalıdır."
    elif skor < 0.8:
        return "Yüksek Risk", "Yüksek risk tespit edildi. Dermatologa başvurun."
    else:
        return "Çok Yüksek Risk", "Ciddi risk mevcut. Acilen tıbbi değerlendirme yapılmalıdır."

@app.route('/predict/skin', methods=['POST'])
def predict_skin():
    if 'image' not in request.files:
        return jsonify({'error': 'Görsel bulunamadı'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Dosya adı boş'}), 400

    file_path = os.path.join('temp', file.filename)
    file.save(file_path)

    try:
        # ✅ Anamnez verisini hem JSON hem Swagger form-data şeklinde destekle
        anamnez_json = request.form.get("anamnez_data")
        if anamnez_json:
            anamnez_data = json.loads(anamnez_json)
        else:
            anamnez_data = parse_swagger_anamnez(request.form)

        anamnez_df = pd.DataFrame([anamnez_data])
        girdi = hazirla_gorsel(file_path)
        gorsel_skor = float(CNN_MODEL.predict(girdi).ravel()[0])
        anamnez_skor = float(ANAMNEZ_MODEL_SKIN.predict_proba(anamnez_df)[0][1])

        ensemble_skor = round(0.7 * gorsel_skor + 0.3 * anamnez_skor, 2)
        sonuc = "Malignant (Kötü Huylu)" if ensemble_skor >= 0.5 else "Benign (İyi Huylu)"
        risk_seviye, yorum = ensemble_raporu(ensemble_skor)

        os.remove(file_path)

        return jsonify({
            "gorsel_skor": f"{gorsel_skor:.2f}",
            "anamnez_skor": f"{anamnez_skor:.2f}",
            "ensemble_skor": f"{ensemble_skor:.2f}",
            "tahmin": sonuc,
            "risk": risk_seviye,
            "yorum": yorum
        }), 200

    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
