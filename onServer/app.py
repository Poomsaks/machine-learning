import os
import joblib
import numpy as np
from flask import Flask, jsonify, request
from pathlib import Path

app = Flask(__name__)

# --- การจัดการ Path ---
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "knn_model.joblib"

model = None

def load_ml_assets():
    global model
    print("-" * 30)
    if MODEL_PATH.exists():
        try:
            model = joblib.load(MODEL_PATH)
            print(f"✅ SUCCESS: Loaded Model from {MODEL_PATH}")
        except Exception as e:
            print(f"❌ ERROR: Failed to load model: {e}")
    else:
        print(f"❌ ERROR: Model file NOT FOUND at {MODEL_PATH}")
        print(f"📁 Files available: {os.listdir(BASE_DIR)}")
    print("-" * 30)

load_ml_assets()

@app.route('/')
def home():
    return jsonify({
        "status": "online",
        "model_loaded": model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded", "path": str(MODEL_PATH)}), 500
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        # ดึงข้อมูลจาก 'data' หรือ 'features'
        features = data.get('data') or data.get('features')

        # ถ้าส่งมาเป็น Dictionary (Object) ให้ดึงค่าตาม Key
        if isinstance(features, dict):
            feature_keys = [
                'bps', 'bpd', 'bw', 'height', 'fbs', 'bmi', 'tg', 'hdl', 
                'creatinine', 'hba1c', 'fh', 'waist', 'smoking_type_id', 
                'drinking_type_id', 'egfr'
            ]
            features = [features.get(k, 0) for k in feature_keys]

        # ตรวจสอบความถูกต้อง
        if not features or len(features) != 15:
            return jsonify({"error": f"Need 15 features, got {len(features) if features else 0}"}), 400

        # Predict (ใส่ข้อมูลดิบเข้า Model ทันที)
        input_data = np.array(features, dtype=float).reshape(1, -1)
        prediction = model.predict(input_data)[0]
        
        # คำนวณความมั่นใจ (ถ้ามี)
        confidence = None
        if hasattr(model, 'predict_proba'):
            confidence = float(np.max(model.predict_proba(input_data)))

        return jsonify({
            "status": "success",
            "prediction": "เป็น" if prediction == 1 else "ไม่เป็น",
            "raw_value": int(prediction),
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6000, debug=True)