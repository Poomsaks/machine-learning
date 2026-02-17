import os
import joblib
import numpy as np
from flask import Flask, jsonify, request
from pathlib import Path

# --- ตั้งค่า Flask ---
app = Flask(__name__)

# --- การจัดการ Path (แก้ปัญหาหา Model ไม่เจอ) ---
# ดึงตำแหน่ง Folder ปัจจุบันที่ไฟล์ app.py นี้วางอยู่
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "knn_model.joblib"
SCALER_PATH = BASE_DIR / "scaler.pkl"

# Global Variables สำหรับเก็บ Model
model = None
scaler = None

def load_ml_assets():
    """ฟังก์ชันสำหรับโหลด Model และ Scaler เมื่อ Start Server"""
    global model, scaler
    
    print("-" * 30)
    print(f"🚀 Starting System at: {BASE_DIR}")
    
    # 1. ตรวจสอบและโหลด Model
    if MODEL_PATH.exists():
        try:
            model = joblib.load(MODEL_PATH)
            print(f"✅ SUCCESS: Loaded Model from {MODEL_PATH}")
        except Exception as e:
            print(f"❌ ERROR: Failed to load model: {e}")
    else:
        print(f"❌ ERROR: Model file NOT FOUND at {MODEL_PATH}")
        print(f"📁 Available files in directory: {os.listdir(BASE_DIR)}")

    # 2. ตรวจสอบและโหลด Scaler (ถ้ามี)
    if SCALER_PATH.exists():
        try:
            scaler = joblib.load(SCALER_PATH)
            print(f"✅ SUCCESS: Loaded Scaler from {SCALER_PATH}")
        except Exception as e:
            print(f"⚠️ WARNING: Found scaler but failed to load: {e}")
    else:
        print(f"ℹ️ INFO: Scaler not found, system will skip scaling.")
    
    print("-" * 30)

# สั่งให้โหลด Assets ทันทีเมื่อโปรแกรมเริ่มทำงาน
load_ml_assets()

# --- API Routes ---

@app.route('/')
def home():
    """Route สำหรับเช็คสถานะของ Server"""
    return jsonify({
        "status": "online",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "api_version": "1.0.0"
    })

@app.route('/predict', methods=['POST'])
def predict():
    # เช็คก่อนว่าโมเดลพร้อมใช้งานไหม
    if model is None:
        return jsonify({
            "error": "Model is not loaded on the server. Please check file path or server logs.",
            "path_searched": str(MODEL_PATH)
        }), 500
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        features = None
        
        # รองรับการรับข้อมูล 2 รูปแบบ (แบบ Array หรือ แบบ Object)
        if 'data' in data and isinstance(data['data'], list):
            features = data['data']
        
        elif 'features' in data:
            f = data['features']
            # เรียงลำดับ Feature ตามที่ Model ต้องการ (15 ตัว)
            feature_keys = [
                'bps', 'bpd', 'bw', 'height', 'fbs', 'bmi', 'tg', 'hdl', 
                'creatinine', 'hba1c', 'fh', 'waist', 'smoking_type_id', 
                'drinking_type_id', 'egfr'
            ]
            try:
                features = [float(f.get(k, 0)) for k in feature_keys]
            except (ValueError, TypeError):
                return jsonify({"error": "Feature values must be numeric"}), 400

        # ตรวจสอบจำนวน Feature (ต้องเป็น 15)
        if not features or len(features) != 15:
            return jsonify({
                "error": f"Invalid input. Expected 15 features, received {len(features) if features else 0}"
            }), 400

        # แปลงข้อมูลเป็น NumPy Array สำหรับการ Predict
        input_data = np.array(features).reshape(1, -1)
        
        # ถ้ามี Scaler ให้ทำการ Transform ข้อมูลก่อน
        if scaler is not None:
            input_data = scaler.transform(input_data)

        # ทำการทำนายผล
        prediction = model.predict(input_data)[0]
        result_text = "เป็น" if prediction == 1 else "ไม่เป็น"
        
        # คำนวณค่าความเชื่อมั่น (ถ้า Model รองรับ)
        confidence = None
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(input_data)[0]
            confidence = float(probs[prediction])

        return jsonify({
            "status": "success",
            "prediction": result_text,
            "raw_value": int(prediction),
            "confidence": confidence,
            "input_received": features
        })

    except Exception as e:
        print(f"❌ Runtime Error: {e}")
        return jsonify({"error": "An internal error occurred during prediction"}), 500

if __name__ == '__main__':
    # รันด้วย Port 6000
    # แนะนำ: ใช้ debug=False ในโปรดักชั่น แต่ใช้ debug=True ตอนแก้ปัญหา
    app.run(host='0.0.0.0', port=6000, debug=True)