import os
import joblib
import numpy as np
from flask import Flask, jsonify, request

app = Flask(__name__)

# --- ส่วนโหลด Model ---
model = None
scaler = None

def load_model():
    global model, scaler
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, "knn_model.joblib")
    scaler_path = os.path.join(BASE_DIR, "scaler.pkl")

    print(f"🔍 Looking for Model at: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found at {model_path}")
        return False

    try:
        model = joblib.load(model_path)
        print(f"✅ Model Loaded Successfully")
        
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            print(f"✅ Scaler Loaded Successfully")
        
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

load_model()

@app.route('/')
def home():
    return jsonify({
        "status": "running", 
        "model_loaded": model is not None,
        "features_required": 15
    })

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        data = request.get_json()
        features = None
        
        # Option 1: รับเป็น Array [139, 84, ...]
        if 'data' in data and isinstance(data['data'], list):
            features = data['data']
        
        # Option 2: รับเป็น Object {"bps": 139, "bpd": 84, ...}
        # ✅ แก้ไขบรรทัดนี้ (เดิมเขียนไม่ครบ)
        elif 'features' in data:
            f = data['features']
            features = [
                f.get('bps'),
                f.get('bpd'),
                f.get('bw'),
                f.get('height'),
                f.get('fbs'),
                f.get('bmi'),
                f.get('tg'),
                f.get('hdl'),
                f.get('creatinine'),
                f.get('hba1c'),
                f.get('fh'),
                f.get('waist'),
                f.get('smoking_type_id'),
                f.get('drinking_type_id'),
                f.get('egfr')
            ]
        
        if not features or len(features) != 15:
            return jsonify({
                "error": f"Invalid input. Expected 15 features, got {len(features) if features else 0}"
            }), 400

        input_data = np.array(features).reshape(1, -1)
        
        if scaler is not None:
            input_data = scaler.transform(input_data)

        prediction = model.predict(input_data)[0]
        result_text = "เป็น" if prediction == 1 else "ไม่ เป็น"
        
        confidence = None
        if hasattr(model, 'predict_proba'):
            confidence = float(model.predict_proba(input_data)[0][prediction])

        return jsonify({
            "prediction": result_text,
            "raw_value": int(prediction),
            "confidence": confidence,
            "input_features": features
        })

    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6000)