from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# โหลด Model
model = joblib.load("D:\machine learning\dataset16-02-2563\knn_model.joblib")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # รับข้อมูล JSON จาก cURL
        data = request.get_json()
        
        # ดึงค่าและเรียงลำดับให้ตรงกับที่โมเดลต้องการ (15 features)
        features = [
            data['bps'], data['bpd'], data['bw'], data['height'],
            data['fbs'], data['bmi'], data['tg'], data['hdl'],
            data['creatinine'], data['hba1c'], data['fh'], data['waist'],
            data['smoking_type_id'], data['drinking_type_id'], data['egfr']
        ]
        
        # แปลงเป็น Array และทำนาย
        input_array = np.array([features])
        prediction = model.predict(input_array)
        probability = model.predict_proba(input_array)

        # ส่งผลลัพธ์กลับเป็น JSON
        return jsonify({
            "status": "success",
            "prediction": int(prediction[0]),
            "result": "มีความเสี่ยง" if prediction[0] == 1 else "ปกติ",
            "confidence": f"{round(np.max(probability) * 100, 2)}%"
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=6000)