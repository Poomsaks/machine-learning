# pip install flask
from flask import Flask, request, jsonify
import joblib
import numpy as np

# สร้างตัวแปร Flask App
app = Flask(__name__)

# 1. โหลด Model ที่เตรียมไว้
model = joblib.load("titanic_knn_model.joblib")

# 2. สร้าง Route สำหรับรับค่าทำนาย
@app.route('/predict', methods=['GET'])
def predict():
    try:
        pclass = int(request.args.get('pclass'))
        sex_input = request.args.get('sex').lower()
        age = float(request.args.get('age'))
        sibsp = int(request.args.get('sibsp'))
        parch = int(request.args.get('parch'))
        fare = float(request.args.get('fare'))
        embarked_input = request.args.get('embarked').upper()

        # 3. Preprocessing (แปลงข้อมูลเหมือนเดิม)
        sex = 1 if sex_input == 'male' else 0
        embarked_map = {'C': 0, 'Q': 1, 'S': 2}
        embarked = embarked_map.get(embarked_input, 2)

        # 4. เตรียมข้อมูลและทำนาย
        person_data = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])
        prediction = model.predict(person_data)
        probability = model.predict_proba(person_data)

        # 5. ส่งผลลัพธ์กลับเป็น JSON
        result = {
            "survived": int(prediction[0]),
            "status": "รอดชีวิต" if prediction[0] == 1 else "ไม่รอดชีวิต",
            "confidence": round(float(probability[0][prediction[0]]) * 100, 2)
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# รัน Server ที่ Port 5000
if __name__ == '__main__':
    print("--- Titanic API Server is Running! ---")
    app.run(debug=True, port=5000)