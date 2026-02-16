from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# โหลด Model ที่เราเตรียมไว้
model = joblib.load("titanic_knn_model.joblib")

@app.route('/')
def index():
    # แสดงหน้าแรก (ฟอร์มกรอกข้อมูล)
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # รับค่าจากฟอร์ม (HTML Form)
            pclass = int(request.form['pclass'])
            sex_input = request.form['sex'].lower()
            age = float(request.form['age'])
            sibsp = int(request.form['sibsp'])
            parch = int(request.form['parch'])
            fare = float(request.form['fare'])
            embarked_input = request.form['embarked'].upper()

            # Preprocessing
            sex = 1 if sex_input == 'male' else 0
            embarked_map = {'C': 0, 'Q': 1, 'S': 2}
            embarked = embarked_map.get(embarked_input, 2)

            # ทำนายผล
            person_data = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])
            prediction = model.predict(person_data)
            probability = model.predict_proba(person_data)

            # สรุปผลลัพธ์เพื่อส่งกลับไปโชว์ที่หน้าเว็บ
            result_text = "รอดชีวิต" if prediction[0] == 1 else "ไม่รอดชีวิต"
            confidence = round(float(probability[0][prediction[0]]) * 100, 2)
            
            return render_template('index.html', 
                                 result=result_text, 
                                 conf=confidence,
                                 p_class="success" if prediction[0] == 1 else "danger")
        except:
            return render_template('index.html', result="เกิดข้อผิดพลาดในการกรอกข้อมูล", p_class="warning")

if __name__ == '__main__':
    app.run(debug=True)