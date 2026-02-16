import joblib
import numpy as np

# 1. โหลด Model ที่เราบันทึกไว้
model_name = "titanic_knn_model.joblib"
model = joblib.load(model_name)

print(f"--- ระบบทำนายโอกาสรอดชีวิตจาก Titanic (Model: {model_name}) ---")
print("กรุณากรอกข้อมูลของคุณเพื่อจำลองเหตุการณ์:")

# 2. รับค่าจากผู้ใช้งาน (Input)
# [จุดเน้นในการสอน: ต้องรับค่าให้ครบตามจำนวน Feature ที่เราใช้สอน Model]
pclass = int(input("ชั้นที่นั่ง (1, 2, 3): "))
sex_input = input("เพศ (male/female): ").lower()
age = float(input("อายุ: "))
sibsp = int(input("จำนวนพี่น้อง/คู่สมรสที่เดินทางมาด้วย: "))
parch = int(input("จำนวนพ่อแม่/ลูกที่เดินทางมาด้วย: "))
fare = float(input("ราคาตั๋ว (สมมติ 10-500): "))
embarked_input = input("ท่าเรือที่ขึ้น (C, Q, S): ").upper()

# 3. การแปลงข้อมูล (Preprocessing)
# [สำคัญมาก: ต้องแปลงให้เหมือนกับตอนที่ใช้ LabelEncoder ในไฟล์แรก]
# Sex: female = 0, male = 1
sex = 1 if sex_input == 'male' else 0

# Embarked: C = 0, Q = 1, S = 2 (เรียงตามตัวอักษร)
embarked_map = {'C': 0, 'Q': 1, 'S': 2}
embarked = embarked_map.get(embarked_input, 2) # ถ้ากรอกผิดให้เป็น S (2)

# 4. รวมข้อมูลเป็นรูปแบบที่ Model ต้องการ (List ซ้อน List)
# เรียงลำดับ Column ให้ตรงกับตอน Train: [Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]
person_data = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])

# 5. ทำการทายผล (Prediction)
prediction = model.predict(person_data)
probability = model.predict_proba(person_data) # ดูความน่าจะเป็น (%)

# 6. แสดงผลลัพธ์
print("\n" + "="*30)
if prediction[0] == 1:
    print(f"ผลทำนาย: >>> ยินดีด้วย คุณมีโอกาส 'รอดชีวิต' <<<")
else:
    print(f"ผลทำนาย: >>> เสียใจด้วย คุณอาจ 'ไม่รอดชีวิต' <<<")

# แสดงความมั่นใจของ AI
# probability[0][1] คือความมั่นใจในฝั่ง "รอด"
print(f"ความเชื่อมั่นของ AI: {probability[0][prediction[0]]*100:.2f}%")
print("="*30)