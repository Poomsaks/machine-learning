# # ขั้นตอนการติดตั้ง Library:
# # pip install pandas numpy scikit-learn xgboost joblib

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif

# --- 1. การดึงข้อมูล (Data Inspection) ---
# url: ที่อยู่ของไฟล์ข้อมูล Titanic (CSV)
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

# df (DataFrame): ข้อมูลที่โหลดมาจาก CSV
df = pd.read_csv(url)

# การแสดงข้อมูลในไฟล์:
print("--- ตัวอย่างข้อมูลในไฟล์ CSV ---")
print(df.head())    # แสดง 5 แถวแรก
print(df.info())    # แสดงโครงสร้างข้อมูลและประเภทของแต่ละ Column
print(df.describe()) # แสดงค่าสถิติ (ค่าเฉลี่ย, ค่าสูงสุด-ต่ำสุด) ของข้อมูลตัวเลข

# --- 2. การทำความสะอาดข้อมูล (Data Cleaning) ---
# เติมค่าว่างใน Column 'Age' ด้วยค่า Median (ค่ากลาง) เพื่อให้ข้อมูลไม่ขาดหาย
df['Age'] = df['Age'].fillna(df['Age'].median())

# เติมค่าว่างใน Column 'Embarked' ด้วย Mode (ค่าที่มีคนใช้บริการมากที่สุด)
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# le (LabelEncoder): ตัวแปรที่ใช้เปลี่ยนข้อความให้เป็นตัวเลข
le = LabelEncoder()

# แปลง 'Sex' (เพศ) จาก male/female ให้กลายเป็น 1/0
df['Sex'] = le.fit_transform(df['Sex']) 

# แปลง 'Embarked' (ท่าเรือ) จากตัวอักษรให้กลายเป็นตัวเลข 0, 1, 2
df['Embarked'] = le.fit_transform(df['Embarked'])

# df_ml: เลือกเฉพาะตัวแปรที่มีผลต่อการคำนวณ (Feature Selection)
# ลบชื่อ (Name), เลขตั๋ว (Ticket), เบอร์ห้อง (Cabin) ออกเพราะ AI นำไปคำนวณความน่าจะเป็นไม่ได้
df_ml = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])

# --- 3. การเตรียม Features และเป้าหมาย (Features & Target) ---
# X: ข้อมูลปัจจัยนำเข้า (เช่น อายุ, เพศ, ราคาตั๋ว) ที่จะให้ AI ใช้เรียนรู้
X = df_ml.drop('Survived', axis=1)

# y: ข้อมูลคำตอบ (Target) คือ รอด (1) หรือ ไม่รอด (0)
y = df_ml['Survived']

# mi_scores: คะแนนความสัมพันธ์ที่บอกว่าตัวแปรไหนใน X มีผลต่อ y มากที่สุด
mi_scores = mutual_info_classif(X, y, random_state=42)

# --- 4. การแบ่งข้อมูลฝึกสอนและทดสอบ (Train/Test Split) ---
# X_train, y_train: ข้อมูล 70% ที่ใช้สอน AI
# X_test, y_test: ข้อมูล 30% ที่เก็บไว้สอบ AI
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- 5. การสร้างและฝึกสอน Model (Model Training) ---
# [ส่วนเลือก Model]: คุณสามารถเลือกเปิดใช้งาน Model ตัวใดตัวหนึ่งได้โดยการนำ # ออก
# -------------------------------------------------------------------------
# 5.1 K-Nearest Neighbors (โมเดลที่เราเลือกใช้ปัจจุบัน: ทายผลจากกลุ่มเพื่อนบ้านที่ใกล้ที่สุด)
model = KNeighborsClassifier(n_neighbors=5)

# 5.2 Logistic Regression (โมเดลพื้นฐาน: เหมาะสำหรับทำนายผลแบบ 0 หรือ 1)
# from sklearn.linear_model import LogisticRegression
# model = LogisticRegression(max_iter=1000)

# 5.3 Random Forest (โมเดลยอดนิยม: ใช้ต้นไม้ตัดสินใจหลายๆต้นมาช่วยกันหาคำตอบ)
# from sklearn.ensemble import RandomForestClassifier
# model = RandomForestClassifier(n_estimators=100, random_state=42)

# 5.4 Support Vector Machine (โมเดลทรงพลัง: ใช้การขีดเส้นแบ่งกลุ่มข้อมูลให้ห่างกันมากที่สุด)
# from sklearn.svm import SVC
# model = SVC()
# -------------------------------------------------------------------------

# kf (KFold): วิธีการแบ่งข้อมูลเป็น 5 ส่วนสลับกันตรวจ เพื่อหาค่าเฉลี่ยความแม่นยำ (Cross Validation)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# cv_scores: ผลคะแนนความแม่นยำจากการทดสอบย่อย 5 รอบ
cv_scores = cross_val_score(model, X_train, y_train, cv=kf)

# คำสั่งสอน AI ให้เรียนรู้รูปแบบข้อมูลจริง
model.fit(X_train, y_train)

# --- 6. การสรุปผลและส่งออก (Evaluation & Export) ---
# final_score: คะแนนความแม่นยำสุดท้ายที่ได้จากการทำข้อสอบไล่ (X_test)
final_score = model.score(X_test, y_test)

print(f"\nความแม่นยำเฉลี่ย (Cross-Validation Score): {np.mean(cv_scores)*100:.2f}%")
print(f"ความแม่นยำสอบไล่ (Final Test Accuracy): {final_score*100:.2f}%")

# joblib_filename: ชื่อไฟล์สำหรับบันทึกโมเดลที่เรียนเสร็จแล้ว
joblib_filename = "titanic_knn_model.joblib"

# บันทึกโมเดลลงไฟล์เพื่อนำไปใช้งานจริงในระบบอื่นๆ
joblib.dump(model, joblib_filename)
print(f"\nบันทึกไฟล์ Model เรียบร้อยในชื่อ: {joblib_filename}")