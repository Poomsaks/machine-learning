import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import RidgeClassifier  # เปลี่ยนมาใช้ Ridge
from sklearn.preprocessing import LabelEncoder, StandardScaler # เพิ่ม StandardScaler
from sklearn.feature_selection import mutual_info_classif

# --- 1. การดึงข้อมูล (Data Inspection) ---
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# --- 2. การทำความสะอาดข้อมูล (Data Cleaning) ---
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex']) 
df['Embarked'] = le.fit_transform(df['Embarked'])

# ลบ Column ที่ไม่จำเป็นต่อการคำนวณออก
df_ml = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])

# --- 3. การเตรียม Features และเป้าหมาย (Features & Target) ---
X = df_ml.drop('Survived', axis=1)
y = df_ml['Survived']

# --- [เพิ่มเติม] การปรับสเกลข้อมูล (Feature Scaling) ---
# สำคัญมากสำหรับ Ridge: ทำให้ตัวเลขที่มีค่ามาก (เช่น Fare) ไม่ข่มตัวเลขที่มีค่าน้อย (เช่น Age)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 4. การแบ่งข้อมูลฝึกสอนและทดสอบ ---
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# --- 5. การสร้างและฝึกสอน Model (Ridge Classifier) ---
# alpha: คือค่าควบคุมความซับซ้อน (Regularization) ป้องกัน Model จำคำตอบจนเกินไป (Overfitting)
model = RidgeClassifier(alpha=1.0)

# การทดสอบความแม่นยำเบื้องต้นด้วย Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train, y_train, cv=kf)

# ฝึกสอนโมเดลด้วยข้อมูลฝึกสอน
model.fit(X_train, y_train)

# --- 6. การสรุปผลและส่งออก ---
final_score = model.score(X_test, y_test)

print("--- ผลลัพธ์การใช้งาน Ridge Classifier ---")
print(f"ความแม่นยำเฉลี่ย (Cross-Validation): {np.mean(cv_scores)*100:.2f}%")
print(f"ความแม่นยำสอบไล่ (Final Test Accuracy): {final_score*100:.2f}%")

# บันทึกโมเดล
joblib_filename = "titanic_ridge_model.joblib"
joblib.dump(model, joblib_filename)
print(f"\nบันทึกไฟล์ Model เรียบร้อยในชื่อ: {joblib_filename}")