# 1. ใช้ Python เป็น Base Image
FROM python:3.10-slim

# 2. ตั้งค่า Working Directory
WORKDIR /app

# 3. ติดตั้ง Library ที่จำเป็น
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. คัดลอกไฟล์โปรเจกต์ทั้งหมดเข้าไปใน Image
# (รวมถึง knn_model.joblib จะถูกฝังไปด้วย)
COPY . .

# 5. เปิด Port 6000
EXPOSE 6000

# 6. รันด้วย Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:6000", "--workers", "4", "app:app"]