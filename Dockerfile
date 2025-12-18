# ใช้ Python 3.10 ที่เสถียรกับ Tensorflow 2.12
FROM python:3.10

# ตั้งค่า Working Directory
WORKDIR /code

# ติดตั้ง libGL สำหรับ OpenCV (สำคัญมาก เพราะ OpenCV มักถามหาไฟล์นี้ใน Linux)
RUN apt-get update && apt-get install -y libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*

# Copy requirements และติดตั้ง Library
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy ไฟล์ทั้งหมดในโปรเจกต์เข้าไป
COPY . .

# สั่งรันโปรแกรม
CMD ["python", "app.py"]
