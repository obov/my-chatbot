# NVIDIA의 PyTorch 베이스 이미지 사용 (CUDA 11.7, PyTorch 2.0)
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 의존성 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성 설치
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# 코드 복사
COPY . .

# 훈련 스크립트 실행 (기본 명령 설정)
CMD ["python", "train.py"]