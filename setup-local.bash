#!/bin/bash

# 가상 환경 생성
python3 -m venv venv || python -m venv venv

# 가상 환경 활성화
source venv/bin/activate

# 패키지 설치
pip3 install -r requirements.txt || pip install -r requirements.txt