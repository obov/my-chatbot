#!/bin/bash

ACTION=$1

case $ACTION in
    deploy)
        # 의존성 설치
        echo "의존성을 설치합니다..."
        pip install -r requirements.txt --break-system-packages

        # 기존에 실행 중인 앱이 있으면 종료
        echo "기존 앱을 종료합니다..."
        pkill -f 'streamlit run app.py'

        # 앱 실행
        echo "앱을 백그라운드에서 실행합니다..."
        nohup streamlit run app.py --server.port 8501 --server.headless true &

        echo "배포 작업이 완료되었습니다."
        exit
        ;;
    start)
        # 앱 실행
        echo "앱을 백그라운드에서 실행합니다..."
        nohup streamlit run app.py --server.port 8501 --server.headless true &

        echo "시작 후 설정 작업이 완료되었습니다."
        exit
        ;;
    stop)
        # 앱 종료
        echo "앱을 종료합니다..."
        pkill -f 'streamlit run app.py'

        echo "종료 전 설정 작업이 완료되었습니다."
        exit
        ;;
    *)
        echo "사용법: $0 {deploy|start|stop}"
        exit 1
        ;;
esac