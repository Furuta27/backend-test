# ai-backend-server/app.py

import os
import io
import json
import torch
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app) # 모든 도메인에서의 요청을 허용 (개발용)

# --- 모델 로딩 ---
# yolov5s.pt 또는 yolov5s3.torchscript.ptl 파일이 ai-backend-server 폴더에 있어야 합니다.
# 모델 로딩 방식은 사용하는 모델의 형식에 따라 달라질 수 있습니다.
# 여기서는 일반적인 YOLOv5 PyTorch 모델 로딩 방식을 사용합니다.
# 만약 torchscript (.ptl) 파일을 사용한다면, torch.jit.load를 사용해야 합니다.

MODEL_PATH = 'best.torchscript.ptl' # 또는 'yolov5s3.torchscript.ptl'
model = None

try:
    # PyTorch Hub에서 미리 학습된 YOLOv5s 모델 로드
    # (인터넷 연결 필요, 초기 1회 다운로드)
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    print("YOLOv5 model loaded from PyTorch Hub.")
except Exception as e:
    print(f"Error loading model from PyTorch Hub: {e}")
    print(f"Attempting to load model from local path: {MODEL_PATH}")
    try:
        if MODEL_PATH.endswith('.pt'):
            # 로컬 .pt 파일 로드
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=False, path=MODEL_PATH)
            print(f"YOLOv5 model loaded from local .pt file: {MODEL_PATH}")
        elif MODEL_PATH.endswith('.ptl') or MODEL_PATH.endswith('.torchscript'):
            # 로컬 torchscript (.ptl) 파일 로드
            model = torch.jit.load(MODEL_PATH)
            model.eval() # 추론 모드 설정
            print(f"YOLOv5 torchscript model loaded from local .ptl file: {MODEL_PATH}")
        else:
            print(f"Unsupported model file extension: {MODEL_PATH}. Only .pt or .ptl/.torchscript are supported.")
    except Exception as e:
        print(f"Failed to load model from local path {MODEL_PATH}: {e}")
        model = None # 모델 로드 실패 시 None으로 설정

if model is None:
    print("CRITICAL: AI model could not be loaded. Detection will not work.")
    
# 모델을 GPU (CUDA)가 사용 가능하다면 GPU로 옮깁니다.
if torch.cuda.is_available():
    model.cuda()
    print("Model moved to GPU.")
else:
    print("CUDA not available. Model running on CPU.")

# --- 라벨 파일 로딩 ---
# YOLOv5의 기본 라벨 (예: 'person', 'car') 파일 경로
# yolov5s.pt 모델 사용 시 기본적으로 model.names 속성에 포함됩니다.
# 커스텀 학습 모델이라면 해당 models 폴더의 *.yaml 파일에서 names를 참조하거나
# 별도의 labels.txt 파일을 만들어 사용합니다.

CLASS_LABELS_PATH = 'labels.txt' # 예시: COCO 데이터셋 라벨
JUNK_LABELS_PATH = 'junk.txt'    # Android 프로젝트에서 가져온 필터링 라벨

ALL_MODEL_CLASS_NAMES = [] # 모델이 감지할 수 있는 모든 클래스 이름
JUNK_NAMES_TO_FILTER = []  # junk.txt에 정의된 필터링할 클래스 이름들

# 모델에서 직접 라벨을 가져오는 것이 가장 정확합니다.
if model and hasattr(model, 'names'):
    ALL_MODEL_CLASS_NAMES = model.names
    print(f"Loaded {len(ALL_MODEL_CLASS_NAMES)} class names directly from model.")
else:
    # 모델에서 라벨을 가져올 수 없을 경우 labels.txt에서 로드 시도
    try:
        with open(CLASS_LABELS_PATH, 'r', encoding='utf-8') as f:
            ALL_MODEL_CLASS_NAMES = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(ALL_MODEL_CLASS_NAMES)} class labels from {CLASS_LABELS_PATH}")
    except FileNotFoundError:
        print(f"Warning: {CLASS_LABELS_PATH} not found. Using placeholder labels for detection output.")
        ALL_MODEL_CLASS_NAMES = [f'class_{i}' for i in range(80)] # COCO 데이터셋 기준 기본값

# junk.txt 라벨 로딩
try:
    with open(JUNK_LABELS_PATH, 'r', encoding='utf-8') as f:
        JUNK_NAMES_TO_FILTER = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(JUNK_NAMES_TO_FILTER)} junk labels from {JUNK_LABELS_PATH}")
except FileNotFoundError:
    print(f"Warning: {JUNK_LABELS_PATH} not found. No junk filtering will be applied.")
    JUNK_NAMES_TO_FILTER = []

# JUNK_NAMES_TO_FILTER에 해당하는 클래스들의 인덱스를 미리 찾아둡니다.
# 필터링할 클래스 이름이 ALL_MODEL_CLASS_NAMES에 존재해야 합니다.
JUNK_CLASS_INDICES_TO_FILTER = [
    ALL_MODEL_CLASS_NAMES.index(name)
    for name in JUNK_NAMES_TO_FILTER
    if name in ALL_MODEL_CLASS_NAMES
]
print(f"Junk class indices to filter: {JUNK_CLASS_INDICES_TO_FILTER}")


# --- REST API 엔드포인트 ---

@app.route('/api/detect/image', methods=['POST'])
def detect_object_from_image():
    if model is None:
        return jsonify({'error': 'AI model not loaded. Please check server logs.'}), 500

    if 'image_file' not in request.files:
        return jsonify({'error': 'No image_file part in the request'}), 400

    image_file = request.files['image_file']
    if image_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # 이미지 로드
        img = Image.open(io.BytesIO(image_file.read())).convert("RGB")

        # 모델 추론
        # model(img, size=...)에서 img가 RGB PIL Image 객체여야 합니다.
        results = model(img, size=640) # size는 모델 입력 크기에 맞춰 조정 (예: YOLOv5 기본 640)

        # 판다스 DataFrame으로 결과 얻기 (xyxy[0]은 첫 번째 이미지의 결과)
        detections_df = results.pandas().xyxy[0]

        processed_detections = []
        for _, row in detections_df.iterrows():
            class_index = int(row['class'])
            confidence_score = float(row['confidence'])

            # 🟢 junk.txt 기반 필터링 적용 (해당 클래스 인덱스가 JUNK_CLASS_INDICES_TO_FILTER에 없으면 무시)
            # 즉, junk.txt에 있는 클래스만 통과시킵니다.
            if JUNK_NAMES_TO_FILTER and class_index not in JUNK_CLASS_INDICES_TO_FILTER:
                continue 
            
            # 신뢰도(score) 임계값 적용 (예시: 25% 미만 신뢰도는 무시)
            if confidence_score < 0.25: 
                continue

            processed_detections.append({
                'classIndex': class_index,
                'score': confidence_score,
                'rect': {
                    'left': float(row['xmin']),
                    'top': float(row['ymin']),
                    'right': float(row['xmax']),
                    'bottom': float(row['ymax'])
                }
            })

        # 필터링된 결과와 함께 모델의 모든 클래스 이름을 프론트엔드로 전달합니다.
        # 프론트엔드는 이 ALL_MODEL_CLASS_NAMES를 사용하여 classIndex를 실제 라벨로 변환합니다.
        return jsonify({
            'detections': processed_detections,
            'class_names': ALL_MODEL_CLASS_NAMES 
        })

    except Exception as e:
        print(f"Error during detection: {e}")
        return jsonify({'error': f'Detection failed: {str(e)}'}), 500

# --- Flask 서버 실행 ---
if __name__ == '__main__':
    # debug=True 설정 시 코드 변경 감지 및 자동 재시작
    app.run(host='0.0.0.0', port=5001, debug=True)