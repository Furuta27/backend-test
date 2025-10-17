# app.py
import os
from io import BytesIO
from typing import List, Tuple

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

import torch
import torchvision

# -------------------------
# Flask + CORS
# -------------------------
app = Flask(__name__)
CORS(app)  # 모바일/웹 클라이언트의 CORS 문제 방지

# 요청 본문 최대 크기(예: 20MB) - 과도한 업로드 방지
app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("MAX_CONTENT_LENGTH", 20 * 1024 * 1024))

# -------------------------
# 모델 로드 (서버 기동 시 1회)
# -------------------------
DEVICE = "cpu"  # Render 무료 플랜이면 보통 CPU
MODEL_PATH = os.environ.get("MODEL_PATH", "best.torchscript.ptl")

model = None
class_names = [
    "coca cola", "coke zero", "pepsi zero",
    "chilsung cider", "chilsung cider zero",
    "fanta", "hot6", "hot6 force"
]

def load_model():
    global model
    try:
        m = torch.jit.load(MODEL_PATH, map_location=DEVICE)
        m.eval()
        # (선택) 더미 텐서로 워밍업
        with torch.no_grad():
            _ = m(torch.zeros(1, 3, 640, 640))
        return m
    except Exception as e:
        print(f"[startup] model load failed: {e}")
        return None

model = load_model()

# -------------------------
# 유틸: NMS
# -------------------------
def nms(boxes: List[List[float]], scores: List[float], iou_th=0.45, top_k=50) -> List[int]:
    if not boxes:
        return []
    keep = torchvision.ops.nms(
        torch.tensor(boxes, dtype=torch.float32),
        torch.tensor(scores, dtype=torch.float32),
        iou_th
    )
    return keep[:top_k].tolist()

# -------------------------
# 유틸: 이미지 읽기/전처리
# -------------------------
IMG_SIZE = 640

def read_image_to_tensor(file_storage) -> Tuple[Image.Image, torch.Tensor]:
    """업로드 파일을 PIL.Image, Tensor([1,3,640,640])로 반환"""
    # 일부 클라이언트가 webp를 보낼 수 있음 → Pillow가 처리
    raw = file_storage.read()
    img = Image.open(BytesIO(raw)).convert("RGB")
    # 640x640 리사이즈 (간단한 방식)
    resized = img.resize((IMG_SIZE, IMG_SIZE))
    x = torchvision.transforms.functional.to_tensor(resized).unsqueeze(0)  # [1,3,640,640]
    return img, x

# -------------------------
# 라우트
# -------------------------
@app.get("/")
def index():
    return jsonify(message="OK. Try POST /detect with form-data field 'file'."), 200

@app.get("/health")
def health():
    return jsonify(status="ok"), 200

@app.post("/detect")
def detect():
    """
    요청 형식: multipart/form-data
      - 필드명: 'file' (호환 위해 'image'도 지원)
    응답 형식:
      {
        "class_names": [...],
        "detections": [
          {"x":..., "y":..., "w":..., "h":..., "score":..., "classIndex": ...},
          ...
        ]
      }
    """
    if model is None:
        return jsonify(error="model not loaded"), 500

    f = request.files.get("file") or request.files.get("image")
    if not f:
        return jsonify(error='no file field "file" (or "image")'), 400

    try:
        img, x = read_image_to_tensor(f)
    except Exception as e:
        return jsonify(error=f"invalid image: {e}"), 400

    # 추론 (TorchScript는 인자 1개만!)
    with torch.no_grad():
        y = model(x)  # 보통 [1, 25200, 85] 또는 (tuple) 형태
        # 모델이 tuple 반환 시 첫 텐서만 사용
        if isinstance(y, (list, tuple)):
            y = y[0]
        # [N, 25200, C] → [25200, C]
        y = y.squeeze(0).cpu()

    boxes = []
    scores = []
    classes = []

    # YOLOv5형식 가정: [cx, cy, w, h, obj, cls...]
    conf_th = float(os.environ.get("CONF_TH", 0.25))
    img_w, img_h = img.size

    for row in y:
        obj = float(row[4])
        if obj < conf_th:
            continue
        cls_scores = row[5:]
        cls_idx = int(torch.argmax(cls_scores))
        cls_conf = float(cls_scores[cls_idx])
        score = obj * cls_conf
        if score < conf_th:
            continue

        cx, cy, w, h = [float(v) for v in row[:4]]
        x1 = max(0.0, cx - w / 2)
        y1 = max(0.0, cy - h / 2)
        x2 = min(float(IMG_SIZE), cx + w / 2)
        y2 = min(float(IMG_SIZE), cy + h / 2)

        boxes.append([x1, y1, x2, y2])
        scores.append(score)
        classes.append(cls_idx)

    keep = nms(boxes, scores, iou_th=0.45, top_k=50)

    # 640 기준 → 원본 스케일 복원
    sx, sy = img_w / IMG_SIZE, img_h / IMG_SIZE
    detections = []
    for i in keep:
        x1, y1, x2, y2 = boxes[i]
        detections.append({
            "x": int(x1 * sx),
            "y": int(y1 * sy),
            "w": int((x2 - x1) * sx),
            "h": int((y2 - y1) * sy),
            "score": round(float(scores[i]), 4),
            "classIndex": int(classes[i]),
        })

    return jsonify(
        class_names=class_names,
        detections=detections
    ), 200

# -------------------------
# 엔트리포인트
# -------------------------
if __name__ == "__main__":
    print("URL MAP:", app.url_map)
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)
