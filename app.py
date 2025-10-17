# app.py
import os
from io import BytesIO
from flask import Flask, request, jsonify
from PIL import Image
import torch
import torchvision
import time

app = Flask(__name__)

# --- 모델 로드 (서버 시작 시 1회) ---
MODEL_PATH = os.environ.get("MODEL_PATH", "best.torchscript.ptl")
DEVICE = "cpu"  # Render 무료 플랜은 보통 CPU

try:
    model = torch.jit.load(MODEL_PATH, map_location=DEVICE)
    model.eval()
except Exception as e:
    print("[startup] model load failed:", e)
    model = None

# 간단 NMS 유틸
def nms(boxes, scores, iou_th=0.45, top_k=50):
    if len(boxes) == 0:
        return []
    keep = torchvision.ops.nms(
        torch.tensor(boxes, dtype=torch.float32),
        torch.tensor(scores, dtype=torch.float32),
        iou_th
    )
    keep = keep[:top_k].tolist()
    return keep

@app.get("/health")
def health():
    return jsonify(status="ok"), 200

@app.post("/detect")
def detect():
    """
    클라이언트는 multipart/form-data 로 'file' 필드에 이미지를 보냅니다.
    (필드명: file 권장. 호환 위해 image 도 허용)
    """
    f = request.files.get("file") or request.files.get("image")
    if not f:
        return jsonify(error='no file field "file" (or "image")'), 400

    if model is None:
        return jsonify(error="model not loaded"), 500

    # 이미지 로딩
    file_bytes = f.read()
    try:
        img = Image.open(BytesIO(file_bytes)).convert("RGB")
    except Exception:
        return jsonify(error="invalid image"), 400

    # 전처리: 640x640 resize + [0,1] normalize + CHW
    img_sz = 640
    img_resized = img.resize((img_sz, img_sz))
    x = torch.from_numpy(
        (torchvision.transforms.functional.to_tensor(img_resized)).numpy()
    ).unsqueeze(0)  # [1,3,640,640]

    with torch.no_grad():
        # TorchScript forward는 인자 1개만! (중요)
        y = model(x)[0]  # 보통 [N, 25200, 85] 형태 (x,y,w,h,obj,cls...)
        # y: Tensor

    y = y.squeeze(0).cpu()  # [25200, 85]
    boxes = []
    scores = []
    classes = []

    # YOLOv5 형식 가정: cx, cy, w, h 는 0~img_sz 기준
    # obj_conf * class_conf 최대치로 스코어 산출
    for row in y:
        obj = float(row[4])
        if obj < 0.25:  # confidence threshold
            continue
        cls_confs = row[5:]
        cls_idx = int(torch.argmax(cls_confs))
        cls_conf = float(cls_confs[cls_idx])
        score = obj * cls_conf
        if score < 0.25:
            continue

        cx, cy, w, h = [float(v) for v in row[:4]]
        x1 = max(0.0, cx - w / 2)
        y1 = max(0.0, cy - h / 2)
        x2 = min(float(img_sz), cx + w / 2)
        y2 = min(float(img_sz), cy + h / 2)

        boxes.append([x1, y1, x2, y2])
        scores.append(score)
        classes.append(cls_idx)

    keep_idx = nms(boxes, scores, iou_th=0.45, top_k=50)

    # 원본 이미지 스케일로 다시 변환
    W, H = img.size
    detections = []
    for i in keep_idx:
        x1, y1, x2, y2 = boxes[i]
        # 640 기준 -> 원본 비율 변환
        sx, sy = W / img_sz, H / img_sz
        detections.append({
            "x": int(x1 * sx),
            "y": int(y1 * sy),
            "w": int((x2 - x1) * sx),
            "h": int((y2 - y1) * sy),
            "score": round(scores[i], 4),
            "classIndex": int(classes[i]),
        })

    return jsonify(
        class_names=[
            "coca cola","coke zero","pepsi zero",
            "chilsung cider","chilsung cider zero",
            "fanta","hot6","hot6 force"
        ],
        detections=detections
    ), 200

if __name__ == "__main__":
    print("URL MAP:", app.url_map)
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
