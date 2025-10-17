# app.py
import os
from io import BytesIO
from flask import Flask, request, jsonify
from PIL import Image
import torch
import torchvision

# --- Flask & CORS ---
app = Flask(__name__)
try:
    from flask_cors import CORS
    CORS(app, resources={r"/*": {"origins": "*"}})
except Exception:
    # CORS 미설치시에도 앱은 동작
    pass

# 업로드 최대 10MB
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

# --- 모델 로드 (서버 시작 시 1회) ---
MODEL_PATH = os.environ.get("MODEL_PATH", "best.torchscript.ptl")
DEVICE = "cpu"  # Render 무료 플랜은 보통 CPU

try:
    model = torch.jit.load(MODEL_PATH, map_location=DEVICE)
    model.eval()
    print(f"[startup] model loaded from: {MODEL_PATH}")
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

# ---------- 디버그/헬스 ----------
@app.get("/")
def index():
    return jsonify(
        message="OK: use GET /health and POST /detect",
        routes=[str(r) for r in app.url_map.iter_rules()],
    ), 200

@app.get("/health")
def health():
    return jsonify(status="ok"), 200

@app.get("/version")
def version():
    return jsonify(
        torch=str(torch.__version__),
        torchvision=str(torchvision.__version__),
        model_loaded=bool(model),
        model_path=MODEL_PATH,
    ), 200

# ---------- 탐지 ----------
# /detect와 /detect/ 모두 허용
@app.route("/detect", methods=["POST"])
@app.route("/detect/", methods=["POST"])
def detect():
    """
    multipart/form-data 로 'file' (또는 'image') 필드에 이미지를 보냄
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

    # 전처리: 640x640 resize + CHW float[0,1]
    img_sz = 640
    img_resized = img.resize((img_sz, img_sz))
    x = torchvision.transforms.functional.to_tensor(img_resized).unsqueeze(0)  # [1,3,640,640]

    with torch.no_grad():
        # TorchScript forward는 인자 1개만!
        out = model(x)
        # 모델에 따라 out이 tuple/list일 수 있으니 방어적으로 처리
        if isinstance(out, (list, tuple)):
            y = out[0]
        else:
            y = out
        # 기대 형태: [1, 25200, 85] (cx, cy, w, h, obj, cls...)
        if y.dim() == 3:
            y = y.squeeze(0)
        elif y.dim() == 2:
            pass
        else:
            return jsonify(error=f"unexpected model output shape: {list(y.shape)}"), 500

    y = y.cpu()  # [25200, 85] 가정
    boxes, scores, classes = [], [], []

    # obj_conf * class_conf 최대치로 스코어 산출
    for row in y:
        obj = float(row[4])
        if obj < 0.25:
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
    sx, sy = W / img_sz, H / img_sz
    detections = []
    for i in keep_idx:
        x1, y1, x2, y2 = boxes[i]
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

# ---------- 에러 핸들러 ----------
@app.errorhandler(404)
def not_found(e):
    return jsonify(
        error="not found",
        hint="Use GET /health and POST /detect",
        routes=[str(r) for r in app.url_map.iter_rules()],
    ), 404

if __name__ == "__main__":
    print("URL MAP:", app.url_map)
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
