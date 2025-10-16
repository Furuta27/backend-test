# app.py — Flask + TorchScript(.ptl) 전용 서버
# - /            : 서비스 정보
# - /health      : 헬스체크
# - /diag        : 실행/모델 진단
# - /api/detect/image (POST, multipart/form-data, key=image_file|image)

import io
import os
import sys
import json
import numpy as np
from typing import List, Dict, Any

from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS

import torch
import torchvision

# ---------------- Config ----------------
DEVICE = os.getenv("DEVICE", "cpu")
MODEL_PATH = os.getenv("MODEL_PATH", "best.torchscript.ptl")   # ✅ TorchScript만!
LABELS_PATH = os.getenv("LABELS_PATH", "labels.txt")
IMG_SIZE = int(os.getenv("IMG_SIZE", "640"))
CONF_THRES = float(os.getenv("CONF_THRES", "0.25"))
IOU_THRES  = float(os.getenv("IOU_THRES", "0.45"))
MAX_BODY   = int(os.getenv("MAX_CONTENT_LENGTH", str(20 * 1024 * 1024)))  # 20MB 기본

# ---------------- App -------------------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_BODY
CORS(app)  # 필요 시 origins 화이트리스트로 제한 권장

# ---------------- Labels ----------------
if os.path.exists(LABELS_PATH):
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        CLASS_NAMES = [ln.strip() for ln in f if ln.strip()]
else:
    CLASS_NAMES = []

# ---------------- Model -----------------
print(f"[BOOT] python={sys.version}", flush=True)
print(f"[BOOT] torch={torch.__version__} torchvision={torchvision.__version__}", flush=True)
print(f"[BOOT] loading TorchScript: {MODEL_PATH}", flush=True)

try:
    model = torch.jit.load(MODEL_PATH, map_location=DEVICE)  # ❗ Ultralytics YOLO API 금지
    model.eval()
    print(f"[BOOT] model loaded: {type(model).__name__}", flush=True)
except Exception as e:
    print(f"[BOOT][FATAL] failed to load model: {e}", flush=True)
    raise

# ---------------- Utils -----------------
def preprocess(pil_img: Image.Image, img_size: int = IMG_SIZE) -> torch.Tensor:
    """PIL → 1x3xHxW, float32 [0..1]"""
    img = pil_img.convert("RGB").resize((img_size, img_size))
    arr = np.asarray(img).astype(np.float32) / 255.0        # HWC
    arr = np.transpose(arr, (2, 0, 1))                      # CHW
    x = torch.from_numpy(arr).unsqueeze(0)                  # 1x3xHxW
    return x

def postprocess(pred: Any, conf_thres=CONF_THRES, iou_thres=IOU_THRES) -> List[Dict[str, Any]]:
    """
    기대 포맷: (N, 6) with [x1,y1,x2,y2,score,cls]
    TorchScript가 (pred,) 또는 [pred]로 감싸서 반환할 수 있어 안전 처리.
    """
    if isinstance(pred, (list, tuple)):
        pred = pred[0]

    if not isinstance(pred, torch.Tensor):
        raise RuntimeError(f"unexpected pred type: {type(pred)}")

    p = pred.detach().cpu()
    if p.ndim != 2 or p.shape[1] < 6:
        # 모델 아웃풋 포맷이 다르면 여기에서 매핑하거나 빈 리스트 반환
        return []

    if p.numel() == 0:
        return []

    # confidence 필터
    mask = p[:, 4] >= conf_thres
    p = p[mask]
    if p.numel() == 0:
        return []

    boxes = p[:, :4]
    scores = p[:, 4]
    clses  = p[:, 5].to(torch.int64)

    # NMS
    keep = torchvision.ops.nms(boxes, scores, iou_thres)
    boxes = boxes[keep].numpy()
    scores = scores[keep].numpy()
    clses  = clses[keep].numpy()

    results: List[Dict[str, Any]] = []
    for (x1, y1, x2, y2), sc, c in zip(boxes, scores, clses):
        results.append({
            "classIndex": int(c),
            "score": float(sc),
            "rect": {
                "left": float(x1), "top": float(y1),
                "right": float(x2), "bottom": float(y2)
            }
        })
    return results

# ---------------- Routes ----------------
@app.get("/")
def index():
    return jsonify({
        "service": "ok",
        "endpoints": ["/health", "/diag", "/api/detect/image (POST multipart/form-data)"],
        "img_size": IMG_SIZE,
        "conf_thres": CONF_THRES,
        "iou_thres": IOU_THRES,
        "max_body_bytes": MAX_BODY,
    })

@app.get("/health")
def health():
    return jsonify({"ok": True})

@app.get("/diag")
def diag():
    import importlib.util as ilu
    return jsonify({
        "file": __file__,
        "cwd": os.getcwd(),
        "python": sys.version,
        "model_path": MODEL_PATH,
        "model_type": type(model).__name__,
        "labels_count": len(CLASS_NAMES),
        "has_ultralytics": bool(ilu.find_spec("ultralytics")),
    })

@app.post("/api/detect/image")
def detect_image():
    # 클라이언트가 image_file 또는 image 키를 보낼 수 있도록 둘 다 허용
    f = request.files.get("image_file") or request.files.get("image")
    if f is None:
        return jsonify({"error": "No file provided (use 'image_file' or 'image')"}), 400

    try:
        pil = Image.open(io.BytesIO(f.read())).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Invalid image: {e}"}), 400

    x = preprocess(pil)

    # ✅ TorchScript는 오직 단일 인자만 허용: model(x)
    try:
        with torch.no_grad():
            print("[INFER] calling model(x) only", flush=True)
            pred = model(x)
    except TypeError as e:
        # 이 메시지가 보이면 다른 엔트리/코드가 실행 중일 확률 높음
        return jsonify({"error": f"TS forward call failed: {e}"}), 500
    except Exception as e:
        return jsonify({"error": f"infer failed: {type(e).__name__}: {e}"}), 500

    try:
        detections = postprocess(pred)
    except Exception as e:
        return jsonify({"error": f"postprocess failed: {e}"}), 500

    return jsonify({"detections": detections, "class_names": CLASS_NAMES})

# ---------------- Main ------------------
if __name__ == "__main__":
    # 로컬 테스트용 (Render에선 gunicorn 사용)
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
