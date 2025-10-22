# app.py — Flask 3.x 호환 / YOLO TorchScript 백엔드 (Render 배포용)
# - /health : 상태 체크
# - /detect : 멀티파트 업로드 (file|image)
# - /detect-json (/detect_json) : JSON(base64) 업로드
# - /jobs/<id> : 202(jobId) 폴링하여 결과 수신
# - before_first_request 제거된 Flask 3.x를 위한 호환 shim 포함
# - warmup_and_load() 워밍업 호출 버그 수정: 로컬 m(...)로 invoke 후 model=m 적용

import os
import io
import time
import uuid
import base64
import queue
import threading
import logging
from typing import List, Dict, Any, Tuple

from flask import Flask, request, jsonify, redirect
from werkzeug.exceptions import HTTPException
from PIL import Image

# 선택적 CORS
try:
    from flask_cors import CORS
    CORS_AVAILABLE = True
except Exception:
    CORS_AVAILABLE = False

# 선택적 PyTorch/torchvision
try:
    import torch  # type: ignore
    import torchvision  # type: ignore
    TV_AVAILABLE = True
    try:
        # 과도한 CPU 스레드 사용 방지(선택)
        torch.set_num_threads(1)  # type: ignore
    except Exception:
        pass
except Exception:
    torch = None  # type: ignore
    torchvision = None  # type: ignore
    TV_AVAILABLE = False

# ───────────────────────────── 설정값 ─────────────────────────────
PORT = int(os.environ.get("PORT", "10000"))
MODEL_PATH = os.environ.get("MODEL_PATH", "best.torchscript.ptl")
CLASSES_PATH = os.environ.get("CLASSES_PATH", "labels.txt")

CONF_TH = float(os.environ.get("CONF_TH", "0.25"))
IOU_TH = float(os.environ.get("IOU_TH", "0.45"))
TOP_K = int(os.environ.get("TOP_K", "50"))
INPUT_SIZE = int(os.environ.get("INPUT_SIZE", "640"))

WORKERS = int(os.environ.get("WORKERS", "2"))
JOB_TTL_SEC = int(os.environ.get("JOB_TTL_SEC", str(15 * 60)))
MAX_CONTENT_LENGTH_MB = int(os.environ.get("MAX_CONTENT_LENGTH_MB", "12"))

# ───────────────────────────── Flask ─────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH_MB * 1024 * 1024
if CORS_AVAILABLE:
    CORS(app)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("app")

# ────────────────────── Flask 3.x compat: before_first_request ──────────────────────
# Flask 3.x에는 before_first_request가 없으므로 동일 동작을 before_request로 에뮬레이트
import threading as _flask3_th
if not hasattr(app, "before_first_request"):
    _bff_lock = _flask3_th.Lock()
    _bff_done = {"v": False}

    def _before_first_request_compat(func):
        def _guard(*args, **kwargs):
            if _bff_done["v"]:
                return
            with _bff_lock:
                if _bff_done["v"]:
                    return
                _bff_done["v"] = True
                return func()
        app.before_request(_guard)
        return func

    app.before_first_request = _before_first_request_compat  # type: ignore

# ───────────────────────────── 전역 상태 ─────────────────────────
model = None  # type: ignore
model_err: str | None = None
classes: List[str] = []

jobs: Dict[str, Dict[str, Any]] = {}
in_queue: "queue.Queue[Tuple[str, bytes, str]]" = queue.Queue()

# ───────────────────────────── 유틸 ─────────────────────────────
def load_classes() -> None:
    """labels.txt 로드 (없으면 더미)"""
    global classes
    try:
        if os.path.isfile(CLASSES_PATH):
            with open(CLASSES_PATH, "r", encoding="utf-8") as f:
                classes = [ln.strip() for ln in f if ln.strip()]
            log.info(f"[startup] loaded {len(classes)} classes")
    except Exception as e:
        log.warning(f"[startup] labels load fail: {e}")
    if not classes:
        classes = [f"cls_{i}" for i in range(100)]

def yolo_forward(x):
    """TorchScript YOLO forward (list/tuple 지원)"""
    y = model(x)
    return y[0] if isinstance(y, (list, tuple)) else y

def py_nms(boxes: List[List[float]], scores: List[float], iou_th: float, top_k: int) -> List[int]:
    import numpy as np
    if not boxes:
        return []
    b = np.array(boxes, np.float32)
    s = np.array(scores, np.float32)
    order = s.argsort()[::-1]
    keep: List[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if len(keep) >= top_k:
            break
        xx1 = np.maximum(b[i, 0], b[order[1:], 0])
        yy1 = np.maximum(b[i, 1], b[order[1:], 1])
        xx2 = np.minimum(b[i, 2], b[order[1:], 2])
        yy2 = np.minimum(b[i, 3], b[order[1:], 3])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ai = (b[i, 2] - b[i, 0]) * (b[i, 3] - b[i, 1])
        aj = (b[order[1:], 2] - b[order[1:], 0]) * (b[order[1:], 3] - b[order[1:], 1])
        iou = inter / (ai + aj - inter + 1e-6)
        inds = (iou <= iou_th).nonzero()[0]
        order = order[inds + 1]
    return keep

def run_nms(boxes: List[List[float]], scores: List[float], iou_th: float, top_k: int) -> List[int]:
    if not boxes:
        return []
    if TV_AVAILABLE:
        with torch.no_grad():  # type: ignore
            keep = torchvision.ops.nms(  # type: ignore
                torch.tensor(boxes),  # type: ignore
                torch.tensor(scores),  # type: ignore
                iou_th,
            )
            return keep[:top_k].tolist()
    return py_nms(boxes, scores, iou_th, top_k)

def preprocess(img: Image.Image):
    """[1,3,H,W] 텐서 생성 (torchvision 사용) — TV_AVAILABLE일 때만 호출"""
    from torchvision.transforms.functional import to_tensor  # type: ignore
    im = img.resize((INPUT_SIZE, INPUT_SIZE))
    return to_tensor(im).unsqueeze(0)  # [1,3,H,W]

def postprocess(out, W: int, H: int) -> List[Dict[str, Any]]:
    """YOLOv5(TorchScript) 출력 파싱:
       row: [cx,cy,w,h,obj, cls0, cls1, ...]
    """
    if not TV_AVAILABLE or torch is None:
        return []

    if hasattr(out, "dim") and out.dim() == 3:
        out = out.squeeze(0).cpu()
    elif hasattr(out, "cpu"):
        out = out.cpu()

    boxes: List[List[float]] = []
    scores: List[float] = []
    cls_idx: List[int] = []

    for row in out:
        obj = float(row[4])
        if obj < 1e-6:
            continue
        cls_confs = row[5:]
        ci = int(torch.argmax(cls_confs))  # type: ignore
        cc = float(cls_confs[ci])
        sc = obj * cc
        if sc < CONF_TH:
            continue
        cx, cy, w, h = [float(v) for v in row[:4]]
        x1 = max(0.0, cx - w / 2)
        y1 = max(0.0, cy - h / 2)
        x2 = min(float(INPUT_SIZE), cx + w / 2)
        y2 = min(float(INPUT_SIZE), cy + h / 2)
        boxes.append([x1, y1, x2, y2])
        scores.append(sc)
        cls_idx.append(ci)

    keep = run_nms(boxes, scores, IOU_TH, TOP_K)
    sx, sy = W / INPUT_SIZE, H / INPUT_SIZE

    dets: List[Dict[str, Any]] = []
    for i in keep:
        x1, y1, x2, y2 = boxes[i]
        dets.append(
            {
                "x": int(x1 * sx),
                "y": int(y1 * sy),
                "w": int((x2 - x1) * sx),
                "h": int((y2 - y1) * sy),
                "score": round(scores[i], 4),
                "classIndex": int(cls_idx[i]),
                "className": classes[cls_idx[i]] if 0 <= cls_idx[i] < len(classes) else str(cls_idx[i]),
            }
        )
    return dets

def warmup_and_load() -> None:
    """모델/라벨 로딩 + 더미 인퍼런스로 워밍업 (버그 수정: m(...)로 invoke 후 model=m)"""
    global model, model_err
    try:
        load_classes()
        if not TV_AVAILABLE:
            raise RuntimeError("torch/torchvision not available in this runtime")

        # 1) 모델 로드
        m = torch.jit.load(MODEL_PATH, map_location="cpu")  # type: ignore
        m.eval()  # type: ignore

        # 2) 로컬 모델로 워밍업 (전역 model이 None이므로 yolo_forward() 사용 금지)
        with torch.no_grad():  # type: ignore
            _ = m(torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE))  # type: ignore

        # 3) 성공 시 전역 반영
        model = m
        model_err = None
        log.info("[startup] model ready")
    except Exception as e:
        model = None
        model_err = str(e)
        log.exception("[startup] model load failed")

def worker_loop(worker_idx: int) -> None:
    """비동기 잡 소비자"""
    while True:
        job_id, data, filename = in_queue.get()
        meta = jobs.get(job_id, {})
        try:
            t0 = time.time()
            img = Image.open(io.BytesIO(data)).convert("RGB")
            W, H = img.size

            if not TV_AVAILABLE or model is None:
                dets: List[Dict[str, Any]] = []
            else:
                x = preprocess(img)
                with torch.no_grad():  # type: ignore
                    raw = yolo_forward(x)
                dets = postprocess(raw, W, H)

            ms = int((time.time() - t0) * 1000)
            jobs[job_id] = {
                "status": "done",
                "result": {"class_names": classes, "detections": dets, "time_ms": ms},
                "t0": meta.get("t0", time.time()),
            }
            log.info(f"[job {job_id}] done {ms}ms det={len(dets)}")
        except Exception as e:
            jobs[job_id] = {"status": "error", "error": str(e), "t0": meta.get("t0", time.time())}
            log.exception(f"[job {job_id}] error")

def start_workers(n: int) -> None:
    n = max(1, n)
    for i in range(n):
        t = threading.Thread(target=worker_loop, args=(i,), daemon=True)
        t.start()
        log.info(f"[worker] started #{i}")

def _gc_loop():
    while True:
        now = time.time()
        for k, v in list(jobs.items()):
            if now - v.get("t0", now) > JOB_TTL_SEC:
                jobs.pop(k, None)
        time.sleep(30)

# ───────────────────── 최초 1회 초기화 (Flask 3.x에서도 동작) ─────────────────────
@app.before_first_request  # Flask 3.x에서는 위 shim을 통해 실행
def kickoff():
    threading.Thread(target=warmup_and_load, daemon=True).start()
    start_workers(WORKERS)
    threading.Thread(target=_gc_loop, daemon=True).start()
    log.info("[startup] background threads started")

# ───────────────────────────── 라우팅 ─────────────────────────────
@app.get("/")
def root():
    return redirect("/health")

@app.get("/health")
def health():
    status = "ready" if model is not None else ("error" if model_err else "warming")
    payload = {"status": status}
    if model_err:
        payload["error"] = model_err
    return jsonify(payload), 200

# 멀티파트 업로드 (/detect)
@app.post("/detect")
def detect():
    if model_err:
        return jsonify(error=f"model error: {model_err}"), 500
    if model is None:
        return jsonify(error="model loading, retry later"), 503

    f = request.files.get("file") or request.files.get("image")
    if not f:
        return jsonify(error='no file (fields: "file" or "image")'), 400

    data = f.read()
    if not data:
        return jsonify(error="empty file"), 400

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "queued", "t0": time.time()}
    in_queue.put((job_id, data, getattr(f, "filename", "image")))
    return jsonify({"jobId": job_id}), 202

# JSON(base64) 업로드 (/detect-json, /detect_json)
@app.post("/detect-json")
def detect_json():
    if model_err:
        return jsonify(error=f"model error: {model_err}"), 500
    if model is None:
        return jsonify(error="model loading, retry later"), 503

    j = request.get_json(silent=True) or {}
    b64 = j.get("data")
    filename = j.get("filename") or "image.jpg"
    if not b64:
        return jsonify(error="no base64 'data' field"), 400

    try:
        data = base64.b64decode(b64)
    except Exception as e:
        return jsonify(error=f"invalid base64: {e}"), 400

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "queued", "t0": time.time()}
    in_queue.put((job_id, data, filename))
    return jsonify({"jobId": job_id}), 202

@app.post("/detect_json")
def detect_json_alias():
    return detect_json()

# 잡 상태 폴링
@app.get("/jobs/<job_id>")
def job(job_id: str):
    meta = jobs.get(job_id)
    if not meta:
        return jsonify(error="job not found"), 404
    if meta["status"] == "done":
        return jsonify(meta["result"]), 200
    if meta["status"] == "error":
        return jsonify(error=meta.get("error", "job failed")), 500
    return jsonify(status=meta["status"]), 202

# 글로벌 에러 핸들러
@app.errorhandler(Exception)
def handle_ex(e):
    if isinstance(e, HTTPException):
        return jsonify(error=str(e)), (e.code or 500)
    log.exception("unhandled exception")
    return jsonify(error="internal error"), 500

# 로컬 실행 (Render에서는 gunicorn 사용)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
