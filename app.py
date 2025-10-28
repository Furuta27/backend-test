import os, io, gc, json, uuid, time, threading, queue
from typing import Dict, Any, Optional, Tuple, List

from flask import Flask, request, jsonify

# ─────────────────────────────────────────────────────────────────────────────
# Flask
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Global State
# ─────────────────────────────────────────────────────────────────────────────
# 모델은 "yolov5s3.torchscript.ptl"만 사용 (환경변수로 경로 변경 가능)
TS_MODEL_PATH = os.getenv("TS_MODEL_PATH", "yolov5s4.torchscript.ptl")
INPUT_SIZE = int(os.getenv("INPUT_SIZE", "640"))   # YOLO 입력 크기 정사각
CONF_THRES = float(os.getenv("CONF_THRES", "0.25"))  # confidence threshold
ALLOW_SYNC = True

model_kind: str = "none"
model = None  # torchscript jit module
model_err: Optional[str] = None
ready: bool = False

LABELS: List[str] = []

# 202 비동기 처리용
_jobs: Dict[str, Dict[str, Any]] = {}
_job_q: "queue.Queue[Tuple[str, bytes, str]]" = queue.Queue()
_infer_lock = threading.Lock()
_started_once = False

# ─────────────────────────────────────────────────────────────────────────────
# Utils
# ─────────────────────────────────────────────────────────────────────────────
def _read_labels():
    """labels.txt 를 읽어 클래스 이름 리스트를 구성"""
    global LABELS
    p = os.path.join(os.getcwd(), "junk.txt")
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            LABELS = [ln.strip() for ln in f if ln.strip()]
    else:
        LABELS = []

def _pil_open(img_bytes: bytes):
    from PIL import Image
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")

def _to_numpy(img):
    import numpy as np
    return np.array(img)

def _letterbox_pil(im, new_size: int):
    """
    PIL.Image -> (letterboxed PIL.Image, scale, pad_x, pad_y, orig_w, orig_h)
    좌우/상하 여백을 114,114,114(회색)로 채워 new_size x new_size 로 맞춤
    """
    from PIL import Image
    ow, oh = im.size
    if ow == 0 or oh == 0:
        raise ValueError("invalid image size")

    scale = min(new_size / ow, new_size / oh)
    nw, nh = int(round(ow * scale)), int(round(oh * scale))
    im_resized = im.resize((nw, nh), Image.BILINEAR)

    canvas = Image.new("RGB", (new_size, new_size), (114, 114, 114))
    pad_x = (new_size - nw) // 2
    pad_y = (new_size - nh) // 2
    canvas.paste(im_resized, (pad_x, pad_y))
    return canvas, scale, pad_x, pad_y, ow, oh

def _unwrap_to_numpy(x):
    """torch.Tensor | list | tuple → numpy.ndarray 로 통일"""
    import numpy as np
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    if isinstance(x, (list, tuple)):
        return _unwrap_to_numpy(x[0]) if x and hasattr(x[0], "__array__") is False else \
               _unwrap_to_numpy(np.array(x))
    try:
        return x if hasattr(x, "shape") else _unwrap_to_numpy(np.array(x))
    except Exception:
        return None

def _parse_yolo_output(arr, conf_thres: float) -> List[List[float]]:
    """
    YOLOv5s TorchScript 출력 해석:
    - 기대 형식: (N, 6) or (1, N, 6) with [x1, y1, x2, y2, conf, cls]
    - 필요 시 list/ndarray 다양한 경우 방어적으로 처리
    반환: [[x1,y1,x2,y2,conf,cls], ...]
    """
    import numpy as np

    if arr is None:
        return []

    a = arr
    if isinstance(a, (list, tuple)):
        try:
            a = np.array(a)
        except Exception:
            return []

    if a.ndim == 3 and a.shape[0] == 1:
        a = a[0]  # (1, N, 6) → (N, 6)
    if a.ndim == 2 and a.shape[1] >= 6:
        pass
    else:
        # 일부 구현은 (N,) 의 object 배열에 박스 배열이 들어있을 수 있음
        try:
            flat = []
            for row in a:
                r = np.array(row).reshape(-1)
                if r.shape[0] >= 6:
                    flat.append(r[:6])
            a = np.array(flat)
        except Exception:
            return []

    if a.size == 0:
        return []

    # conf 필터링
    out = []
    for r in a:
        x1, y1, x2, y2, conf, cls = float(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])
        if conf >= conf_thres:
            out.append([x1, y1, x2, y2, conf, cls])
    return out

# ─────────────────────────────────────────────────────────────────────────────
# Model Loading / Inference
# ─────────────────────────────────────────────────────────────────────────────
def load_model_bg():
    """백그라운드에서 TorchScript 모델만 로드. 실패 시 ready=False + model_err 설정."""
    global model, model_kind, model_err, ready
    model = None
    model_kind = "none"
    model_err = None
    ready = False

    try:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        try:
            import torch
            torch.set_num_threads(1)
        except Exception:
            pass

        _read_labels()

        if not os.path.exists(TS_MODEL_PATH):
            model_err = f"torchscript model not found: {TS_MODEL_PATH}"
            app.logger.error("[startup] " + model_err)
            return

        import torch
        m = torch.jit.load(TS_MODEL_PATH, map_location="cpu")
        m.eval()
        # 더 일찍 터지도록 더미 추론 1회
        dummy = torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE)
        with torch.no_grad():
            _ = m(dummy)

        model = m
        model_kind = "torchscript"
        ready = True
        model_err = None
        app.logger.info(f"[startup] torchscript loaded: {TS_MODEL_PATH}")

    except Exception as e:
        model = None
        model_kind = "none"
        model_err = str(e)
        ready = False
        app.logger.exception("[startup] model load failed")

def run_inference(img_bytes: bytes) -> Dict[str, Any]:
    """단일 이미지 바이트에 대한 추론. 결과 박스는 원본 이미지 좌표계로 반환."""
    t0 = time.time()
    if not ready or model is None or model_kind != "torchscript":
        raise RuntimeError("model not ready")

    from PIL import Image
    import torch, numpy as np

    # 원본 로드
    pil = _pil_open(img_bytes)
    # letterbox
    canvas, scale, pad_x, pad_y, ow, oh = _letterbox_pil(pil, INPUT_SIZE)

    # 텐서화
    np_img = _to_numpy(canvas)  # HWC RGB [0..255]
    inp = torch.from_numpy(np_img).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    with torch.no_grad():
        out = model(inp)
    arr = _unwrap_to_numpy(out)  # 다양한 타입 → ndarray

    # YOLO 출력 파싱 (letterbox 좌표)
    det_raw = _parse_yolo_output(arr, CONF_THRES)

    # letterbox → 원본 좌표로 복원
    dets: List[Dict[str, Any]] = []
    for x1, y1, x2, y2, conf, cls in det_raw:
        # 원본계로 역변환
        ox1 = max(0.0, min(ow, (x1 - pad_x) / scale))
        oy1 = max(0.0, min(oh, (y1 - pad_y) / scale))
        ox2 = max(0.0, min(ow, (x2 - pad_x) / scale))
        oy2 = max(0.0, min(oh, (y2 - pad_y) / scale))

        w = max(0.0, ox2 - ox1)
        h = max(0.0, oy2 - oy1)
        cls_i = int(round(cls))
        cls_name = LABELS[cls_i] if 0 <= cls_i < len(LABELS) else None

        dets.append({
            "classIndex": cls_i,
            "className": cls_name,
            "x": int(round(ox1)),
            "y": int(round(oy1)),
            "w": int(round(w)),
            "h": int(round(h)),
            "score": float(conf),
        })

    return {
        "class_names": LABELS,
        "detections": dets,
        "time_ms": int((time.time() - t0) * 1000),
        "__serverImageW": ow, "__serverImageH": oh,
        "__input_size": INPUT_SIZE,
        "__model": model_kind,
    }

# ─────────────────────────────────────────────────────────────────────────────
# Worker (202 async)
# ─────────────────────────────────────────────────────────────────────────────
def _start_worker_once():
    def _worker():
        app.logger.info("[worker] started")
        while True:
            try:
                job_id, img_bytes, req_kind = _job_q.get()
                _jobs[job_id] = {"status": "running"}
                with _infer_lock:
                    out = run_inference(img_bytes)
                _jobs[job_id] = {"status": "done", "result": out}
            except Exception as e:
                _jobs[job_id] = {"status": "error", "error": str(e)}
            finally:
                try:
                    del img_bytes
                except Exception:
                    pass
                gc.collect()
                _job_q.task_done()
    t = threading.Thread(target=_worker, daemon=True)
    t.start()

def _startup_once():
    global _started_once
    if _started_once:
        return
    _started_once = True
    threading.Thread(target=load_model_bg, daemon=True).start()
    _start_worker_once()
    app.logger.info("[startup] background threads started")

# 모든 요청 전에 1회 스타트 보장
@app.before_request
def _ensure_started():
    _startup_once()
    request.start_ts = time.time()

# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return jsonify(ok=True, health="/health"), 200

@app.get("/health")
def health():
    status = "ready" if ready and model_err is None else ("error" if model_err else "warming")
    info = {
        "status": status,
        "model": model_kind,
        "model_file": TS_MODEL_PATH if model_kind == "torchscript" else None,
        "labels_count": len(LABELS),
        "input_size": INPUT_SIZE,
        "conf_thres": CONF_THRES,
        "error": model_err,
    }
    code = 200 if status == "ready" else (500 if status == "error" else 503)
    return jsonify(info), code

def _read_image_from_request() -> Tuple[Optional[bytes], Optional[str]]:
    """multipart(file|image) 또는 JSON({data: base64}) 지원"""
    file = request.files.get("file") or request.files.get("image")
    if file:
        return file.read(), "multipart"
    try:
        body = request.get_json(silent=True) or {}
        b64 = body.get("data")
        if b64:
            import base64
            return base64.b64decode(b64), "json"
    except Exception:
        pass
    return None, None

@app.post("/detect")
def detect():
    if not ready or model is None or model_kind != "torchscript":
        # 로딩 실패/진행중이면 명확히 503/500로 알림
        if model_err:
            return jsonify(error=model_err), 500
        return jsonify(error="loading"), 503

    img_bytes, kind = _read_image_from_request()
    if not img_bytes:
        return jsonify(error="no file"), 400

    sync = ALLOW_SYNC and (request.args.get("sync") == "1" or request.headers.get("X-Detect-Sync") == "1")

    if sync:
        try:
            with _infer_lock:
                out = run_inference(img_bytes)
            return jsonify(out), 200
        except Exception as e:
            app.logger.exception("[detect] sync failed")
            return jsonify(error=str(e)), 500
        finally:
            try:
                del img_bytes
            except Exception:
                pass
            gc.collect()

    # async 202
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "queued"}
    _job_q.put((job_id, img_bytes, kind or "multipart"))
    return jsonify(jobId=job_id), 202

@app.post("/detect-json")
def detect_json():
    return detect()

@app.get("/jobs/<job_id>")
def job_status(job_id: str):
    j = _jobs.get(job_id)
    if not j:
        return jsonify(error="not found"), 404
    if j["status"] == "done":
        return jsonify(j["result"]), 200
    if j["status"] == "error":
        return jsonify(error=j.get("error", "unknown")), 500
    return jsonify(status=j["status"]), 202

# ─────────────────────────────────────────────────────────────────────────────
# Local run
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    _startup_once()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
