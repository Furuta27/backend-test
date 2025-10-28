import os, io, gc, json, uuid, time, threading, queue
from typing import Dict, Any, Optional, Tuple, List

from flask import Flask, request, jsonify

# ─────────────────────────────────────────────────────────────────────────────
# Flask
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Config (env로 조정 가능)
# ─────────────────────────────────────────────────────────────────────────────
TS_MODEL_PATH = os.getenv("TS_MODEL_PATH", "yolov5s4.torchscript.ptl")  # ← 이 파일만 사용
INPUT_SIZE    = int(os.getenv("INPUT_SIZE", "640"))                      # YOLO 입력 정사각 크기
CONF_THRES    = float(os.getenv("CONF_THRES", "0.25"))                   # confidence 임계값
ALLOW_SYNC    = True                                                     # ?sync=1 허용 여부

# ─────────────────────────────────────────────────────────────────────────────
# Global state
# ─────────────────────────────────────────────────────────────────────────────
model_kind: str = "none"   # "torchscript" | "none"
model = None               # torchscript jit module
model_err: Optional[str] = None
ready: bool = False

LABELS: List[str] = []

# 202 비동기 작업 큐
_jobs: Dict[str, Dict[str, Any]] = {}
_job_q: "queue.Queue[Tuple[str, bytes, str]]" = queue.Queue()
_infer_lock = threading.Lock()  # 추론 동시 1개로 제한(메모리 안정)
_started_once = False

# ─────────────────────────────────────────────────────────────────────────────
# Utils
# ─────────────────────────────────────────────────────────────────────────────
def _read_labels():
    """labels.txt 파일을 읽어 클래스 이름을 구성"""
    global LABELS
    p = os.path.join(os.getcwd(), "labels.txt")
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
    여백(114,114,114)으로 new_size x new_size 정사각에 맞춤(비율 유지)
    """
    from PIL import Image
    ow, oh = im.size
    if ow <= 0 or oh <= 0:
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
    """torch.Tensor | list | tuple | ndarray → ndarray 로 통일"""
    import numpy as np
    try:
        import torch
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    if isinstance(x, (list, tuple)):
        try:
            return np.array(x)
        except Exception:
            return None
    try:
        return x if hasattr(x, "shape") else np.array(x)
    except Exception:
        return None

def _parse_yolo_output(arr, conf_thres: float) -> List[List[float]]:
    """
    기대 형식: (N,6) 또는 (1,N,6) with [x1,y1,x2,y2,conf,cls]
    """
    import numpy as np
    if arr is None:
        return []
    a = arr
    if isinstance(a, (list, tuple)):
        a = np.array(a)
    if a.ndim == 3 and a.shape[0] == 1:
        a = a[0]  # (1,N,6) → (N,6)
    if not (a.ndim == 2 and a.shape[1] >= 6):
        # object 배열 방어적으로 평탄화
        flat = []
        try:
            for row in a:
                r = np.array(row).reshape(-1)
                if r.shape[0] >= 6:
                    flat.append(r[:6])
            a = np.array(flat)
        except Exception:
            return []
    out = []
    for r in a:
        x1,y1,x2,y2,conf,cls = float(r[0]),float(r[1]),float(r[2]),float(r[3]),float(r[4]),float(r[5])
        if conf >= conf_thres:
            out.append([x1,y1,x2,y2,conf,cls])
    return out

# ─────────────────────────────────────────────────────────────────────────────
# Model loading / inference
# ─────────────────────────────────────────────────────────────────────────────
def load_model_sync():
    """동기 로딩: import 시점에 바로 모델을 올려 ready 상태로 만든다."""
    global model, model_kind, model_err, ready
    model = None
    model_kind = "none"
    model_err = None
    ready = False
    try:
        # 스레드 수 제한(메모리/CPU 안정)
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
        # 조기 실패를 위해 더미 입력으로 1회 실행
        with torch.no_grad():
            _ = m(torch.zeros(1,3,INPUT_SIZE,INPUT_SIZE))
        model = m
        model_kind = "torchscript"
        ready = True
        app.logger.info(f"[startup] torchscript loaded: {TS_MODEL_PATH}")
    except Exception as e:
        model = None
        model_kind = "none"
        model_err = str(e)
        ready = False
        app.logger.exception("[startup] model load failed")

def run_inference(img_bytes: bytes) -> Dict[str, Any]:
    """단일 이미지 바이트에 대한 추론. 결과 박스는 원본 이미지 좌표로 반환."""
    if not ready or model is None or model_kind != "torchscript":
        raise RuntimeError("model not ready")

    import torch
    t0 = time.time()

    pil = _pil_open(img_bytes)
    canvas, scale, pad_x, pad_y, ow, oh = _letterbox_pil(pil, INPUT_SIZE)
    np_img = _to_numpy(canvas)  # HWC RGB
    inp = torch.from_numpy(np_img).permute(2,0,1).unsqueeze(0).float() / 255.0

    with torch.no_grad():
        out = model(inp)
    arr = _unwrap_to_numpy(out)
    raw = _parse_yolo_output(arr, CONF_THRES)

    dets: List[Dict[str, Any]] = []
    for x1,y1,x2,y2,conf,cls in raw:
        # letterbox → 원본 좌표계로 역변환
        ox1 = max(0.0, min(ow, (x1 - pad_x)/scale))
        oy1 = max(0.0, min(oh, (y1 - pad_y)/scale))
        ox2 = max(0.0, min(ow, (x2 - pad_x)/scale))
        oy2 = max(0.0, min(oh, (y2 - pad_y)/scale))
        w = max(0.0, ox2-ox1)
        h = max(0.0, oy2-oy1)
        ci = int(round(cls))
        cname = LABELS[ci] if 0 <= ci < len(LABELS) else None
        dets.append({
            "classIndex": ci, "className": cname,
            "x": int(round(ox1)), "y": int(round(oy1)),
            "w": int(round(w)), "h": int(round(h)),
            "score": float(conf),
        })

    return {
        "class_names": LABELS,
        "detections": dets,
        "time_ms": int((time.time()-t0)*1000),
        "__serverImageW": ow, "__serverImageH": oh,
        "__input_size": INPUT_SIZE, "__model": model_kind,
    }

# ─────────────────────────────────────────────────────────────────────────────
# Worker (async 202)
# ─────────────────────────────────────────────────────────────────────────────
def _start_worker_once():
    def _worker():
        app.logger.info("[worker] started")
        while True:
            try:
                job_id, img_bytes, kind = _job_q.get()
                _jobs[job_id] = {"status": "running"}
                with _infer_lock:
                    out = run_inference(img_bytes)
                _jobs[job_id] = {"status": "done", "result": out}
            except Exception as e:
                _jobs[job_id] = {"status": "error", "error": str(e)}
            finally:
                try: del img_bytes
                except Exception: pass
                gc.collect()
                _job_q.task_done()
    threading.Thread(target=_worker, daemon=True).start()

def _startup_once(sync: bool = True):
    global _started_once
    if _started_once: return
    _started_once = True
    if sync:
        load_model_sync()        # ★ import 시 동기 로딩
    else:
        threading.Thread(target=load_model_sync, daemon=True).start()
    _start_worker_once()
    app.logger.info("[startup] background threads started")

# ★ import 시점에 바로 실행 → gunicorn import 중에 모델 로딩 완료
_startup_once(sync=True)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
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

def _wait_until_ready(timeout_sec: float = 8.0) -> bool:
    """detect 진입 시 최대 timeout_sec 동안 준비 대기"""
    t0 = time.time()
    while time.time() - t0 < timeout_sec:
        if ready and model_err is None and model_kind == "torchscript":
            return True
        time.sleep(0.2)
    return False

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

@app.post("/detect")
def detect():
    # 준비 대기
    if not (ready and model_err is None and model_kind == "torchscript"):
        if not _wait_until_ready(8.0):
            if model_err:
                app.logger.error(f"[detect] reject: model_err={model_err}")
                return jsonify(error=model_err), 500
            app.logger.warning("[detect] reject: still loading")
            return jsonify(error="loading"), 503

    img_bytes, kind = _read_image_from_request()
    if not img_bytes:
        return jsonify(error="no file"), 400

    # sync 요청이어도 바쁘면 즉시 큐로 (502 회피)
    want_sync = ALLOW_SYNC and (request.args.get("sync") == "1" or request.headers.get("X-Detect-Sync") == "1")
    if want_sync:
        if not _infer_lock.acquire(blocking=False):
            job_id = str(uuid.uuid4())
            _jobs[job_id] = {"status": "queued"}
            _job_q.put((job_id, img_bytes, kind or "multipart"))
            app.logger.info(f"[detect] busy->queue {job_id}")
            return jsonify(jobId=job_id), 202
        try:
            t0 = time.time()
            out = run_inference(img_bytes)
            dt = int((time.time() - t0) * 1000)
            app.logger.info(f"[detect] sync ok in {dt} ms, det={len(out.get('detections', []))}")
            return jsonify(out), 200
        except Exception as e:
            app.logger.exception("[detect] sync failed")
            return jsonify(error=str(e)), 500
        finally:
            try: del img_bytes
            except Exception: pass
            _infer_lock.release()
            gc.collect()

    # async 경로
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "queued"}
    _job_q.put((job_id, img_bytes, kind or "multipart"))
    app.logger.info(f"[detect] async queued {job_id}")
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
        return jsonify(error=j.get("error","unknown")), 500
    return jsonify(status=j["status"]), 202

# 로컬 실행용
if __name__ == "__main__":
    # import 시 이미 _startup_once(sync=True) 실행됨
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
