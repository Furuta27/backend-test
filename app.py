import os, io, gc, uuid, time, threading, queue
from typing import Dict, Any, Optional, Tuple, List
from flask import Flask, request, jsonify

app = Flask(__name__)

# ── Config ───────────────────────────────────────────────────────────────────
TS_MODEL_PATH = os.getenv("TS_MODEL_PATH", "yolov5s4.torchscript.ptl")
INPUT_SIZE    = int(os.getenv("INPUT_SIZE", "640"))
CONF_THRES    = float(os.getenv("CONF_THRES", "0.25"))
HIST_KEEP     = int(os.getenv("HIST_KEEP", "20"))   # 평균 추론시간 계산용 저장 개수

# ── Global state ─────────────────────────────────────────────────────────────
model_kind: str = "none"   # "torchscript" | "none"
model = None               # torchscript jit module
model_err: Optional[str] = None
ready: bool = False
LABELS: List[str] = []

_jobs: Dict[str, Dict[str, Any]] = {}
_job_q: "queue.Queue[Tuple[str, bytes, str, float]]" = queue.Queue()
_infer_lock = threading.Lock()
_started_once = False

_last_durations_ms: List[int] = []       # 최근 추론 시간
_avg_ms: int = 0

# ── Utils ────────────────────────────────────────────────────────────────────
def _read_labels():
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
    import numpy as np
    if arr is None:
        return []
    a = arr
    if isinstance(a, (list, tuple)):
        a = np.array(a)
    if a.ndim == 3 and a.shape[0] == 1:
        a = a[0]
    if not (a.ndim == 2 and a.shape[1] >= 6):
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

def _update_hist(ms: int):
    global _avg_ms
    _last_durations_ms.append(ms)
    if len(_last_durations_ms) > HIST_KEEP:
        del _last_durations_ms[0:len(_last_durations_ms)-HIST_KEEP]
    _avg_ms = int(sum(_last_durations_ms) / max(1, len(_last_durations_ms)))

# ── Model load / inference ───────────────────────────────────────────────────
def load_model_sync():
    global model, model_kind, model_err, ready
    model = None; model_kind = "none"; model_err = None; ready = False
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
            app.logger.error("[startup] " + model_err); return

        import torch
        m = torch.jit.load(TS_MODEL_PATH, map_location="cpu")
        m.eval()
        with torch.no_grad():
            _ = m(torch.zeros(1,3,INPUT_SIZE,INPUT_SIZE))
        model = m; model_kind = "torchscript"; ready = True
        app.logger.info(f"[startup] torchscript loaded: {TS_MODEL_PATH}")
    except Exception as e:
        model = None; model_kind = "none"; model_err = str(e); ready = False
        app.logger.exception("[startup] model load failed")

def run_inference(img_bytes: bytes) -> Dict[str, Any]:
    if not ready or model is None or model_kind != "torchscript":
        raise RuntimeError("model not ready")
    import torch
    t0 = time.time()
    pil = _pil_open(img_bytes)
    canvas, scale, pad_x, pad_y, ow, oh = _letterbox_pil(pil, INPUT_SIZE)
    np_img = _to_numpy(canvas)
    inp = torch.from_numpy(np_img).permute(2,0,1).unsqueeze(0).float() / 255.0
    with torch.no_grad():
        out = model(inp)
    arr = _unwrap_to_numpy(out)
    raw = _parse_yolo_output(arr, CONF_THRES)

    dets: List[Dict[str, Any]] = []
    for x1,y1,x2,y2,conf,cls in raw:
        ox1 = max(0.0, min(ow, (x1 - pad_x)/scale))
        oy1 = max(0.0, min(oh, (y1 - pad_y)/scale))
        ox2 = max(0.0, min(ow, (x2 - pad_x)/scale))
        oy2 = max(0.0, min(oh, (y2 - pad_y)/scale))
        w = max(0.0, ox2-ox1); h = max(0.0, oy2-oy1)
        ci = int(round(cls))
        cname = LABELS[ci] if 0 <= ci < len(LABELS) else None
        dets.append({
            "classIndex": ci, "className": cname,
            "x": int(round(ox1)), "y": int(round(oy1)),
            "w": int(round(w)), "h": int(round(h)),
            "score": float(conf),
        })
    ms = int((time.time()-t0)*1000)
    _update_hist(ms)
    return {
        "class_names": LABELS,
        "detections": dets,
        "time_ms": ms,
        "__serverImageW": ow, "__serverImageH": oh,
        "__input_size": INPUT_SIZE, "__model": model_kind,
    }

# ── Worker & startup ─────────────────────────────────────────────────────────
def _start_worker_once():
    def _worker():
        app.logger.info("[worker] started")
        while True:
            try:
                job_id, img_bytes, kind, queued_at = _job_q.get()
                _jobs[job_id].update({"status": "running", "started_at": time.time()})
                with _infer_lock:
                    out = run_inference(img_bytes)
                _jobs[job_id].update({
                    "status": "done",
                    "result": out,
                    "finished_at": time.time()
                })
            except Exception as e:
                _jobs[job_id].update({"status": "error", "error": str(e), "finished_at": time.time()})
            finally:
                try: del img_bytes
                except Exception: pass
                gc.collect()
                _job_q.task_done()
    threading.Thread(target=_worker, daemon=True).start()

def _startup_once():
    global _started_once
    if _started_once: return
    _started_once = True
    load_model_sync()          # 동기 로드
    _start_worker_once()
    app.logger.info("[startup] background threads started")

_startup_once()

# ── Helpers ──────────────────────────────────────────────────────────────────
def _read_image_from_request() -> Tuple[Optional[bytes], Optional[str]]:
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

def _queue_position(job_id: str) -> int:
    # queued_at 기준으로 정렬하여 현재 job의 앞선 queued 수 + running 0/1 반영
    if job_id not in _jobs: return -1
    me = _jobs[job_id]
    if me["status"] != "queued":
        return 0
    my_t = me.get("queued_at", 0.0)
    # 자신보다 먼저 큐잉된 'queued' 건수
    ahead = sum(1 for j in _jobs.values() if j.get("status") == "queued" and j.get("queued_at", 1e15) < my_t)
    # 현재 실행 중이 있으면 +1 (자신 앞에서 하나 처리 중)
    running = any(j.get("status") == "running" for j in _jobs.values())
    return ahead + (1 if running else 0)

# ── Routes ───────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return jsonify(ok=True, health="/health"), 200

@app.get("/health")
def health():
    status = "ready" if ready and model_err is None else ("error" if model_err else "warming")
    info = {
        "status": status, "model": model_kind,
        "model_file": TS_MODEL_PATH if model_kind == "torchscript" else None,
        "labels_count": len(LABELS), "input_size": INPUT_SIZE,
        "conf_thres": CONF_THRES, "error": model_err,
        "avg_time_ms": _avg_ms,
    }
    code = 200 if status == "ready" else (500 if status == "error" else 503)
    return jsonify(info), code

@app.post("/detect")
def detect():
    # 항상 202 비동기
    if not (ready and model_err is None and model_kind == "torchscript"):
        code = 500 if model_err else 503
        return jsonify(error=model_err or "loading"), code

    img_bytes, kind = _read_image_from_request()
    if not img_bytes:
        return jsonify(error="no file"), 400

    job_id = str(uuid.uuid4())
    now = time.time()
    _jobs[job_id] = {
        "status": "queued",
        "queued_at": now,
    }
    _job_q.put((job_id, img_bytes, kind or "multipart", now))

    pos = _queue_position(job_id)
    # 평균 추론 시간으로 대략적 ETA 추정
    eta_ms = max(0, pos) * max(1, _avg_ms or 3000)

    app.logger.info(f"[detect] queued {job_id} pos={pos} eta≈{eta_ms}ms")
    return jsonify(jobId=job_id, status="queued", position=pos, eta_ms=eta_ms), 202

@app.get("/jobs/<job_id>")
def job_status(job_id: str):
    j = _jobs.get(job_id)
    if not j:
        return jsonify(error="not found"), 404
    s = j["status"]
    if s == "done":
        return jsonify(j["result"]), 200
    if s == "error":
        return jsonify(error=j.get("error","unknown")), 500
    # queued or running
    now = time.time()
    if s == "queued":
        pos = _queue_position(job_id)
        waited = int((now - j.get("queued_at", now)) * 1000)
        eta_ms = max(0, pos) * max(1, _avg_ms or 3000)
        return jsonify(status="queued", position=pos, waited_ms=waited, eta_ms=eta_ms), 202
    if s == "running":
        started = j.get("started_at", now)
        running_for = int((now - started) * 1000)
        return jsonify(status="running", running_ms=running_for, avg_time_ms=_avg_ms), 202
    return jsonify(status=s), 202

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
