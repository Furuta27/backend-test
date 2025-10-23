# app.py — Low-mem Flask backend (Render 512MB 대응) + Sync 모드
# - /health : 상태 체크 (ready/warming/error)
# - /detect : 멀티파트 업로드 (file|image)  [sync=1 지원 → 즉시 200]
# - /detect-json (/detect_json) : JSON(base64) 업로드 [sync=1 지원]
# - /jobs/<id> : 비동기 모드 폴링용 (202/200)
# - before_first_request (Flask 3.x) 호환 shim 포함
# - torchvision 불필요(NumPy NMS + PIL 전처리), 큐는 "파일 경로"만 저장해 메모리 절약

import os, io, time, uuid, base64, queue, threading, logging, tempfile, gc
from typing import List, Dict, Any, Tuple

from flask import Flask, request, jsonify, redirect
from werkzeug.exceptions import HTTPException
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 20_000_000

# 선택적 CORS
try:
    from flask_cors import CORS
    CORS_AVAILABLE = True
except Exception:
    CORS_AVAILABLE = False

# torch만 사용(vision 미사용)
try:
    import torch  # type: ignore
    TORCH_OK = True
    try:
        torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "1")))  # type: ignore
    except Exception:
        pass
except Exception:
    torch = None  # type: ignore
    TORCH_OK = False

import numpy as np

# ───────────────────── 설정 ─────────────────────
PORT = int(os.environ.get("PORT", "10000"))
MODEL_PATH = os.environ.get("MODEL_PATH", "best.torchscript.ptl")
CLASSES_PATH = os.environ.get("CLASSES_PATH", "labels.txt")

CONF_TH = float(os.environ.get("CONF_TH", "0.25"))
IOU_TH  = float(os.environ.get("IOU_TH",  "0.45"))
TOP_K   = int(os.environ.get("TOP_K", "50"))
INPUT_SIZE = int(os.environ.get("INPUT_SIZE", "640"))

WORKERS = int(os.environ.get("WORKERS", "1"))                 # 기본 1 (저메모리)
JOB_TTL_SEC = int(os.environ.get("JOB_TTL_SEC", str(15*60)))
MAX_CONTENT_LENGTH_MB = int(os.environ.get("MAX_CONTENT_LENGTH_MB", "4"))  # 기본 4MB
QUEUE_MAX = int(os.environ.get("QUEUE_MAX", "2"))             # 큐 길이 제한(스파이크 방지)

# ───────────────────── Flask ─────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH_MB * 1024 * 1024
if CORS_AVAILABLE: CORS(app)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("app")

# Flask 3.x 호환: before_first_request shim
if not hasattr(app, "before_first_request"):
    _lock = threading.Lock()
    _done = {"v": False}
    def _bffr(func):
        def _guard(*a, **k):
            if _done["v"]: return
            with _lock:
                if _done["v"]: return
                _done["v"] = True
                return func()
        app.before_request(_guard)
        return func
    app.before_first_request = _bffr  # type: ignore

# ───────────────────── 전역 상태 ─────────────────────
model = None  # type: ignore
model_err: str | None = None
classes: List[str] = []

# 경로 기반 큐: (job_id, path, filename)
in_queue: "queue.Queue[Tuple[str, str, str]]" = queue.Queue(maxsize=QUEUE_MAX)
jobs: Dict[str, Dict[str, Any]] = {}

# ───────────────────── 유틸 ─────────────────────
def load_classes():
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

def preprocess_np(img: Image.Image):
    im = img.resize((INPUT_SIZE, INPUT_SIZE))
    arr = np.asarray(im, dtype=np.float32) / 255.0      # [H,W,3]
    arr = np.transpose(arr, (2, 0, 1))                  # [3,H,W]
    x = torch.from_numpy(np.ascontiguousarray(arr)).unsqueeze(0)  # type: ignore
    return x

def nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_th=0.45, top_k=50):
    if boxes.size == 0: return []
    x1,y1,x2,y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    areas = (x2-x1) * (y2-y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = int(order[0]); keep.append(i)
        if len(keep) >= top_k: break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2-xx1)
        h = np.maximum(0.0, yy2-yy1)
        inter = w*h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_th)[0]
        order = order[inds+1]
    return keep

def postprocess(out, W:int, H:int):
    if torch is None or out is None: return []
    t = out.squeeze(0).cpu().numpy() if hasattr(out, "cpu") else np.array(out)
    boxes, scores, cls_idx = [], [], []
    for row in t:
        obj = float(row[4])
        if obj < 1e-6: continue
        cls_confs = row[5:]
        ci = int(np.argmax(cls_confs))
        sc = obj * float(cls_confs[ci])
        if sc < CONF_TH: continue
        cx, cy, w, h = [float(v) for v in row[:4]]
        x1 = max(0.0, cx - w/2); y1 = max(0.0, cy - h/2)
        x2 = min(float(INPUT_SIZE), cx + w/2); y2 = min(float(INPUT_SIZE), cy + h/2)
        boxes.append([x1,y1,x2,y2]); scores.append(sc); cls_idx.append(ci)
    if not boxes: return []
    b = np.array(boxes, dtype=np.float32); s = np.array(scores, dtype=np.float32)
    keep = nms_numpy(b, s, IOU_TH, TOP_K)
    sx, sy = W/INPUT_SIZE, H/INPUT_SIZE
    dets = []
    for i in keep:
        x1,y1,x2,y2 = b[i]
        dets.append({
            "x": int(x1*sx), "y": int(y1*sy),
            "w": int((x2-x1)*sx), "h": int((y2-y1)*sy),
            "score": round(float(s[i]),4),
            "classIndex": int(cls_idx[i]),
            "className": classes[cls_idx[i]] if 0 <= cls_idx[i] < len(classes) else str(cls_idx[i]),
        })
    return dets

def run_detect_bytes(data: bytes, filename: str):
    """업로드 바이트를 즉시 인퍼런스 → 결과 JSON 딕셔너리 반환 (Sync 모드)"""
    t0 = time.time()
    img = Image.open(io.BytesIO(data)).convert("RGB")
    try:
        W, H = img.size
        if not TORCH_OK or model is None:
            dets = []
        else:
            x = preprocess_np(img)
            with torch.no_grad():     # type: ignore
                raw = model(x)        # type: ignore
            dets = postprocess(raw, W, H)
            del x, raw
        ms = int((time.time() - t0) * 1000)
        return {"class_names": classes, "detections": dets, "time_ms": ms}
    finally:
        try: img.close()
        except: pass

def warmup_and_load():
    global model, model_err
    try:
        load_classes()
        if not TORCH_OK: raise RuntimeError("torch not available")
        m = torch.jit.load(MODEL_PATH, map_location="cpu")   # type: ignore
        m.eval()                                             # type: ignore
        with torch.no_grad():                                # type: ignore
            _ = m(torch.zeros(1,3,INPUT_SIZE,INPUT_SIZE))    # type: ignore
        model = m; model_err = None
        log.info("[startup] model ready")
    except Exception as e:
        model = None; model_err = str(e)
        log.exception("[startup] model load failed")

def worker_loop(idx:int):
    while True:
        job_id, path, filename = in_queue.get()
        meta = jobs.get(job_id, {})
        img = None
        try:
            t0 = time.time()
            img = Image.open(path).convert("RGB")
            W,H = img.size
            if not TORCH_OK or model is None:
                dets = []
            else:
                x = preprocess_np(img)
                with torch.no_grad(): raw = model(x)         # type: ignore
                dets = postprocess(raw, W, H)
                del x, raw
            ms = int((time.time()-t0)*1000)
            jobs[job_id] = {"status":"done","result":{"class_names":classes,"detections":dets,"time_ms":ms},"t0":meta.get("t0",time.time())}
            log.info(f"[job {job_id}] done {ms}ms det={len(dets)}")
        except Exception as e:
            jobs[job_id] = {"status":"error","error":str(e),"t0":meta.get("t0", time.time())}
            log.exception(f"[job {job_id}] error")
        finally:
            try:
                if img: img.close()
            except: pass
            try: os.remove(path)
            except: pass
            gc.collect()

def start_workers(n:int):
    n = max(1, n)
    for i in range(n):
        threading.Thread(target=worker_loop, args=(i,), daemon=True).start()
        log.info(f"[worker] started #{i}")

def _gc_forever():
    while True:
        now = time.time()
        for k,v in list(jobs.items()):
            if now - v.get("t0", now) > JOB_TTL_SEC:
                jobs.pop(k, None)
        gc.collect()
        time.sleep(30)

@app.before_first_request
def kickoff():
    threading.Thread(target=warmup_and_load, daemon=True).start()
    start_workers(WORKERS)
    threading.Thread(target=_gc_forever, daemon=True).start()
    log.info("[startup] background threads started (low-mem + sync)")

# ───────────────────── 라우팅 ─────────────────────
@app.get("/")
def root(): return redirect("/health")

@app.get("/health")
def health():
    status = "ready" if model is not None else ("error" if model_err else "warming")
    body = {"status": status}
    if model_err: body["error"] = model_err
    return jsonify(body), 200

@app.post("/detect")
def detect():
    if model_err: return jsonify(error=f"model error: {model_err}"), 500
    if model is None: return jsonify(error="model loading, retry later"), 503

    f = request.files.get("file") or request.files.get("image")
    if not f: return jsonify(error='no file (fields: "file" or "image")'), 400
    data = f.read()
    if not data: return jsonify(error="empty file"), 400

    # ✅ Sync 모드: ?sync=1 또는 헤더 X-Detect-Sync: 1
    sync = (request.args.get("sync") in ("1", "true", "yes")) or (request.headers.get("X-Detect-Sync") == "1")
    if sync:
        try:
            out = run_detect_bytes(data, getattr(f, "filename", "image.jpg"))
            return jsonify(out), 200
        except Exception as e:
            return jsonify(error=str(e)), 500

    # ⬇️ 비동기 모드 (경로 큐)
    if in_queue.full(): return jsonify(error="busy", retryAfterMs=3000), 429
    suffix = os.path.splitext(getattr(f, "filename", "image.jpg"))[1] or ".jpg"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(data); tmp.flush(); tmp.close()
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status":"queued","t0":time.time()}
    in_queue.put((job_id, tmp.name, getattr(f,"filename","image.jpg")))
    return jsonify({"jobId": job_id}), 202

@app.post("/detect-json")
def detect_json():
    if model_err: return jsonify(error=f"model error: {model_err}"), 500
    if model is None: return jsonify(error="model loading, retry later"), 503

    j = request.get_json(silent=True) or {}
    b64 = j.get("data"); filename = j.get("filename") or "image.jpg"
    if not b64: return jsonify(error="no base64 'data' field"), 400
    try:
        data = base64.b64decode(b64, validate=True)
    except Exception as e:
        return jsonify(error=f"invalid base64: {e}"), 400

    # ✅ Sync 모드
    sync = (request.args.get("sync") in ("1", "true", "yes")) or (request.headers.get("X-Detect-Sync") == "1")
    if sync:
        try:
            out = run_detect_bytes(data, filename)
            return jsonify(out), 200
        except Exception as e:
            return jsonify(error=str(e)), 500

    # ⬇️ 비동기 모드 (경로 큐)
    if in_queue.full(): return jsonify(error="busy", retryAfterMs=3000), 429
    suffix = os.path.splitext(filename)[1] or ".jpg"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(data); tmp.flush(); tmp.close()
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status":"queued","t0":time.time()}
    in_queue.put((job_id, tmp.name, filename))
    return jsonify({"jobId": job_id}), 202

@app.post("/detect_json")
def detect_json_alias():
    return detect_json()

@app.get("/jobs/<job_id>")
def job(job_id:str):
    meta = jobs.get(job_id)
    if not meta: return jsonify(error="job not found"), 404
    if meta["status"] == "done": return jsonify(meta["result"]), 200
    if meta["status"] == "error": return jsonify(error=meta.get("error","job failed")), 500
    return jsonify(status=meta["status"]), 202

@app.errorhandler(Exception)
def handle_ex(e):
    if isinstance(e, HTTPException): return jsonify(error=str(e)), (e.code or 500)
    return jsonify(error="internal error"), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
