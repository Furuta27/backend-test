# app.py — TorchScript YOLO + 비동기 잡(202+jobId) + /jobs/:id + /detect-json
import os, io, time, threading, uuid, queue, logging, base64
from typing import List, Dict, Any
from flask import Flask, request, jsonify, redirect
from werkzeug.exceptions import HTTPException
from PIL import Image

try:
    from flask_cors import CORS
    CORS_AVAILABLE = True
except Exception:
    CORS_AVAILABLE = False

try:
    import torch, torchvision
    TV = True
except Exception:
    TV = False
    torch = None
    torchvision = None

PORT = int(os.environ.get("PORT", 10000))
MODEL_PATH = os.environ.get("MODEL_PATH", "best.torchscript.ptl")
CLASSES_PATH = os.environ.get("CLASSES_PATH", "labels.txt")
CONF_TH = float(os.environ.get("CONF_TH", "0.25"))
IOU_TH  = float(os.environ.get("IOU_TH", "0.45"))
TOP_K   = int(os.environ.get("TOP_K", "50"))
INPUT_SIZE = int(os.environ.get("INPUT_SIZE", "640"))
JOB_TTL_SEC = 15 * 60
WORKERS = int(os.environ.get("WORKERS", "2"))
MAX_CONTENT_LENGTH_MB = int(os.environ.get("MAX_CONTENT_LENGTH_MB", "12"))

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH_MB * 1024 * 1024
if CORS_AVAILABLE: CORS(app)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("app")

model = None
model_err = None
classes: List[str] = []
jobs: Dict[str, Dict[str, Any]] = {}
in_queue: "queue.Queue[tuple[str, bytes, str]]" = queue.Queue()

def load_classes():
    global classes
    if os.path.isfile(CLASSES_PATH):
        try:
            with open(CLASSES_PATH, "r", encoding="utf-8") as f:
                classes = [ln.strip() for ln in f if ln.strip()]
            log.info(f"[startup] loaded {len(classes)} classes")
        except Exception as e:
            log.warning(f"[startup] labels load fail: {e}")
    if not classes:
        classes = [f"cls_{i}" for i in range(100)]

def yolo_forward(x):
    y = model(x)
    return y[0] if isinstance(y, (list, tuple)) else y

def py_nms(boxes, scores, iou_th, top_k):
    import numpy as np
    if not boxes: return []
    b, s = np.array(boxes, np.float32), np.array(scores, np.float32)
    order = s.argsort()[::-1]; keep=[]
    while order.size>0:
        i = order[0]; keep.append(int(i))
        if len(keep)>=top_k: break
        xx1 = np.maximum(b[i,0], b[order[1:],0]); yy1 = np.maximum(b[i,1], b[order[1:],1])
        xx2 = np.minimum(b[i,2], b[order[1:],2]); yy2 = np.minimum(b[i,3], b[order[1:],3])
        w = np.maximum(0.0, xx2-xx1); h = np.maximum(0.0, yy2-yy1)
        inter = w*h
        ai = (b[i,2]-b[i,0])*(b[i,3]-b[i,1]); aj = (b[order[1:],2]-b[order[1:],0])*(b[order[1:],3]-b[order[1:],1])
        iou = inter/(ai+aj-inter+1e-6)
        inds = (iou<=iou_th).nonzero()[0]
        order = order[inds+1]
    return keep

def run_nms(boxes, scores, iou_th, top_k):
    if not boxes: return []
    if TV:
        with torch.no_grad():
            keep = torchvision.ops.nms(torch.tensor(boxes), torch.tensor(scores), iou_th)
            return keep[:top_k].tolist()
    return py_nms(boxes, scores, iou_th, top_k)

def preprocess(img: Image.Image):
    from torchvision.transforms.functional import to_tensor
    im = img.resize((INPUT_SIZE, INPUT_SIZE))
    return to_tensor(im).unsqueeze(0)  # [1,3,H,W]

def postprocess(out: "torch.Tensor", W: int, H: int):
    out = out.squeeze(0).cpu() if out.dim()==3 else out.cpu()
    boxes, scores, cls_idx = [], [], []
    for row in out:
        obj = float(row[4]); 
        if obj < 1e-6: continue
        cls_confs = row[5:]; ci = int(torch.argmax(cls_confs)); cc = float(cls_confs[ci])
        sc = obj*cc
        if sc < CONF_TH: continue
        cx, cy, w, h = [float(v) for v in row[:4]]
        x1 = max(0.0, cx - w/2); y1 = max(0.0, cy - h/2)
        x2 = min(float(INPUT_SIZE), cx + w/2); y2 = min(float(INPUT_SIZE), cy + h/2)
        boxes.append([x1,y1,x2,y2]); scores.append(sc); cls_idx.append(ci)
    keep = run_nms(boxes, scores, IOU_TH, TOP_K)
    sx, sy = W/INPUT_SIZE, H/INPUT_SIZE
    dets=[]
    for i in keep:
        x1,y1,x2,y2 = boxes[i]
        dets.append({
            "x": int(x1*sx), "y": int(y1*sy),
            "w": int((x2-x1)*sx), "h": int((y2-y1)*sy),
            "score": round(scores[i],4),
            "classIndex": int(cls_idx[i]),
            "className": classes[cls_idx[i]] if 0<=cls_idx[i]<len(classes) else str(cls_idx[i])
        })
    return dets

def warmup_and_load():
    global model, model_err
    try:
        load_classes()
        if not TV: raise RuntimeError("torch/torchvision not available")
        m = torch.jit.load(MODEL_PATH, map_location="cpu"); m.eval()
        with torch.no_grad(): _ = yolo_forward(torch.zeros(1,3,INPUT_SIZE,INPUT_SIZE))
        model = m; log.info("[startup] model ready")
    except Exception as e:
        model_err = str(e); log.exception("[startup] model load failed")

def worker_loop(idx: int):
    while True:
        job_id, data, filename = in_queue.get()
        meta = jobs.get(job_id, {})
        try:
            t0 = time.time()
            img = Image.open(io.BytesIO(data)).convert("RGB")
            W,H = img.size
            if not TV or model is None: dets=[]
            else:
                x = preprocess(img)
                with torch.no_grad(): raw = yolo_forward(x)
                dets = postprocess(raw, W, H)
            ms = int((time.time()-t0)*1000)
            jobs[job_id] = { "status":"done", "result": { "class_names": classes, "detections": dets, "time_ms": ms }, "t0": meta.get("t0", time.time()) }
            log.info(f"[job {job_id}] done {ms}ms det={len(dets)}")
        except Exception as e:
            jobs[job_id] = { "status":"error", "error": str(e), "t0": meta.get("t0", time.time()) }
            log.exception(f"[job {job_id}] error")

def start_workers(n: int):
    for i in range(max(1, n)):
        t = threading.Thread(target=worker_loop, args=(i,), daemon=True); t.start()
        log.info(f"[worker] started #{i}")

@app.before_first_request
def kickoff():
    threading.Thread(target=warmup_and_load, daemon=True).start()
    start_workers(WORKERS)
    def gc():
        while True:
            now = time.time()
            for k,v in list(jobs.items()):
                if now - v.get("t0", now) > JOB_TTL_SEC: jobs.pop(k, None)
            time.sleep(30)
    threading.Thread(target=gc, daemon=True).start()

@app.get("/")
def root(): return redirect("/health")

@app.get("/health")
def health():
    status = "ready" if model is not None else ("error" if model_err else "warming")
    return jsonify(status=status), 200

# 멀티파트 업로드
@app.post("/detect")
def detect():
    if model_err: return jsonify(error=f"model error: {model_err}"), 500
    if model is None: return jsonify(error="model loading, retry later"), 503
    f = request.files.get("file") or request.files.get("image")
    if not f: return jsonify(error='no file (fields: "file" or "image")'), 400
    data = f.read()
    if not data: return jsonify(error="empty file"), 400
    job_id = str(uuid.uuid4()); jobs[job_id] = { "status":"queued", "t0": time.time() }
    in_queue.put((job_id, data, getattr(f, "filename", "image")))
    return jsonify({ "jobId": job_id }), 202

# JSON(base64) 업로드
@app.post("/detect-json")
def detect_json():
    if model_err: return jsonify(error=f"model error: {model_err}"), 500
    if model is None: return jsonify(error="model loading, retry later"), 503
    j = request.get_json(silent=True) or {}
    b64 = j.get("data"); filename = j.get("filename") or "image.jpg"
    if not b64: return jsonify(error="no base64 'data' field"), 400
    try:
        data = base64.b64decode(b64)
    except Exception as e:
        return jsonify(error=f"invalid base64: {e}"), 400
    job_id = str(uuid.uuid4()); jobs[job_id] = { "status":"queued", "t0": time.time() }
    in_queue.put((job_id, data, filename))
    return jsonify({ "jobId": job_id }), 202

@app.get("/jobs/<job_id>")
def job(job_id: str):
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
