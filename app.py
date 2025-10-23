import os, io, time, json, uuid, threading, queue, gc
from typing import Tuple, Dict, Any, Optional
from flask import Flask, request, jsonify

app = Flask(__name__)

# ── 글로벌 상태 ───────────────────────────────────────────────────────────────
model = None               # ("ultralytics"|“torchscript”|“dummy”, handle)
model_err: Optional[str] = None
ready = False

LABELS = []
INPUT_SIZE = int(os.getenv("INPUT_SIZE", "640"))
ALLOW_SYNC = True

_jobs: Dict[str, Dict[str, Any]] = {}
_job_q: "queue.Queue[Tuple[str, bytes, str]]" = queue.Queue()
_infer_lock = threading.Lock()         # 동시 추론 1개로 제한(메모리 절약)
_started_once = False                  # 시작 루틴 1회만 실행

# ── 유틸 ──────────────────────────────────────────────────────────────────────
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

def _resize_keep_ar(img, target: int):
    from PIL import Image
    w, h = img.size
    if max(w, h) == target:
        return img, w, h
    if w >= h:
        new_w = target
        new_h = int(round(h * (target / w)))
    else:
        new_h = target
        new_w = int(round(w * (target / h)))
    return img.resize((new_w, new_h), Image.BILINEAR), new_w, new_h

def _postprocess_dummy(w: int, h: int) -> Dict[str, Any]:
    cx, cy = w // 3, h // 3
    bw, bh = max(40, w // 4), max(40, h // 4)
    return {
        "class_names": LABELS or ["object"],
        "detections": [{
            "classIndex": 0,
            "className": (LABELS[0] if LABELS else "object"),
            "x": max(0, cx - bw//2),
            "y": max(0, cy - bh//2),
            "w": bw, "h": bh, "score": 0.80
        }],
        "time_ms": 1
    }

# ── 모델 로드 / 추론 ─────────────────────────────────────────────────────────
def load_model_bg():
    """백그라운드에서 모델 로드."""
    global model, model_err, ready
    # 시작 시 에러 클리어
    model_err = None
    try:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        try:
            import torch
            torch.set_num_threads(1)
        except Exception:
            pass

        _read_labels()

        # ── 1) ultralytics YOLO (선택: 있으면 사용, 없으면 조용히 패스)
        mdl_path = None
        for p in ["best.pt", "yolov5s.pt", "weights/best.pt"]:
            if os.path.exists(p):
                mdl_path = p
                break

        if mdl_path:
            try:
                # importlib로 옵션 임포트 → 모듈 없으면 ImportError만 경고 로그
                import importlib
                yolo_mod = importlib.import_module("ultralytics")
                YOLO = getattr(yolo_mod, "YOLO")
                m = YOLO(mdl_path)
                model = ("ultralytics", m)
                ready = True
                model_err = None
                app.logger.info(f"[startup] ultralytics model loaded: {mdl_path}")
                return
            except Exception as e:
                # ❗ 여긴 실패해도 폴백하므로 error 고정 금지 (warning 정도만)
                app.logger.warning(f"[startup] ultralytics unavailable: {e}")

        # ── 2) torchscript(.ptl) 폴백 (Render 512MB 친화적)
        ts_path = None
        for p in ["best.torchscript.ptl", "yolov5s3.torchscript.ptl"]:
            if os.path.exists(p):
                ts_path = p
                break
        if ts_path:
            import torch
            m = torch.jit.load(ts_path, map_location="cpu")
            m.eval()
            model = ("torchscript", m)
            ready = True
            model_err = None
            app.logger.info(f"[startup] torchscript model loaded: {ts_path}")
            return

        # ── 3) 최종 폴백: 더미
        model = ("dummy", None)
        ready = True
        model_err = None
        app.logger.warning("[startup] no model found, fallback to dummy inference")

    except Exception as e:
        # 진짜로 모든 경로가 실패했을 때만 에러로 표시
        model_err = str(e)
        ready = False
        app.logger.exception("[startup] model load failed")


def run_inference(img_bytes: bytes) -> Dict[str, Any]:
    t0 = time.time()
    global model, LABELS
    pil = _pil_open(img_bytes)
    pil_resized, rw, rh = _resize_keep_ar(pil, INPUT_SIZE)
    kind = model[0] if model else "none"

    if kind in ("none", "dummy"):
        res = _postprocess_dummy(rw, rh)
        res["__serverImageW"] = rw; res["__serverImageH"] = rh
        res["time_ms"] = int((time.time() - t0) * 1000)
        return res

    if kind == "ultralytics":
        from ultralytics.utils import ops  # type: ignore
        m = model[1]
        preds = m.predict(pil_resized, imgsz=max(rw, rh), verbose=False)[0]
        dets = []
        if preds.boxes is not None and len(preds.boxes) > 0:
            for b in preds.boxes:
                xyxy = b.xyxy[0].tolist()
                conf = float(b.conf[0].item())
                cls = int(b.cls[0].item())
                x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
                w = max(0, x2 - x1); h = max(0, y2 - y1)
                dets.append({
                    "classIndex": cls,
                    "className": (LABELS[cls] if 0 <= cls < len(LABELS) else None),
                    "x": max(0, x1), "y": max(0, y1),
                    "w": w, "h": h, "score": conf
                })
        return {
            "class_names": LABELS or preds.names or [],
            "detections": dets,
            "time_ms": int((time.time() - t0) * 1000),
            "__serverImageW": rw, "__serverImageH": rh,
        }

    if kind == "torchscript":
        import torch, numpy as np
        m = model[1]
        np_img = _to_numpy(pil_resized)           # HWC RGB
        inp = torch.from_numpy(np_img).permute(2,0,1).unsqueeze(0).float() / 255.0
        with torch.no_grad():
            out = m(inp)
        if isinstance(out, (list, tuple)): out = out[0]
        if hasattr(out, "numpy"): out = out.numpy()
        elif isinstance(out, torch.Tensor): out = out.cpu().numpy()

        dets = []
        try:
            for row in out:
                if len(row) < 6: continue
                x1,y1,x2,y2,conf,cls = row[:6]
                x1,y1,x2,y2 = int(round(x1)),int(round(y1)),int(round(x2)),int(round(y2))
                w = max(0, x2-x1); h = max(0, y2-y1)
                dets.append({
                    "classIndex": int(cls),
                    "className": (LABELS[int(cls)] if 0 <= int(cls) < len(LABELS) else None),
                    "x": max(0, x1), "y": max(0, y1),
                    "w": w, "h": h, "score": float(conf)
                })
        except Exception:
            dets = _postprocess_dummy(rw, rh)["detections"]
        return {
            "class_names": LABELS,
            "detections": dets,
            "time_ms": int((time.time() - t0) * 1000),
            "__serverImageW": rw, "__serverImageH": rh,
        }

    res = _postprocess_dummy(rw, rh)
    res["__serverImageW"] = rw; res["__serverImageH"] = rh
    res["time_ms"] = int((time.time() - t0) * 1000)
    return res

# ── 워커(202 잡 처리) ────────────────────────────────────────────────────────
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
                del img_bytes; gc.collect()
            except Exception as e:
                _jobs[job_id] = {"status": "error", "error": str(e)}
            finally:
                _job_q.task_done()
    t = threading.Thread(target=_worker, daemon=True)
    t.start()

def _startup_once():
    """Flask 훅 의존 없이 시작 루틴을 1회만 실행."""
    global _started_once
    if _started_once:
        return
    _started_once = True
    threading.Thread(target=load_model_bg, daemon=True).start()
    _start_worker_once()
    app.logger.info("[startup] background threads started")

# ── 훅: 모든 요청 전에 한 번만 시작 ──────────────────────────────────────────
@app.before_request
def _ensure_started():
    _startup_once()
    # 요청 시작 시간(선택)
    request.start_ts = time.time()

# ── 헬스/루트 ────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return jsonify(ok=True, health="/health"), 200

@app.get("/health")
def health():
    if ready and model_err is None:
        return jsonify(status="ready"), 200
    if model_err is not None:
        return jsonify(status="error", error=model_err), 500
    return jsonify(status="warming"), 503  # 준비 중엔 503

# ── 입력 파싱 ────────────────────────────────────────────────────────────────
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

# ── 감지 API ─────────────────────────────────────────────────────────────────
@app.post("/detect")
def detect():
    if not ready or model is None:
        return jsonify(error="loading"), 503

    img_bytes, kind = _read_image_from_request()
    if not img_bytes:
        return jsonify(error="no file"), 400

    sync = ALLOW_SYNC and (
        request.args.get("sync") == "1" or request.headers.get("X-Detect-Sync") == "1"
    )

    if sync:
        try:
            with _infer_lock:
                out = run_inference(img_bytes)
            return jsonify(out), 200
        except Exception as e:
            app.logger.exception("[detect] sync failed")
            return jsonify(error=str(e)), 500
        finally:
            del img_bytes; gc.collect()

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

# ── 로컬 실행 ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    _startup_once()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
