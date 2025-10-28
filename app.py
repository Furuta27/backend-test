import os, io, time, uuid, threading, queue, gc, hashlib
from typing import Tuple, Dict, Any, Optional
from flask import Flask, request, jsonify

app = Flask(__name__)

# ── 글로벌 상태 (감지 서버) ───────────────────────────────────────────────────
model: Optional[tuple] = None       # ("torchscript" | "dummy", handle)
model_err: Optional[str] = None
ready = False

LABELS = []
INPUT_SIZE = int(os.getenv("INPUT_SIZE", "640"))   # 416/384로 낮추면 메모리/502 완화
ALLOW_SYNC = False                                  # 항상 비동기(202 + /jobs 폴링)

_jobs: Dict[str, Dict[str, Any]] = {}
_job_q: "queue.Queue[Tuple[str, bytes, str]]" = queue.Queue()
_infer_lock = threading.Lock()         # 동시 추론 1개 제한(메모리 절약)
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
    # 디버그용 더미 박스 1개 (이미 Top-1 형태)
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
    """오직 yolov5s3.torchscript.ptl (또는 MODEL_TS_PATH)만 로드. .pt/ultralytics는 사용하지 않음."""
    global model, model_err, ready

    model = None
    model_err = None
    ready = False

    try:
        # 스레드/옵션 최소화(메모리 절약)
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        try:
            import torch
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            torch.set_grad_enabled(False)  # type: ignore[attr-defined]
            try:
                import torch.backends.mkldnn as mkldnn  # type: ignore
                mkldnn.enabled = False  # type: ignore[attr-defined]
            except Exception:
                pass
        except Exception:
            pass

        _read_labels()

        # TorchScript 경로만 허용
        ts_path = os.getenv("MODEL_TS_PATH", "yolov5s3.torchscript.ptl")
        if not os.path.exists(ts_path):
            model_err = f"torchscript model not found: {ts_path}"
            ready = False
            app.logger.error(f"[startup] {model_err}")
            return

        # TorchScript 로드 (CPU)
        import torch
        m = torch.jit.load(ts_path, map_location="cpu")
        m.eval()
        model = ("torchscript", m)
        ready = True
        model_err = None
        app.logger.info(f"[startup] torchscript model loaded: {ts_path}")

    except Exception as e:
        model = None
        ready = False
        model_err = str(e)
        app.logger.exception("[startup] torchscript load failed")

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

    if kind == "torchscript":
        import torch
        m = model[1]
        np_img = _to_numpy(pil_resized)           # HWC RGB
        inp = torch.from_numpy(np_img).permute(2,0,1).unsqueeze(0).float() / 255.0
        with torch.no_grad():
            out = m(inp)

        # 다양한 TorchScript export 포맷 대응 → numpy/리스트로 정규화
        if isinstance(out, (list, tuple)):
            out = out[0]
        if isinstance(out, torch.Tensor):
            out = out.cpu().numpy()
        try:
            arr = out
            if hasattr(arr, "shape") and len(arr.shape) == 3:
                arr = arr[0]              # (1,N,6) → (N,6)
        except Exception:
            arr = out

        dets = []
        try:
            for row in arr:
                if len(row) < 6:
                    continue
                x1,y1,x2,y2,conf,cls = row[:6]
                x1,y1,x2,y2 = int(round(float(x1))), int(round(float(y1))), int(round(float(x2))), int(round(float(y2)))
                w = max(0, x2-x1); h = max(0, y2-y1)
                dets.append({
                    "classIndex": int(cls),
                    "className": (LABELS[int(cls)] if 0 <= int(cls) < len(LABELS) else None),
                    "x": max(0, x1), "y": max(0, y1),
                    "w": w, "h": h, "score": float(conf)
                })
        except Exception:
            dets = _postprocess_dummy(rw, rh)["detections"]

        # Top-1만 남기기
        dets.sort(key=lambda d: float(d.get("score", 0.0)), reverse=True)
        dets = dets[:1]

        return {
            "class_names": LABELS,
            "detections": dets,
            "time_ms": int((time.time() - t0) * 1000),
            "__serverImageW": rw, "__serverImageH": rh,
        }

    # 이외는 더미
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

# ── 감지 API (항상 비동기 큐) ────────────────────────────────────────────────
@app.post("/detect")
def detect():
    if not ready or model is None:
        code = 500 if model_err else 503
        return jsonify(error="loading" if code == 503 else model_err), code

    img_bytes, kind = _read_image_from_request()
    if not img_bytes:
        return jsonify(error="no file"), 400

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

# ── 공공데이터 포털 프록시 (인코딩/디코딩 자동 대처) ─────────────────────────
# ENV:
#   GOV_SERVICE_KEY_DEC : (권장) 디코딩된 일반 키
#   GOV_SERVICE_KEY_ENC : (선택) 인코딩된 키
#   GOV_BASE            : 기본 https://apis.data.go.kr
#   GOV_CACHE_TTL_SEC   : 기본 300초

import requests, urllib.parse as _url
try:
    import xmltodict  # pip install xmltodict
except Exception:
    xmltodict = None

GOV_BASE = os.getenv("GOV_BASE", "https://apis.data.go.kr").rstrip("/")
GOV_KEY_DEC = os.getenv("GOV_SERVICE_KEY_DEC", "")
GOV_KEY_ENC = os.getenv("GOV_SERVICE_KEY_ENC", "")
GOV_TTL  = int(os.getenv("GOV_CACHE_TTL_SEC", "300"))

_gov_cache: Dict[str, Dict[str, Any]] = {}

def _cache_get(k: str):
    v = _gov_cache.get(k)
    if not v: return None
    if time.time() - v["ts"] > GOV_TTL:
        try: del _gov_cache[k]
        except Exception: pass
        return None
    return v["data"]

def _cache_set(k: str, data: Any):
    _gov_cache[k] = {"ts": time.time(), "data": data}

def _safe_jsonify(data: Any):
    try:
        return jsonify(data), 200
    except Exception:
        return jsonify(raw=str(data)[:2000]), 200

def _ok_json_or_xml(r: requests.Response):
    ct = (r.headers.get("content-type") or "").lower()
    if "application/json" in ct:
        try: return True, r.json()
        except Exception: return False, r.text
    if "xml" in ct and xmltodict is not None:
        try: return True, xmltodict.parse(r.text)
        except Exception: return False, r.text
    return True, {"status_code": r.status_code, "content": r.text[:5000]}

def _looks_like_key_error(payload: Any) -> bool:
    s = str(payload).lower()
    return ("service key" in s and ("invalid" in s or "not" in s or "등록" in s or "인증" in s))

def _call_gov(url: str, qs: Dict[str, Any]) -> Any:
    # 우선순위: 쿼리에 이미 있으면 그대로 → 없으면 DEC → ENC
    if "serviceKey" in qs:
        candidates = [qs["serviceKey"]]
    else:
        dec = GOV_KEY_DEC.strip()
        enc = (GOV_KEY_ENC.strip() or _url.quote_plus(dec)) if dec else GOV_KEY_ENC.strip()
        candidates = [dec, enc]
        candidates = [c for i, c in enumerate(candidates) if c and c not in candidates[:i]]

    last_error = None
    for key in candidates:
        q = dict(qs)
        q["serviceKey"] = key
        try:
            r = requests.get(url, params=q, timeout=15)
            ok, payload = _ok_json_or_xml(r)
            if r.status_code == 200 and ok and not _looks_like_key_error(payload):
                return payload
            last_error = payload
        except requests.Timeout:
            last_error = {"error": "upstream timeout"}
        except Exception as e:
            last_error = {"error": str(e)}
    return {"error": "key_failed", "detail": last_error}

@app.get("/govapi/<path:subpath>")
def gov_proxy(subpath: str):
    if not (GOV_KEY_DEC or GOV_KEY_ENC):
        return jsonify(error="server not configured: GOV_SERVICE_KEY_DEC/ENC"), 500

    qs = dict(request.args)
    qs.setdefault("returnType", "json")
    qs.setdefault("type", "json")

    url = f"{GOV_BASE}/{subpath.lstrip('/')}"

    key_raw = url + "?" + "&".join(sorted([f"{k}={qs[k]}" for k in qs]))
    h = hashlib.sha1(key_raw.encode("utf-8")).hexdigest()
    cached = _cache_get(h)
    if cached is not None:
        return _safe_jsonify(cached)

    data = _call_gov(url, qs)
    _cache_set(h, data)
    return _safe_jsonify(data)

# ── 로컬 실행 ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    _startup_once()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
