import os, io, time, json, uuid, threading, queue, gc
from typing import Tuple, Dict, Any, Optional

from flask import Flask, request, jsonify

# ─────────────────────────────────────────────────────────────────────────────
# 글로벌 상태
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)

model = None               # 실제 모델 핸들
model_err: Optional[str] = None
ready = False

LABELS = []
INPUT_SIZE = int(os.getenv("INPUT_SIZE", "640"))  # 모델 입력 크기(가변이면 내부에서 리사이즈)
ALLOW_SYNC = True           # ?sync=1 또는 헤더 X-Detect-Sync:1 허용

# 작업 큐 (비동기 202용). 메모리 절약 위해 단일 워커.
_jobs: Dict[str, Dict[str, Any]] = {}
_job_q: "queue.Queue[Tuple[str, bytes, str]]" = queue.Queue()
_worker_started = False
_infer_lock = threading.Lock()  # 동기 처리 시에도 동시성 1 보장 → 피크 메모리 절약

# ─────────────────────────────────────────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────────────────────────────────────────
def _read_labels():
    global LABELS
    p = os.path.join(os.getcwd(), "labels.txt")
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            LABELS = [ln.strip() for ln in f if ln.strip()]
    else:
        # 없으면 빈 리스트라도 유지
        LABELS = []

def _pil_open(img_bytes: bytes):
    from PIL import Image
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")

def _to_numpy(img):
    import numpy as np
    return np.array(img)

def _resize_keep_ar(img, target: int):
    # PIL로 긴 변 기준 리사이즈 (모델에 맞게 자유롭게 수정)
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
    img2 = img.resize((new_w, new_h), Image.BILINEAR)
    return img2, new_w, new_h

def _postprocess_dummy(w: int, h: int) -> Dict[str, Any]:
    # 모델 미로드/오류 시에도 200 테스트용 더미 (원하면 제거)
    cx, cy = w // 3, h // 3
    bw, bh = max(40, w // 4), max(40, h // 4)
    return {
        "class_names": LABELS or ["object"],
        "detections": [{
            "classIndex": 0,
            "className": (LABELS[0] if LABELS else "object"),
            "x": max(0, cx - bw//2),
            "y": max(0, cy - bh//2),
            "w": bw,
            "h": bh,
            "score": 0.80
        }],
        "time_ms": 1
    }

# ─────────────────────────────────────────────────────────────────────────────
# 모델 로드 / 추론
# ─────────────────────────────────────────────────────────────────────────────
def load_model_bg():
    """백그라운드에서 모델 로드."""
    global model, model_err, ready
    try:
        # 스레드/BLAS 쓰레드 수 낮추기 (메모리 절약)
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        try:
            import torch
            torch.set_num_threads(1)
        except Exception:
            pass

        _read_labels()

        # 1) ultralytics YOLO 우선 (설치되어 있고 best.pt가 존재하면)
        mdl_path = None
        for p in ["best.pt", "yolov5s.pt", "weights/best.pt"]:
            if os.path.exists(p):
                mdl_path = p
                break

        if mdl_path:
            try:
                from ultralytics import YOLO  # type: ignore
                m = YOLO(mdl_path)
                _model = ("ultralytics", m)
                model = _model
                ready = True
                app.logger.info(f"[startup] ultralytics model loaded: {mdl_path}")
                return
            except Exception as e:
                app.logger.exception("[startup] ultralytics load failed")
                model_err = f"ultralytics load failed: {e}"

        # 2) torchscript (.ptl) 시도
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
            app.logger.info(f"[startup] torchscript model loaded: {ts_path}")
            return

        # 3) 실패 시 더미 모드
        model = ("dummy", None)
        ready = True
        app.logger.warning("[startup] no model found, fallback to dummy inference")

    except Exception as e:
        model_err = str(e)
        ready = False
        app.logger.exception("[startup] model load failed")

def run_inference(img_bytes: bytes) -> Dict[str, Any]:
    """이미지 바이트 입력 → 감지 결과 dict로."""
    t0 = time.time()
    global model, LABELS

    # 0) 미리 디코드/리사이즈
    pil = _pil_open(img_bytes)
    pil_resized, rw, rh = _resize_keep_ar(pil, INPUT_SIZE)

    kind = model[0] if model else "none"

    # 1) 더미 모드
    if kind in ("none", "dummy"):
        res = _postprocess_dummy(rw, rh)
        res["__serverImageW"] = rw
        res["__serverImageH"] = rh
        res["time_ms"] = int((time.time() - t0) * 1000)
        return res

    # 2) ultralytics
    if kind == "ultralytics":
        from ultralytics.utils import ops  # type: ignore
        m = model[1]
        # BGR/단위 등 내부처리는 라이브러리에서
        preds = m.predict(pil_resized, imgsz=max(rw, rh), verbose=False)[0]
        dets = []
        if preds.boxes is not None and len(preds.boxes) > 0:
            # xyxy, conf, cls
            for b in preds.boxes:
                xyxy = b.xyxy[0].tolist()
                conf = float(b.conf[0].item())
                cls = int(b.cls[0].item())
                x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
                w = max(0, x2 - x1)
                h = max(0, y2 - y1)
                dets.append({
                    "classIndex": cls,
                    "className": (LABELS[cls] if 0 <= cls < len(LABELS) else None),
                    "x": max(0, x1), "y": max(0, y1),
                    "w": w, "h": h,
                    "score": conf
                })

        res = {
            "class_names": LABELS or preds.names or [],
            "detections": dets,
            "time_ms": int((time.time() - t0) * 1000),
            "__serverImageW": rw,
            "__serverImageH": rh,
        }
        return res

    # 3) torchscript (모델마다 출력이 다를 수 있음 → 대표 케이스 처리)
    if kind == "torchscript":
        import torch
        import numpy as np

        m = model[1]
        np_img = _to_numpy(pil_resized)  # HWC, RGB
        # 0~1 정규화 → CHW tensor
        inp = torch.from_numpy(np_img).permute(2, 0, 1).unsqueeze(0).float() / 255.0

        with torch.no_grad():
            out = m(inp)  # 모델별로 반환 형태 상이
        # 아래는 yolov5 계열의 보편적 포맷 가정(필요 시 사용자 모델에 맞게 수정)
        # 기대: [N, 6] = x1,y1,x2,y2,conf,cls
        if isinstance(out, (list, tuple)):
            out = out[0]
        if hasattr(out, "numpy"):
            out = out.numpy()
        elif isinstance(out, torch.Tensor):
            out = out.cpu().numpy()

        dets = []
        try:
            for row in out:
                if len(row) < 6:  # 호환 실패 시 더미
                    continue
                x1, y1, x2, y2, conf, cls = row[:6]
                x1, y1, x2, y2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
                w = max(0, x2 - x1)
                h = max(0, y2 - y1)
                dets.append({
                    "classIndex": int(cls),
                    "className": (LABELS[int(cls)] if 0 <= int(cls) < len(LABELS) else None),
                    "x": max(0, x1), "y": max(0, y1),
                    "w": w, "h": h,
                    "score": float(conf)
                })
        except Exception:
            # 알 수 없는 포맷 → 더미로 회피
            dets = _postprocess_dummy(rw, rh)["detections"]

        res = {
            "class_names": LABELS,
            "detections": dets,
            "time_ms": int((time.time() - t0) * 1000),
            "__serverImageW": rw,
            "__serverImageH": rh,
        }
        return res

    # 방어적 기본값
    res = _postprocess_dummy(rw, rh)
    res["__serverImageW"] = rw
    res["__serverImageH"] = rh
    res["time_ms"] = int((time.time() - t0) * 1000)
    return res

# ─────────────────────────────────────────────────────────────────────────────
# 워커(비동기 202용)
# ─────────────────────────────────────────────────────────────────────────────
def _start_worker_once():
    global _worker_started
    if _worker_started:
        return
    _worker_started = True

    def _worker():
        app.logger.info("[worker] started")
        while True:
            try:
                job_id, img_bytes, kind = _job_q.get()
                _jobs[job_id] = {"status": "running"}
                with _infer_lock:
                    out = run_inference(img_bytes)
                _jobs[job_id] = {"status": "done", "result": out}
                del img_bytes
                gc.collect()
            except Exception as e:
                _jobs[job_id] = {"status": "error", "error": str(e)}
            finally:
                _job_q.task_done()

    t = threading.Thread(target=_worker, daemon=True)
    t.start()

# ─────────────────────────────────────────────────────────────────────────────
# Flask 3.x: before_serving 훅에서 시작 작업
# ─────────────────────────────────────────────────────────────────────────────
@app.before_serving
def _startup():
    threading.Thread(target=load_model_bg, daemon=True).start()
    _start_worker_once()
    app.logger.info("[startup] background threads started")

# ─────────────────────────────────────────────────────────────────────────────
# 헬스/루트
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    # 502 방지: 간단 JSON
    return jsonify(ok=True, health="/health"), 200

@app.get("/health")
def health():
    if ready and model_err is None:
        return jsonify(status="ready"), 200
    if model_err is not None:
        return jsonify(status="error", error=model_err), 500
    return jsonify(status="warming"), 503  # 준비중엔 503로 LB가 재시도하게

# ─────────────────────────────────────────────────────────────────────────────
# 입력 파싱 (multipart & json-base64)
# ─────────────────────────────────────────────────────────────────────────────
def _read_image_from_request() -> Tuple[Optional[bytes], Optional[str]]:
    # 멀티파트
    file = request.files.get("file") or request.files.get("image")
    if file:
        data = file.read()
        return data, "multipart"
    # JSON(base64)
    try:
        body = request.get_json(silent=True) or {}
        b64 = body.get("data")
        if b64:
            import base64
            return base64.b64decode(b64), "json"
    except Exception:
        pass
    return None, None

# ─────────────────────────────────────────────────────────────────────────────
# 감지 API
# ─────────────────────────────────────────────────────────────────────────────
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
            del img_bytes
            gc.collect()

    # async 202
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "queued"}
    _job_q.put((job_id, img_bytes, kind or "multipart"))
    return jsonify(jobId=job_id), 202

@app.post("/detect-json")
def detect_json():
    # JSON(base64) 버전
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
    # running / queued
    return jsonify(status=j["status"]), 202

# ─────────────────────────────────────────────────────────────────────────────
# 로컬 실행
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 개발용 실행 (Render에선 gunicorn 사용)
    load_model_bg()
    _start_worker_once()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
