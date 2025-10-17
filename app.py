# app.py
import os
from io import BytesIO
from flask import Flask, request, jsonify
from PIL import Image

app = Flask(__name__)

@app.get("/health")
def health():
    return jsonify(status="ok"), 200

@app.post("/detect")
def detect():
    """
    클라이언트는 multipart form으로 'file' 필드에 이미지를 보냅니다.
    (필드명: 'file' 로 고정, React Native 쪽도 동일하게 보냄)
    """
    f = request.files.get("file") or request.files.get("image")
    if not f:
        return jsonify(error='no file field "file" or "image"'), 400

    # 업로드 파일 확인(옵션)
    try:
        img = Image.open(BytesIO(f.read()))
        img.verify()  # 이미지 유효성 체크
    except Exception:
        return jsonify(error="invalid image"), 400

    # TODO: TorchScript(.ptl) 추론 로직 연결
    # 지금은 형식을 보여주기 위해 더미 응답
    return jsonify(
        class_names=[
            "coca cola","coke zero","pepsi zero",
            "chilsung cider","chilsung cider zero",
            "fanta","hot6","hot6 force"
        ],
        detections=[]  # 예: [{"x":100,"y":120,"w":180,"h":200,"score":0.87,"classIndex":0}]
    ), 200

if __name__ == "__main__":
    print("URL MAP:", app.url_map)
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
