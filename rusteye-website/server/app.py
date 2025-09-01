from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import uuid
from PIL import Image
import io
import numpy as np
import cv2

from model.inference import load_model, predict_mask

app = Flask(__name__)
CORS(app)

# load model YOLO
model = load_model("./model/best.pt")  # thay "best.pt" bằng đường dẫn tới model YOLO của bạn

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    img = Image.open(file.stream).convert("RGB")
    img_np = np.array(img)

    # chạy YOLO segmentation
    mask = predict_mask(model, img)

    # nếu mask không cùng kích thước với ảnh gốc → resize về (width, height)
    if mask.shape[:2] != img_np.shape[:2]:
        mask = cv2.resize(
            mask,
            (img_np.shape[1], img_np.shape[0]),  # (width, height)
            interpolation=cv2.INTER_NEAREST
        )

    # lưu mask
    mask_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex}_mask.png")
    Image.fromarray(mask).save(mask_path)

    return send_file(mask_path, mimetype="image/png")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
