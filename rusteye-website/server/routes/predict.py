from flask import Blueprint, request, jsonify, current_app, send_file
import os, uuid, io
import numpy as np
import cv2
from PIL import Image

from services.yolo_service import predict_rust_mask
from services.isnet_service import load_image, predict_isnet, hypar
from services.sam2_service import predict_sam2
from services.decision_service import fuzzy_and_cp_decision
from utils.image_utils import resize_mask, save_mask, calculate_rust_percentage

predict_bp = Blueprint("predict", __name__)

@predict_bp.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_file(os.path.join(current_app.config['UPLOAD_FOLDER'], filename), mimetype='image/png')

@predict_bp.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    img = Image.open(file.stream).convert("RGB")
    img_np = np.array(img)
    h, w = img_np.shape[:2]

    # YOLO mask
    mask_rust = predict_rust_mask(img)

    # ISNet mask
    file.seek(0)
    img_bytes = io.BytesIO(file.read())
    image_tensor, orig_size = load_image(img_bytes, hypar)
    mask_pole = predict_isnet(image_tensor, orig_size)

    # SAM2 mask
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    mask_sam2 = predict_sam2(img_bgr)

    # Resize
    mask_rust = resize_mask(mask_rust, (h, w))
    mask_pole = resize_mask(mask_pole, (h, w))
    mask_sam2 = resize_mask(mask_sam2, (h, w))

    # Rust percentages
    rust_percentage_isnet = calculate_rust_percentage(mask_rust, mask_pole)
    rust_percentage_sam2 = calculate_rust_percentage(mask_rust, mask_sam2)

    # Save masks
    upload_folder = current_app.config["UPLOAD_FOLDER"]
    mask_rust_path = save_mask(mask_rust, upload_folder, f"{uuid.uuid4().hex}_mask_rust")
    mask_pole_path = save_mask(mask_pole, upload_folder, f"{uuid.uuid4().hex}_mask_pole")
    mask_sam2_path = save_mask(mask_sam2, upload_folder, f"{uuid.uuid4().hex}_mask_sam2")

    # Decision
    fuzzy_isnet, sev_isnet, action_isnet = fuzzy_and_cp_decision(rust_percentage_isnet)
    fuzzy_sam2, sev_sam2, action_sam2 = fuzzy_and_cp_decision(rust_percentage_sam2)

    return jsonify({
        "mask_rust_url": f"/uploads/{os.path.basename(mask_rust_path)}",
        "mask_pole_url": f"/uploads/{os.path.basename(mask_pole_path)}",
        "mask_sam2_url": f"/uploads/{os.path.basename(mask_sam2_path)}",
        "isnet": {"rust_percentage": rust_percentage_isnet, "fuzzy_level": fuzzy_isnet, "severity": sev_isnet, "action": action_isnet},
        "sam2": {"rust_percentage": rust_percentage_sam2, "fuzzy_level": fuzzy_sam2, "severity": sev_sam2, "action": action_sam2}
    })
