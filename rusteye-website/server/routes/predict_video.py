from flask import Blueprint, request, jsonify, current_app
import os, uuid, io
import numpy as np
import cv2
from PIL import Image

from services.yolo_service import predict_rust_mask
from services.isnet_service import load_image, predict_isnet, hypar
from services.sam2_service import predict_sam2
from services.decision_service import fuzzy_and_cp_decision
from utils.image_utils import resize_mask, calculate_rust_percentage, overlay_masks, cleanup_folder

predict_video_bp = Blueprint("predict_video", __name__)

@predict_video_bp.route("/predict_video", methods=["POST"])
def predict_video():
    try:
        upload_folder = current_app.config["UPLOAD_FOLDER"]
        cleanup_folder(upload_folder)

        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        if "model_choice" not in request.form:
            return jsonify({"error": "No model_choice provided (isnet or sam2)"}), 400

        file = request.files["file"]
        model_choice = request.form["model_choice"].lower()
        if model_choice not in ['isnet', 'sam2']:
            return jsonify({"error": "Invalid model_choice"}), 400

        # Save input video
        video_uuid = uuid.uuid4().hex
        input_video_path = os.path.join(upload_folder, f"{video_uuid}_input.mp4")
        output_video_path = os.path.join(upload_folder, f"{video_uuid}_output.mp4")
        file.save(input_video_path)

        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            os.remove(input_video_path)
            return jsonify({"error": "Cannot open video file"}), 400

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'H264')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        if not out.isOpened():
            cap.release()
            os.remove(input_video_path)
            return jsonify({"error": "Cannot create output video"}), 500

        frame_count = 0
        rust_percentages = []
        frame_skip = 5

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            if frame_count % frame_skip != 0:
                out.write(frame)
                continue

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_bytes = io.BytesIO()
            img_pil.save(img_bytes, format='PNG')
            img_bytes.seek(0)

            # YOLO rust mask
            mask_rust = predict_rust_mask(img_pil)
            mask_rust = resize_mask(mask_rust, (height, width))

            # Pole mask
            if model_choice == 'isnet':
                image_tensor, orig_size = load_image(img_bytes, hypar)
                mask_pole = predict_isnet(image_tensor, orig_size)
            else:
                mask_pole = predict_sam2(img_rgb)

            mask_pole = resize_mask(mask_pole, (height, width))

            # Rust %
            rust_percentage = calculate_rust_percentage(mask_rust, mask_pole)
            rust_percentages.append(rust_percentage)

            # Overlay
            overlay = overlay_masks(frame, mask_pole, mask_rust)
            out.write(overlay)

        cap.release()
        out.release()
        os.remove(input_video_path)

        if not os.path.exists(output_video_path) or os.path.getsize(output_video_path) == 0:
            return jsonify({"error": "Output video not created or empty"}), 500

        avg_rust_percentage = np.mean(rust_percentages) if rust_percentages else 0.0
        fuzzy_level, severity, action = fuzzy_and_cp_decision(avg_rust_percentage)

        return jsonify({
            "output_video_url": f"/uploads/{os.path.basename(output_video_path)}",
            "avg_rust_percentage": avg_rust_percentage,
            "fuzzy_level": fuzzy_level,
            "severity": severity,
            "action": action,
            "model_used": model_choice.upper()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
