import os
import numpy as np
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Đường dẫn tuyệt đối tới config & checkpoint
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # = server/
model_cfg = os.path.join(BASE_DIR, "configs", "sam2.1", "sam2.1_hiera_t.yaml")
checkpoint = os.path.join(BASE_DIR, "checkpoints", "sam2.1_hiera_tiny.pt")

device = "cpu"  # hoặc "cuda" nếu có GPU

predictor_sam2 = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint, device=device))

def predict_sam2(image: np.ndarray, box=None):
    predictor_sam2.set_image(image)
    if box is None:
        h, w = image.shape[:2]
        box = np.array([0, 0, w, h])
    with torch.inference_mode():
        masks, scores, _ = predictor_sam2.predict(box=box, multimask_output=False)
    best_mask = masks[np.argmax(scores)]
    return (best_mask.astype(np.uint8) * 255)
