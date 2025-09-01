import numpy as np
import cv2
from ultralytics import YOLO

def load_model(weight_path: str):
    return YOLO(weight_path)

def predict_mask(model, img):
    # chạy model
    results = model(img)
    res = results[0]  # YOLO trả về list, lấy ảnh đầu tiên

    img_np = np.array(img)
    h, w = img_np.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    if res.masks is None:
        return mask  # không có object nào được detect

    # duyệt qua từng mask
    for seg in res.masks.data.cpu().numpy():
        seg = seg.astype(np.uint8) * 255  # từ [0,1] → 0/255

        # resize mask về đúng size ảnh gốc
        seg_resized = cv2.resize(
            seg,
            (w, h),
            interpolation=cv2.INTER_NEAREST
        )

        mask = np.maximum(mask, seg_resized)

    return mask
