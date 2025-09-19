import os
import cv2
import numpy as np
from PIL import Image

def resize_mask(mask: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Resize mask về cùng kích thước (h, w)."""
    h, w = target_shape
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

def save_mask(mask: np.ndarray, folder: str, prefix: str) -> str:
    """Lưu mask ra file PNG và trả về path."""
    os.makedirs(folder, exist_ok=True)
    filename = f"{prefix}.png"
    path = os.path.join(folder, filename)
    Image.fromarray(mask).save(path)
    return path

def calculate_rust_percentage(mask_rust: np.ndarray, mask_pole: np.ndarray) -> float:
    """Tính phần trăm diện tích rỉ sét trên cột."""
    mask_rust_on_pole = cv2.bitwise_and(mask_rust, mask_pole)
    pole_area = np.sum(mask_pole > 0)
    rust_area = np.sum(mask_rust_on_pole > 0)
    return (rust_area / pole_area * 100) if pole_area > 0 else 0.0

def overlay_masks(frame: np.ndarray, mask_pole: np.ndarray, mask_rust: np.ndarray) -> np.ndarray:
    """Tạo overlay mask lên frame (blue = pole, red = rust)."""
    overlay = frame.copy()

    # Blue = cột
    pole_color = [255, 0, 0]  # BGR
    mask_pole_colored = np.zeros_like(frame, dtype=np.uint8)
    mask_pole_colored[mask_pole > 0] = pole_color
    overlay = cv2.addWeighted(overlay, 1, mask_pole_colored, 0.5, 0)

    # Red = rỉ sét
    rust_color = [0, 0, 255]
    mask_rust_colored = np.zeros_like(frame, dtype=np.uint8)
    mask_rust_colored[mask_rust > 0] = rust_color
    overlay = cv2.addWeighted(overlay, 1, mask_rust_colored, 0.5, 0)

    return overlay

def cleanup_folder(folder: str):
    """Xóa toàn bộ file trong folder."""
    if not os.path.exists(folder):
        return
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path):
            os.unlink(file_path)
