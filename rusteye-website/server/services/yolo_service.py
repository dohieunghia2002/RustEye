from model.inference import load_model, predict_mask
from PIL import Image

# Load YOLO once
model_yolo = load_model("./model/best.pt")

def predict_rust_mask(image: Image.Image):
    """Dự đoán mask rỉ sét bằng YOLO"""
    return predict_mask(model_yolo, image)
