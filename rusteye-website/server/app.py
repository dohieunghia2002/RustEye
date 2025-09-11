from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import uuid
from PIL import Image
import io
import numpy as np
import cv2

from model.inference import load_model, predict_mask  # YOLO cho vùng hao mòn

# Tích hợp IS-Net
import torch
from torch.autograd import Variable
from torchvision import transforms
import torch.nn.functional as F
import sys

# Thêm đường dẫn đến thư mục isnet trong backend
sys.path.append('./isnet')

from data_loader_cache import normalize, im_reader, im_preprocess
from isnet import ISNetDIS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:4000"}})

# Cấu hình thư mục uploads là static folder
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model YOLO
model_yolo = load_model("./model/best.pt")

# Cấu hình IS-Net
hypar = {
    "model": ISNetDIS(),
    "model_path": "./saved_models",
    "restore_model": "isnet-general-use.pth",
    "model_digit": "full",
    "cache_size": [1024, 1024]
}

class GOSNormalize(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std
    def __call__(self, image):
        image = normalize(image, self.mean, self.std)
        return image

transform = transforms.Compose([GOSNormalize([0.5, 0.5, 0.5], [1.0, 1.0, 1.0])])

def load_image(im_path, hypar):
    im = im_reader(im_path)
    im, im_shp = im_preprocess(im, hypar["cache_size"])
    im = torch.divide(im, 255.0)
    shape = torch.from_numpy(np.array(im_shp))
    return transform(im).unsqueeze(0), shape.unsqueeze(0)

def build_model(hypar, device):
    net = hypar["model"]
    if hypar["model_digit"] == "half":
        net.half()
        for layer in net.modules():
            if isinstance(layer, torch.nn.BatchNorm2d):
                layer.float()
    net.to(device)
    if hypar["restore_model"] != "":
        model_full_path = os.path.join(hypar["model_path"], hypar["restore_model"])
        if not os.path.exists(model_full_path):
            raise FileNotFoundError(f"Model file {model_full_path} not found. Please download manually and place it in {hypar['model_path']}.")
        net.load_state_dict(torch.load(model_full_path, map_location=device))
    net.eval()
    return net

def predict_isnet(net, inputs_val, shapes_val, hypar, device):
    net.eval()
    if hypar["model_digit"] == "full":
        inputs_val = inputs_val.type(torch.FloatTensor)
    else:
        inputs_val = inputs_val.type(torch.HalfTensor)
    inputs_val_v = Variable(inputs_val, requires_grad=False).to(device)
    ds_val = net(inputs_val_v)[0]
    pred_val = ds_val[0][0, :, :, :]
    pred_val = torch.squeeze(F.interpolate(torch.unsqueeze(pred_val, 0), (shapes_val[0][0], shapes_val[0][1]), mode='bilinear'))  # Sửa upsample thành interpolate
    ma = torch.max(pred_val)
    mi = torch.min(pred_val)
    pred_val = (pred_val - mi) / (ma - mi)
    if device == 'cuda':
        torch.cuda.empty_cache()
    return (pred_val.detach().cpu().numpy() * 255).astype(np.uint8)

# Khởi tạo IS-Net
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net_isnet = build_model(hypar, device)

# Route để phục vụ file từ uploads
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename), mimetype='image/png')

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    file.seek(0)
    img = Image.open(file.stream).convert("RGB")
    img_np = np.array(img)

    file.seek(0)
    img_bytes = io.BytesIO(file.read())

    # Chạy YOLO segmentation (vùng hao mòn)
    mask_rust = predict_mask(model_yolo, img)

    # Chạy IS-Net để segment toàn bộ cột viễn thông
    image_tensor, orig_size = load_image(img_bytes, hypar)
    mask_pole = predict_isnet(net_isnet, image_tensor, orig_size, hypar, device)

    # Đảm bảo kích thước mask
    if mask_pole.shape[:2] != img_np.shape[:2]:
        mask_pole = cv2.resize(mask_pole, (img_np.shape[1], img_np.shape[0]), interpolation=cv2.INTER_NEAREST)
    if mask_rust.shape[:2] != img_np.shape[:2]:
        mask_rust = cv2.resize(mask_rust, (img_np.shape[1], img_np.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Giao vùng hao mòn với mask cột
    mask_rust_on_pole = cv2.bitwise_and(mask_rust, mask_pole)

    # Tính diện tích
    pole_area = np.sum(mask_pole > 0)
    rust_area = np.sum(mask_rust_on_pole > 0)

    # Tính phần trăm hao mòn
    rust_percentage = (rust_area / pole_area * 100) if pole_area > 0 else 0.0

    # Lưu mask
    mask_rust_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex}_mask_rust.png")
    Image.fromarray(mask_rust).save(mask_rust_path)
    mask_pole_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex}_mask_pole.png")
    Image.fromarray(mask_pole).save(mask_pole_path)

    # Trả về URL tương đối (Flask sẽ xử lý)
    return jsonify({
        "mask_rust_url": f"/uploads/{os.path.basename(mask_rust_path)}",
        "mask_pole_url": f"/uploads/{os.path.basename(mask_pole_path)}",
        "rust_percentage": rust_percentage
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)