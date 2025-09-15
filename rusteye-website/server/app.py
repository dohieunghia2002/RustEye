from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import uuid
from PIL import Image
import io
import numpy as np
import cv2
import sys
import torch
from torch.autograd import Variable
from torchvision import transforms
import torch.nn.functional as F
import hydra
from hydra.core.global_hydra import GlobalHydra

# YOLO cho vùng hao mòn
from model.inference import load_model, predict_mask  

# IS-Net
sys.path.append('./isnet')
from data_loader_cache import normalize, im_reader, im_preprocess
from isnet import ISNetDIS

# SAM2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Fuzzy + OR-Tools
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from ortools.sat.python import cp_model

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:4000"}})

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ---------------- Load models ---------------- #
# YOLO
model_yolo = load_model("./model/best.pt")

# ISNet config
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

def build_isnet(hypar, device):
    net = hypar["model"]
    if hypar["model_digit"] == "half":
        net.half()
        for layer in net.modules():
            if isinstance(layer, torch.nn.BatchNorm2d):
                layer.float()
    net.to(device)
    model_full_path = os.path.join(hypar["model_path"], hypar["restore_model"])
    if not os.path.exists(model_full_path):
        raise FileNotFoundError(f"Model file {model_full_path} not found.")
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
    pred_val = torch.squeeze(F.interpolate(torch.unsqueeze(pred_val, 0), (shapes_val[0][0], shapes_val[0][1]), mode='bilinear'))
    ma, mi = torch.max(pred_val), torch.min(pred_val)
    pred_val = (pred_val - mi) / (ma - mi)
    if device == 'cuda':
        torch.cuda.empty_cache()
    return (pred_val.detach().cpu().numpy() * 255).astype(np.uint8)


# SAM2 load
checkpoint = "checkpoints/sam2.1_hiera_tiny.pt"
model_cfg = "D:/RustEye/rusteye-website/server/configs/sam2.1/sam2.1_hiera_t.yaml"
device = "cpu"  # Ép buộc chạy trên CPU
predictor_sam2 = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint, device=device))

def predict_sam2(image: np.ndarray, box=None):
    predictor_sam2.set_image(image)
    if box is None:
        h, w = image.shape[:2]
        box = np.array([0, 0, w, h])
    with torch.inference_mode(), torch.autocast("cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16):
        masks, scores, logits = predictor_sam2.predict(
            box=box,
            multimask_output=False
        )
    best_mask = masks[np.argmax(scores)]
    return (best_mask.astype(np.uint8) * 255)

# Init ISNet
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net_isnet = build_isnet(hypar, device)

# ---------------- Decision logic ---------------- #
def fuzzy_and_cp_decision(damage_percentage: float):
    damage_size = ctrl.Antecedent(np.arange(0, 101, 1), 'damage_size')
    damage_level_fuzzy = ctrl.Consequent(np.arange(0, 101, 1), 'damage_level')

    damage_size['Ri0'] = fuzz.trimf(damage_size.universe, [0, 0, 5])
    damage_size['Ri1'] = fuzz.trimf(damage_size.universe, [0, 5, 15])
    damage_size['Ri2'] = fuzz.trimf(damage_size.universe, [10, 20, 30])
    damage_size['Ri3'] = fuzz.trimf(damage_size.universe, [25, 40, 55])
    damage_size['Ri4'] = fuzz.trimf(damage_size.universe, [50, 70, 85])
    damage_size['Ri5'] = fuzz.trimf(damage_size.universe, [80, 100, 100])

    damage_level_fuzzy.automf(names=['Excellent', 'Good', 'Fair', 'Poor', 'Severe', 'Critical'])

    rules = [
        ctrl.Rule(damage_size['Ri0'], damage_level_fuzzy['Excellent']),
        ctrl.Rule(damage_size['Ri1'], damage_level_fuzzy['Good']),
        ctrl.Rule(damage_size['Ri2'], damage_level_fuzzy['Fair']),
        ctrl.Rule(damage_size['Ri3'], damage_level_fuzzy['Poor']),
        ctrl.Rule(damage_size['Ri4'], damage_level_fuzzy['Severe']),
        ctrl.Rule(damage_size['Ri5'], damage_level_fuzzy['Critical']),
    ]

    damage_ctrl = ctrl.ControlSystem(rules)
    sim = ctrl.ControlSystemSimulation(damage_ctrl)
    sim.input['damage_size'] = damage_percentage
    sim.compute()
    fuzzy_level = sim.output['damage_level']

    model_cp = cp_model.CpModel()
    damage_var = model_cp.NewIntVar(0, 5, 'damage_level')
    if damage_percentage < 5:
        model_cp.Add(damage_var == 0)
    elif damage_percentage < 15:
        model_cp.Add(damage_var == 1)
    elif damage_percentage < 30:
        model_cp.Add(damage_var == 2)
    elif damage_percentage < 50:
        model_cp.Add(damage_var == 3)
    elif damage_percentage < 70:
        model_cp.Add(damage_var == 4)
    else:
        model_cp.Add(damage_var == 5)

    solver = cp_model.CpSolver()
    solver.Solve(model_cp)
    severity = solver.Value(damage_var)

    actions = [
        "Không cần hành động",
        "Giám sát và theo dõi",
        "Thực hiện bảo dưỡng nhỏ",
        "Bảo dưỡng khẩn cấp",
        "Sửa chữa hoặc thay thế",
        "Thay thế toàn bộ",
    ]
    action = actions[severity]

    return fuzzy_level, severity, action

# ---------------- Routes ---------------- #
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

    # YOLO rust mask
    mask_rust = predict_mask(model_yolo, img)

    # ISNet pole mask
    image_tensor, orig_size = load_image(img_bytes, hypar)
    mask_pole = predict_isnet(net_isnet, image_tensor, orig_size, hypar, device)

    # SAM2 pole mask
    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    mask_sam2 = predict_sam2(img_rgb)

    # Resize
    h, w = img_np.shape[:2]
    mask_rust = cv2.resize(mask_rust, (w, h), interpolation=cv2.INTER_NEAREST)
    mask_pole = cv2.resize(mask_pole, (w, h), interpolation=cv2.INTER_NEAREST)
    mask_sam2 = cv2.resize(mask_sam2, (w, h), interpolation=cv2.INTER_NEAREST)

    # Rust percentage with ISNet
    mask_rust_on_pole = cv2.bitwise_and(mask_rust, mask_pole)
    pole_area = np.sum(mask_pole > 0)
    rust_area = np.sum(mask_rust_on_pole > 0)
    rust_percentage_isnet = (rust_area / pole_area * 100) if pole_area > 0 else 0.0

    # Rust percentage with SAM2
    mask_rust_on_sam2 = cv2.bitwise_and(mask_rust, mask_sam2)
    sam2_area = np.sum(mask_sam2 > 0)
    rust_area_sam2 = np.sum(mask_rust_on_sam2 > 0)
    rust_percentage_sam2 = (rust_area_sam2 / sam2_area * 100) if sam2_area > 0 else 0.0

    # Save masks
    mask_rust_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex}_mask_rust.png")
    Image.fromarray(mask_rust).save(mask_rust_path)
    mask_pole_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex}_mask_pole.png")
    Image.fromarray(mask_pole).save(mask_pole_path)
    mask_sam2_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex}_mask_sam2.png")
    Image.fromarray(mask_sam2).save(mask_sam2_path)

    # Decision
    fuzzy_isnet, severity_isnet, action_isnet = fuzzy_and_cp_decision(rust_percentage_isnet)
    fuzzy_sam2, severity_sam2, action_sam2 = fuzzy_and_cp_decision(rust_percentage_sam2)

    return jsonify({
        "mask_rust_url": f"/uploads/{os.path.basename(mask_rust_path)}",
        "mask_pole_url": f"/uploads/{os.path.basename(mask_pole_path)}",
        "mask_sam2_url": f"/uploads/{os.path.basename(mask_sam2_path)}",
        "isnet": {
            "rust_percentage": rust_percentage_isnet,
            "fuzzy_level": fuzzy_isnet,
            "severity": severity_isnet,
            "action": action_isnet
        },
        "sam2": {
            "rust_percentage": rust_percentage_sam2,
            "fuzzy_level": fuzzy_sam2,
            "severity": severity_sam2,
            "action": action_sam2
        }
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
