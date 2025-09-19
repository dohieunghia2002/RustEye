import os
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from torchvision import transforms

from isnet.isnet import ISNetDIS
from isnet.data_loader_cache import normalize, im_reader, im_preprocess

# Cấu hình
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
        return normalize(image, self.mean, self.std)

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
    net.load_state_dict(torch.load(model_full_path, map_location=device))
    net.eval()
    return net

# Khởi tạo 1 lần
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net_isnet = build_isnet(hypar, device)

def predict_isnet(inputs_val, shapes_val):
    if hypar["model_digit"] == "full":
        inputs_val = inputs_val.type(torch.FloatTensor)
    else:
        inputs_val = inputs_val.type(torch.HalfTensor)
    inputs_val_v = Variable(inputs_val, requires_grad=False).to(device)
    ds_val = net_isnet(inputs_val_v)[0]
    pred_val = ds_val[0][0, :, :, :]
    pred_val = torch.squeeze(F.interpolate(torch.unsqueeze(pred_val, 0),
                                           (shapes_val[0][0], shapes_val[0][1]),
                                           mode='bilinear'))
    ma, mi = torch.max(pred_val), torch.min(pred_val)
    pred_val = (pred_val - mi) / (ma - mi)
    return (pred_val.detach().cpu().numpy() * 255).astype(np.uint8)
