import torch
import numpy as np
import cv2
from model import UNet
from PIL import Image
import io

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(model_path):
    model = UNet()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def analyze_return(image_path, model_path, threshold=0.5):
    model = load_model(model_path)
    
    with open(image_path, "rb") as f:
        pil_img = Image.open(io.BytesIO(f.read()))
        pil_img = pil_img.convert("RGB")
        original_img = np.array(pil_img)[:, :, ::-1]
        img_gray = np.array(pil_img.convert("L"))

    orig_h, original_w = img_gray.shape
    img_norm = img_gray.astype(np.float32) / 255.0
    img_resized = cv2.resize(img_norm, (624, 320))
    img_tensor = torch.tensor(np.expand_dims(img_resized, axis=(0, 1)), dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        pred = model(img_tensor)
        if not torch.is_floating_point(pred):
            raise ValueError("Model output is not a floating-point tensor")

    mask = (pred.squeeze().cpu().numpy() > float(threshold)).astype(np.uint8)
    mask = cv2.resize(mask, (original_w, orig_h))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = original_img.copy()
    cv2.drawContours(result, contours, -1, (0, 0, 255), 2)

    return result