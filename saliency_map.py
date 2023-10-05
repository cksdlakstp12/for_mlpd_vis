import numpy as np
import cv2
from PIL import Image
import os

name = 'I01419'

def create_saliency_map(image, is_rgb=True):
    if is_rgb:
        # If image is RGB, convert it to BGR
        image_np = np.array(image)[:, :, ::-1]  # RGB to BGR
    else:
        # If image is not RGB (like LWIR), just convert to numpy array
        image_np = np.array(image)

    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    _, saliency_map = saliency.computeSaliency(image_np)

    if is_rgb:
        # Convert grayscale to BGR for RGB images
        saliency_map = cv2.cvtColor(saliency_map, cv2.COLOR_GRAY2BGR)

    saliency_map_pil = Image.fromarray(np.uint8(saliency_map*255))
    return saliency_map_pil

def apply_saliency_map(image, saliency):
# Convert PIL Image to numpy array
    image_np = np.array(image)
    saliency_np = np.array(saliency)

    # Perform the multiplication
    augmented_np = image_np * saliency_np

    # Convert the numpy array back to PIL Image
    augmented = Image.fromarray(augmented_np)

    return augmented

input_size = [512., 640.]

path = f"results/{name}"
os.makedirs(path, exist_ok=True)

vis = Image.open(f"./data/{name}_rgb.jpg")
lwir = Image.open(f"./data/{name}_lwir.jpg").convert('L')

saliency_map_vis = create_saliency_map(vis, is_rgb=True)
saliency_map_lwir = create_saliency_map(lwir, is_rgb=False)
# print(np.array(saliency_map_vis))
# print("="*50)
# print(np.array(saliency_map_lwir))
saliency_map_vis.save(f"{path}/{name}_saliency_map_vis.jpg")
saliency_map_lwir.save(f"{path}/{name}_saliency_map_lwir.jpg")
# print("="*50)
at_vis = apply_saliency_map(vis, saliency_map_vis)
at_lwir = apply_saliency_map(lwir, saliency_map_lwir)
# print(np.array(at_vis))
# print("="*50)
# print(np.array(at_lwir))
at_vis.save(f"{path}/{name}_at_vis.jpg")
at_lwir.save(f"{path}/{name}_at_lwir.jpg")