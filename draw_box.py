import cv2
import os

# saliency_map.py 실행 후 할 것
name = 'I01419'
path = f"results/{name}"
os.makedirs(path, exist_ok=True)
vis = cv2.imread(f'{path}/{name}_saliency_map_vis.jpg')
lwir = cv2.imread(f'{path}/{name}_saliency_map_lwir.jpg')

height, width = [512., 640.]


with open(f"data/{name}_rgb.txt", 'r') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        if i == 0: continue
        bndbox = list(map(int, line.strip().split(" ")[1:5]))
        bndbox[2] = min( bndbox[2] + bndbox[0], width )
        bndbox[3] = min( bndbox[3] + bndbox[1], height )
        x1, y1, x2, y2 = bndbox
        cv2.imwrite(f"./{path}/{name}_box_vis{i}.jpg", vis[y1:y2, x1:x2, :])

    for i, line in enumerate(lines):
        if i == 0: continue
        bndbox = list(map(int, line.strip().split(" ")[1:5]))
        bndbox[2] = min( bndbox[2] + bndbox[0], width )
        bndbox[3] = min( bndbox[3] + bndbox[1], height )
        x1, y1, x2, y2 = bndbox
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2) 

with open(f"data/{name}_lwir.txt", 'r') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        if i == 0: continue
        bndbox = list(map(int, line.strip().split(" ")[1:5]))
        bndbox[2] = min( bndbox[2] + bndbox[0], width )
        bndbox[3] = min( bndbox[3] + bndbox[1], height )
        x1, y1, x2, y2 = bndbox
        cv2.imwrite(f"./{path}/{name}_box_lwir{i}.jpg", lwir[y1:y2, x1:x2, :])
        
    for i, line in enumerate(lines):
        if i == 0: continue
        bndbox = list(map(int, line.strip().split(" ")[1:5]))
        bndbox[2] = min( bndbox[2] + bndbox[0], width )
        bndbox[3] = min( bndbox[3] + bndbox[1], height )
        x1, y1, x2, y2 = bndbox
        cv2.rectangle(lwir, (x1, y1), (x2, y2), (0, 0, 255), 2) 

cv2.imwrite(f"./{path}/{name}_box_vis.jpg", vis)
cv2.imwrite(f"./{path}/{name}_box_lwir.jpg", lwir)