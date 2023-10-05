import cv2
import os
import numpy as np
import csv

# 이미지 로드
name = 'I01782'
path = f"results/{name}"
os.makedirs(path, exist_ok=True)
vis = cv2.imread(f'data/{name}_saliency_map_vis.jpg')
lwir = cv2.imread(f'data/{name}_saliency_map_lwir.jpg')
bg_vis = cv2.imread(f'{path}/bg_vis.jpg')
bg_lwir = cv2.imread(f'{path}/bg_lwir.jpg')

height, width = [512, 640]

data = [
    ["A_mean"],
    ["A_std"],
    ["B_mean"],
    ["B_std"],
    ["C_mean"],
    ["C_std"],
    ["D_mean"],
    ["D_std"],
]

with open(f"data/{name}_rgb.txt", 'r') as f:
    for i, line in enumerate(f.readlines()):
        if i == 0: continue
        bndbox = list(map(int, line.strip().split(" ")[1:5]))
        bndbox[2] = min( bndbox[2] + bndbox[0], width )
        bndbox[3] = min( bndbox[3] + bndbox[1], height )
        x1, y1, x2, y2 = bndbox

        w = x2 - x1
        h = y2 - y1

        obj = vis[y1:y2, x1:x2, :]
        mean_A = np.mean(obj)
        std_A = np.std(obj)
        obj = vis[y1:y2, min(width, int(x1+w/2)):min(width, int(x2+w/2)), :]
        mean_B = np.mean(obj)
        std_B = np.std(obj)
        obj = vis[max(0, int(y1-20)):min(height, int(y2+20)), max(0, int(x1-20)):min(width, int(x2+20)), :]
        mean_C = np.mean(obj)
        std_C = np.std(obj)
        mean_D = np.mean(bg_vis)
        std_D = np.std(bg_vis)

        data[0].append(mean_A)
        data[1].append(std_A)
        data[2].append(mean_B)
        data[3].append(std_B)
        data[4].append(mean_C)
        data[5].append(std_C)
        data[6].append(mean_D)
        data[7].append(std_D)

        break

with open(f"data/{name}_lwir.txt", 'r') as f:
    for i, line in enumerate(f.readlines()):
        if i == 0: continue
        bndbox = list(map(int, line.strip().split(" ")[1:5]))
        bndbox[2] = min( bndbox[2] + bndbox[0], width )
        bndbox[3] = min( bndbox[3] + bndbox[1], height )
        x1, y1, x2, y2 = bndbox
        
        obj = lwir[y1:y2, x1:x2, :]
        mean_A = np.mean(obj)
        std_A = np.std(obj)
        obj = lwir[y1:y2, min(width, int(x1+w/2)):min(width, int(x2+w/2)), :]
        mean_B = np.mean(obj)
        std_B = np.std(obj)
        obj = lwir[max(0, int(y1-20)):min(height, int(y2+20)), max(0, int(x1-20)):min(width, int(x2+20)), :]
        mean_C = np.mean(obj)
        std_C = np.std(obj)
        obj = vis[y1:y2, min(width, int(x1+w/2)):min(width, int(x2+w/2)), :]
        mean_D = np.mean(bg_lwir)
        std_D = np.std(bg_lwir)

        data[0].append(mean_A)
        data[1].append(std_A)
        data[2].append(mean_B)
        data[3].append(std_B)
        data[4].append(mean_C)
        data[5].append(std_C)
        data[6].append(mean_D)
        data[7].append(std_D)

        break


with open(f"{path}/{name}_values.csv", "w", newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)