import cv2
import numpy as np


def img_pipline(read_path, write_path):
    img = cv2.imread(read_path, cv2.IMREAD_GRAYSCALE)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3,3), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    img = cv2.dilate(img, kernel, iterations=1)
    coords = cv2.findNonZero(img)
    if coords is None:
        print(f"No ink detected in frame.")
        return None

    x, y, w, h = cv2.boundingRect(coords)
    img = img[y:y+h, x:x+w]
    scale = min(12.0 / w, 12.0 / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    img = cv2.resize(img, (new_w,new_h), interpolation=cv2.INTER_AREA)
    canvas16 = np.zeros((16, 16), dtype=np.uint8)
    start_x = (16 - new_w) // 2
    start_y = (16 - new_h) // 2
    canvas16[start_y:start_y+new_h, start_x:start_x+new_w] = img
    cv2.imwrite(write_path, canvas16)
