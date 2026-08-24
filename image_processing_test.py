import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from model import ocr_model
import config

sample = "sample1"
digit = 0
correct = 0

model = ocr_model()
state_dict = torch.load(config.FINETUNED_PTH_MODEL, weights_only=True)
model.load_state_dict(state_dict)
model.eval()

def show_img(img, label=digit):
    plt.imshow(img, cmap='gray')
    plt.title(f"Label: {label}")
    plt.axis('off')
    plt.show()


for i in range(10):
    path = f"{config.PIPELINE_TEST_PATH}//{sample}//{digit}.png"
    
    test_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # show_img(test_img, digit)

    # binary_img = cv2.adaptiveThreshold(test_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, blockSize=3, C=2)
    # show_img(binary_img, digit)

    _, binary_img = cv2.threshold(test_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # show_img(binary_img)

    # edges = cv2.Canny(test_img, 100, 200)
    # show_img(edges)

    kernel = np.ones((3,3), np.uint8)
    closing = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
    # show_img(closing)

    dilated = cv2.dilate(closing, kernel, iterations=1)
    # show_img(dilated)

    coords = cv2.findNonZero(dilated)
    if coords is None:
        print(f"Digit {digit}: No ink detected in frame.")
        digit += 1
        continue

    x, y, w, h = cv2.boundingRect(coords)
    cropped_digit = dilated[y:y+h, x:x+w]

    scale = min(12.0 / w, 12.0 / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(cropped_digit, (new_w,new_h), interpolation=cv2.INTER_AREA)
    canvas16 = np.zeros((16, 16), dtype=np.uint8)
    start_x = (16 - new_w) // 2
    start_y = (16 - new_h) // 2
    canvas16[start_y:start_y+new_h, start_x:start_x+new_w] = resized
    # show_img(canvas16)

    # canvas16 = cv2.GaussianBlur(canvas16, (3,3), sigmaX=0.8)
    # show_img(resized)
    # show_img(canvas16)
    canvas16 = torch.tensor(canvas16.reshape(1, 1, 16, 16), dtype=torch.float32) / 255.0

    outputs = model(canvas16)
    # print(outputs, outputs.shape)
    pred = int(torch.argmax(outputs))
    print(pred, i)
    if pred == i:
        correct += 1

    digit += 1

print(correct, f"accuracy: {correct/10}")
