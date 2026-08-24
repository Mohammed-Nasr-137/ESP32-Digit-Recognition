import os
import glob
import cv2
import numpy as np
import torch
import tensorflow as tf
import onnx2tf
from model import ocr_model
import config

def prep_samples():
    sample = []
    for i in range(10):
        sample.append(glob.glob(f"{config.DATASET_PATH}\\{i}\\*.png")[:10])

    return [item for sublist in sample for item in sublist]


def representative_dataset_gen():
    sample_images = prep_samples()
    for img_path in sample_images:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
    
        img = cv2.resize(img, (16, 16), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=(0, -1))
        yield [img]


tf_path = config.TF_PATH
converter = tf.lite.TFLiteConverter.from_saved_model(tf_path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_quant_model = converter.convert()
tflite_path = config.TFLITE_PATH
with open(tflite_path, "wb") as f:
    f.write(tflite_quant_model)

print(f"Quantized TFLite Model Size: {len(tflite_quant_model)} bytes")
