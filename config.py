from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

PTH_MODEL = PROJECT_ROOT / "model_8.pth"
FINETUNED_PTH_MODEL = PROJECT_ROOT / "finetuned_val_model_44.pth"
ONNX_MODEL = PROJECT_ROOT / "finetuned_val_model_44.onnx"
TF_PATH = PROJECT_ROOT / "finetuned_val_model_44_from_onnx"
TFLITE_PATH = PROJECT_ROOT / "finetuned_val_model_44_quantized.tflite"
DATASET_PATH  = PROJECT_ROOT / "dataset"
FINETUNE_DATASET_PATH = PROJECT_ROOT / "finetune_dataset"
HEADER_PATH = PROJECT_ROOT / "esp_code/sample_project/main/model.h"
PIPELINE_TEST_PATH = PROJECT_ROOT / "pipeline_test"
