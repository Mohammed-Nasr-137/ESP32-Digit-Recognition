import torch
import tensorflow as tf
import onnx2tf
from model import ocr_model
from torchinfo import summary
import config

# export pytorch to onnx
model = ocr_model()
model.load_state_dict(torch.load(config.FINETUNED_PTH_MODEL, weights_only=True))
model.eval()

# summary(model, input_size=(1, 1, 16, 16))
dummy_input = torch.randn(1, 1, 16, 16, dtype=torch.float32)
onnx_path = config.ONNX_MODEL

torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    export_params=True,
    # opset_version=13,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes=None
)

tf_path = config.TF_PATH
onnx2tf.convert(
    input_onnx_file_path=onnx_path,
    output_folder_path=tf_path,
    flatbuffer_direct_output_saved_model=True,
    non_verbose=True
)
