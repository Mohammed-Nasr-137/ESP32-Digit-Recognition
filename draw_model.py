import torch
from torchview import draw_graph
from model import ocr_model

model = ocr_model()
model.eval()

dummy_input = torch.randint(0, 1, (1, 1, 16, 16), dtype=torch.float32)
graph = draw_graph(
    model,
    input_data=dummy_input,
    depth=3,
    expand_nested=True,
    save_graph=True,
    filename="ocr_model"
)
