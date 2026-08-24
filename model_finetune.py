import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import ocr_model
from image_processing_pipeline import img_pipline
import config

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(root=config.FINETUNE_DATASET_PATH, transform=transform)
print("class mapping: ", dataset.class_to_idx)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
print(f"Total size: {len(dataset)}, Train size: {train_size}, Val size: {val_size}")
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

training_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

model = ocr_model()
state_dict = torch.load(config.PTH_MODEL, weights_only=True)
model.load_state_dict(state_dict)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

def train_one_epoch():
    current_loss = 0
    last_loss = 0

    for i, data in enumerate(training_loader):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        current_loss += loss.item()

        if i % 100 == 99:
            last_loss = current_loss / 100
            print(f"    batch {i+1}, loss: {last_loss}")
            current_loss = 0
    return last_loss

epochs = 50
# best_avg_loss = 1_000_000.
best_val_loss = 1_000_000.
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}")
    model.train(True)
    avg_loss = train_one_epoch()

    model.eval()
    current_val_loss = 0.0
    with torch.no_grad():
        for i, val_data in enumerate(val_loader):
            val_inputs, val_labels = val_data
            val_outputs = model(val_inputs)
            val_loss = loss_fn(val_outputs, val_labels)
            current_val_loss = val_loss

    avg_val_loss = current_val_loss / (i+1)
    print(f'LOSS train {avg_loss} valid {avg_val_loss}')

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        model_path = f"finetuned_val_model_{epoch}.pth"
        torch.save(model.state_dict(), model_path)
        print(f"A new best model {epoch}, saving...")
'''
    if avg_loss < best_avg_loss:
        best_avg_loss = avg_loss
        model_path = f"finetuned_model_{epoch}.pth"
        torch.save(model.state_dict(), model_path)
        print(f"A new best model {epoch}, saving...")
'''
print("Finished training")
