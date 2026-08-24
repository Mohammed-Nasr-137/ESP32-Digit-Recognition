import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import ocr_model
from torchinfo import summary
import config

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])

full_dataset = datasets.ImageFolder(root=config.DATASET_PATH, transform=transform)
print("PyTorch Class Mapping:", full_dataset.class_to_idx)

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

model = ocr_model()
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
summary(model, input_size=(1, 1, 16, 16))

def train_one_epoch(epoch_index):
    current_loss = 0
    last_loss = 0

    for i, data in enumerate(train_loader):
        inputs, labels = data
        optimizer.zero_grad()

        outputs = model(inputs)

        loss = loss_fn(outputs, labels)
        loss.backward()

        optimizer.step()

        current_loss += loss.item()
        if i % 1000 == 999:
            last_loss = current_loss / 1000
            print(f"    batch {i+1}, loss: {last_loss}")
            current_loss = 0

    return last_loss


epochs = 10
best_val_loss = 1_000_000.

for epoch in range(epochs):
    try:
        print(f"Epoch {epoch + 1}:")

        model.train(True)
        avg_loss = train_one_epoch(epoch)

        model.eval()
        current_val_loss = 0.0
        with torch.no_grad():
            for i, val_data in enumerate(val_loader):
                val_inputs, val_labels = val_data
                val_outputs = model(val_inputs)
                val_loss = loss_fn(val_outputs, val_labels)
                current_val_loss = val_loss

        avg_val_loss = current_val_loss / (i + 1)
        print(f'LOSS train {avg_loss} valid {avg_val_loss}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_path = f"model_{epoch}.pth"
            torch.save(model.state_dict(), model_path)
            print(f"A new best model {epoch}, saving...")
    except KeyboardInterrupt:
        print(f"Training interrupted, saving model {epoch}...")
        model_path = f"model_{epoch}.pth"
        torch.save(model.state_dict(), model_path)
        exit(0)

print("Finished training")
