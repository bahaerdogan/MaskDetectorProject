import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.nn import CrossEntropyLoss
import numpy as np

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  
])

dataset = datasets.ImageFolder(root="./vit_dataset/train", transform=transform)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

class_names = ["with_mask", "without_mask", "mask_weared_incorrect"]
class_counts = np.array([len(os.listdir(os.path.join("./vit_dataset/train", class_name))) for class_name in class_names])
class_weights = 1. / class_counts
weights = torch.tensor(class_weights, dtype=torch.float32).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k", num_labels=len(class_names))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = AdamW(model.parameters(), lr=1e-4)
scheduler = StepLR(optimizer, step_size=3, gamma=0.1) 
criterion = CrossEntropyLoss(weight=weights)

for epoch in range(20):  
    model.train()
    total_loss = 0
    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images).logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    scheduler.step() 
    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(data_loader)}")

torch.save(model.state_dict(), f"vit_model_epoch_{epoch + 1}.pth")

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images).logits
        predictions = torch.argmax(outputs, dim=-1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

print(f"Accuracy: {correct / total * 100:.2f}%")
