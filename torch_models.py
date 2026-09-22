import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class ImagePaths(Dataset):
    def __init__(self, paths, labels, augment=False):
        self.paths = paths
        self.labels = labels
        transformations = [transforms.Resize((64, 64))]
        if augment:
            transformations += [transforms.RandomHorizontalFlip(), transforms.RandomRotation(15)]
        self.transform = transforms.Compose(transformations + [transforms.ToTensor()])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, int(self.labels[index])


def make_loader(paths, labels, augment=False, shuffle=False, batch=16):
    generator = torch.Generator().manual_seed(7331)
    return DataLoader(ImagePaths(paths, labels, augment), batch_size=batch, shuffle=shuffle, generator=generator, num_workers=0)


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 2))

    def forward(self, inputs):
        return self.classifier(self.features(inputs))


class CNNViT(nn.Module):
    def __init__(self, pretrained_cnn):
        super().__init__()
        self.features = pretrained_cnn.features
        for parameter in self.features.parameters():
            parameter.requires_grad = False
        self.projection = nn.Conv2d(64, 64, 1)
        self.position = nn.Parameter(torch.randn(1, 64, 64) * 0.02)
        encoder = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder, num_layers=1, enable_nested_tensor=False)
        self.head = nn.Linear(64, 2)

    def forward(self, inputs):
        with torch.no_grad():
            features = self.features(inputs)
        tokens = self.projection(features).flatten(2).transpose(1, 2)
        return self.head(self.encoder(tokens + self.position).mean(dim=1))


def fit_model(model, train_loader, val_loader, epochs=2, device="cpu"):
    model.to(device)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    history = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}
    best_loss, best_weights = float("inf"), None
    for epoch in range(epochs):
        for phase, loader in (("train", train_loader), ("val", val_loader)):
            model.train(phase == "train")
            correct, total, loss_sum = 0, 0, 0.0
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    if phase == "train":
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                loss_sum += loss.item() * len(labels)
                correct += (outputs.argmax(dim=1) == labels).sum().item()
                total += len(labels)
            suffix = "" if phase == "train" else "val_"
            history[f"{suffix}loss"].append(loss_sum / total)
            history[f"{suffix}accuracy"].append(correct / total)
        if history["val_loss"][-1] < best_loss:
            best_loss = history["val_loss"][-1]
            best_weights = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(f"Epoch {epoch + 1}/{epochs} | train acc {history['accuracy'][-1]:.3f} | val acc {history['val_accuracy'][-1]:.3f} | val loss {history['val_loss'][-1]:.3f}")
    model.load_state_dict(best_weights)
    return history


def predict_probabilities(model, loader, device="cpu"):
    model.to(device).eval()
    probabilities = []
    with torch.no_grad():
        for images, _ in loader:
            probabilities.extend(model(images.to(device)).softmax(dim=1)[:, 1].cpu().numpy())
    return np.array(probabilities)
