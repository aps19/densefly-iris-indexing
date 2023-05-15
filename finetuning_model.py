import os
import torch
from torch import nn, optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd


class IrisDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path).convert("RGB")
        y_label = torch.tensor(float(self.annotations.iloc[index, 1:]))
        
        if self.transform:
            image = self.transform(image)
        
        return (image, y_label)


class IrisModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet50(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.fc = nn.Linear(2048, 2)
        self.model.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.fc.parameters(), lr=0.001)
        self.epochs = 10

    def train(self, loader):
        self.model.train()
        running_loss = 0
        for inputs, labels in loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
        return running_loss / len(loader)

    def validate(self, loader):
        self.model.eval()
        running_loss = 0
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()
        return running_loss / len(loader)

    def fit(self, train_loader, val_loader):
        for epoch in range(self.epochs):
            train_loss = self.train(train_loader)
            val_loss = self.validate(val_loader)
            print(f'Epoch {epoch+1}, Training Loss: {train_loss}, Validation Loss: {val_loss}')


def main():
    # Transforms
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Dataset
    dataset = IrisDataset(csv_file="iris_coordinates.csv", root_dir="$root_path$/IRIS_DATASET", transform=data_transforms)

    # Split the dataset into train, validation, and test
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    # Data loader
    train_loader = DataLoader(dataset=train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=4, shuffle=False)

    # Define the model
    model = IrisModel()

    # Train the model
    model.fit(train_loader, val_loader)


if __name__ == "__main__":
    main()
