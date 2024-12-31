import os
import pandas as pd
import torch
from monai.data import Dataset, DataLoader
from monai.transforms import Compose, LoadImage, EnsureChannelFirst, ScaleIntensity, Resize, ToTensor
from torch.utils.data import random_split
import torch.nn as nn
import torch.optim as optim
from monai.networks.nets import DenseNet121

# Constants
CSV_FILE = "labeled_frames.csv"
IMAGE_SIZE = (224, 224)  # Resize all images to this size (adjust as needed)
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
MODEL_PATH = "echocardiogram_model.pth"

# Load CSV with paths and labels
df = pd.read_csv(CSV_FILE)

# Assuming the CSV has columns 'file_path' and 'label'
image_paths = df['file_path'].tolist()
labels = df['label'].tolist()

# Check for invalid file paths
invalid_paths = [path for path in image_paths if not os.path.exists(path)]
if invalid_paths:
    print("Invalid image paths found:", invalid_paths)

# Define the transformation pipeline for the images
transforms = Compose([
    EnsureChannelFirst(),
    Resize(IMAGE_SIZE),
    ScaleIntensity(),
    ToTensor()
])

# Prepare the dataset
class EchocardiogramDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Check if the image path exists
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            return torch.zeros((1, 224, 224)), label  # Return default image (modify as needed)

        # Load the image
        image = LoadImage(image_only=True)(image_path)
        
        # Apply transformations if any
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Split dataset into training and validation sets
dataset = EchocardiogramDataset(image_paths, labels, transform=transforms)
train_size = int(0.8 * len(dataset))  # 80% training data
val_size = len(dataset) - train_size  # 20% validation data
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoader for batching
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Define the model (using a modified DenseNet121)
class ModifiedDenseNet121(nn.Module):
    def __init__(self):
        super(ModifiedDenseNet121, self).__init__()
        self.base_model = DenseNet121(spatial_dims=2, in_channels=1, out_channels=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.base_model(x)
        return self.sigmoid(x)  # Output between 0 and 1

# Instantiate the model, loss function, and optimizer
model = ModifiedDenseNet121()
criterion = nn.BCELoss()  # Binary Cross Entropy loss for binary classification
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.float().to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss/len(train_loader):.4f}")

    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.float().to(device)
            outputs = model(inputs)
            val_loss += criterion(outputs.squeeze(), targets).item()
            predicted = (outputs.squeeze() > 0.5).float()
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    print(f"Validation Loss: {val_loss/len(val_loader):.4f}, Accuracy: {correct/total:.4f}")

# Save the trained model
torch.save(model.state_dict(), MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")