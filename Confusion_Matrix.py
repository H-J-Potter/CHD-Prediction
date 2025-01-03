import torch
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from monai.data import DataLoader
from monai.transforms import Compose, EnsureChannelFirst, Resize, ScaleIntensity, ToTensor, LoadImage
from monai.networks.nets import DenseNet121
import pandas as pd
import os

# Constants
CSV_FILE = "labeled_frames.csv"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
MODEL_PATH = "echocardiogram_model_1.pth"

# Transformation pipeline
transforms = Compose([
    EnsureChannelFirst(),
    Resize(IMAGE_SIZE),
    ScaleIntensity(),
    ToTensor()
])

# Load CSV
df = pd.read_csv(CSV_FILE)
image_paths = df['file_path'].tolist()
labels = df['label'].tolist()

# Dataset for validation
class EchocardiogramDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = torch.zeros((1, 224, 224))  # Default if file not found
        if os.path.exists(image_path):
            image = LoadImage(image_only=True)(image_path)
            if self.transform:
                image = self.transform(image)
        return image, label

# Validation DataLoader
dataset = EchocardiogramDataset(image_paths, labels, transform=transforms)
val_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load your pre-trained model
class ModifiedDenseNet121(torch.nn.Module):
    def __init__(self):
        super(ModifiedDenseNet121, self).__init__()
        self.base_model = DenseNet121(spatial_dims=2, in_channels=1, out_channels=1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.base_model(x)
        return self.sigmoid(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ModifiedDenseNet121()
model.load_state_dict(torch.load(MODEL_PATH))
model.to(device)
model.eval()

# Collect predictions and labels
all_preds, all_labels = [], []
with torch.no_grad():
    for inputs, targets in val_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        preds = (outputs.squeeze() > 0.5).float().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(targets.numpy())

# Compute confusion matrix
all_preds = [int(p) for p in all_preds]
all_labels = [int(l) for l in all_labels]
cm = confusion_matrix(all_labels, all_preds)

print("Confusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=['No CHD', 'CHD']))

# Visualize confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No CHD', 'CHD'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()