import torch
from torch import nn
import torch.nn.functional as F
import pandas as pd
from PIL import Image
from torchvision.transforms import ToTensor, Resize, Grayscale
from torchvision.io import read_image
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from torchvision import models, transforms
from PIL import Image

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=2)
        self.fc1 = nn.Linear(256 * 6 * 6, 256)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, 256 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc2(x))
        return x
    



    def extract_features(image_paths):
        # Load the pre-trained model
        model = models.resnet50(pretrained=True)

        # Remove the last layer (the classifier)
        model = torch.nn.Sequential(*(list(model.children())[:-1]))

        # Define the image transformations
        transform = transforms.Compose([
            transforms.Resize((178, 218)),
            transforms.Grayscale(num_output_channels=3),  # Convert to RGB
            transforms.ToTensor(),
        ])

        features = []
        for image_path in image_paths:
            # Load and transform the image
            img = Image.open(image_path)
            img = transform(img).unsqueeze(0)  # Add a batch dimension

            # Extract features
            with torch.no_grad():
                feature = model(img)

            features.append(feature)

        features = torch.stack(features)
        return features
# def extract_image_features(images):
#     features = []

#     for image in images:
#         img = read_image(image)
#         img = Resize((178, 218))(img)
#         img = Grayscale()(img)
#         img = ToTensor()(img)
#         features.append(img)

#     features = torch.stack(features)
#     return features


n=300
# Daten vorbereiten
df = pd.read_csv("CICD-Pipeline-Gender-Recognition/model/local_image_path_Gender.csv")
df_sample = df.groupby('Gender', group_keys=False).apply(lambda x: x.sample(n=n, random_state=42))
df_sample = df_sample.sample(frac=1, random_state=42).reset_index(drop=True)
df_sample.to_excel(rf'CICD-Pipeline-Gender-Recognition/model/excel_sheets/Gender_{n}.xlsx', index=False)
X = CNN.extract_features(df_sample['Images'])
# X = X / 255.0
y_gender = np.array(df_sample['Gender'])
X_train, X_test, y_train, y_test = train_test_split(X, y_gender, test_size=0.2, random_state=42)

# PyTorch erwartet (Kanäle, Höhe, Breite), also ändern wir die Form
X_train = X_train.reshape(-1, 1, 4560, 216)
X_test = X_test.reshape(-1, 1, 4560, 216)

# Daten in PyTorch Tensoren umwandeln
X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()
X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float()
# DataLoader erstellen
train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_data, batch_size=32)
test_loader = DataLoader(test_data, batch_size=32)

# Modell erstellen
model = CNN()

# Verlustfunktion und Optimierer definieren
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Trainingsschleife
for epoch in range(30):
    for inputs, labels in train_loader:
        # Nullen Sie die Gradienten
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Verlust berechnen
        loss = criterion(outputs.squeeze(), labels)

        # Backward pass und Optimierung
        loss.backward()
        optimizer.step()

# Modell speichern
torch.save(model.state_dict(), f"model/saved_trained_Models/trained_{n}_model.pt")