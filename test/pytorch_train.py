import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn, cuda
from torchvision import datasets, transforms, models
from torchvision.transforms import ToTensor
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
from torch.utils.data import DataLoader
import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn, cuda
from torchvision import datasets, transforms, models
from torchvision.transforms import ToTensor
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
from torch.utils.data import DataLoader
# from skimage import io
from torch.utils.data import Dataset
from torch.optim import Adam

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 1])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 2]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
class GenderRecognitionDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.annotations.iloc[idx, 1])
        image = io.imread(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[idx, 2]))
        if self.transform:
            image = self.transform(image)
        return (image,y_label)


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Erste Convolutional Layer. Nimmt 3 Eingangskanäle (RGB), gibt 6 Kanäle aus, mit einer Kernelgröße von 5
        self.conv1 = nn.Conv2d(3, 6, 5)  
        # Max-Pooling-Layer mit einem quadratischen Fenster der Kernelgröße=4, Schrittgröße=4
        self.pool = nn.MaxPool2d(4, 4)  
        # Zweite Convolutional Layer. Nimmt 6 Eingangskanäle (von der vorherigen Schicht), gibt 16 Kanäle aus, mit einer Kernelgröße von 5
        self.conv2 = nn.Conv2d(6, 16, 5)
        # Max-Pooling-Layer mit einem quadratischen Fenster der Kernelgröße=2, Schrittgröße=2
        self.pool = nn.MaxPool2d(2,2) 
        # Erste vollständig verbundene Schicht. Nimmt einen abgeflachten Vektor der Größe 33456 auf, gibt einen Vektor der Größe 120 aus
        self.fc1 = nn.Linear(33456 , 120) 
        # Zweite vollständig verbundene Schicht. Nimmt einen Vektor der Größe 120 auf, gibt einen Vektor der Größe 84 aus
        self.fc2 = nn.Linear(120, 84)
        # Dritte vollständig verbundene Schicht. Nimmt einen Vektor der Größe 84 auf, gibt einen Vektor der Größe 2 aus
        self.fc3 = nn.Linear(84, 2) 

    def forward(self, x):
        # Anwendung der ersten Conv-Schicht, dann ReLU-Aktivierungsfunktion, dann Max-Pooling
        x = self.pool(F.relu(self.conv1(x)))
        # Anwendung der zweiten Conv-Schicht, dann ReLU-Aktivierungsfunktion, dann Max-Pooling
        x = self.pool(F.relu(self.conv2(x)))
        # Abflachen des Tensorausgangs von den Conv-Schichten
        x = x.view(x.size(0), -1) 
        # Anwendung der ersten vollständig verbundenen Schicht, dann ReLU-Aktivierungsfunktion
        x = F.relu(self.fc1(x))  
        # Anwendung der zweiten vollständig verbundenen Schicht, dann ReLU-Aktivierungsfunktion
        x = F.relu(self.fc2(x)) 
        # Anwendung der dritten vollständig verbundenen Schicht
        x = self.fc3(x)  
        return x



