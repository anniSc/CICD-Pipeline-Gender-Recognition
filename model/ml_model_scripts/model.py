import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
import torch


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Erste Convolutional Layer. Nimmt 3 Eingangskanäle (RGB), gibt 6 Kanäle aus, mit einer Kernelgröße von 5
        self.conv1 = nn.Conv2d(3, 6, 5)  
        # Max-Pooling-Layer mit einem quadratischen Fenster der Kernelgröße=4, Schrittgröße=4
        self.pool = nn.MaxPool2d(2, 2)  
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

class DynamicInputCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(DynamicInputCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.adaptive_pool(x)
        x = x.view(x.shape[0], -1)  # Flatten the tensor
        x = self.fc(x)
        return x
    
class GenderRecognitionDataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None):
        self.root_dir = root_dir
        self.annotations = pd.read_csv(annotation_file)
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_id = self.annotations.iloc[index, 0]
        img = Image.open(os.path.join(self.root_dir, img_id)).convert("RGB")
        y_label = torch.tensor(float(self.annotations.iloc[index, 1]))

        if self.transform is not None:
            img = self.transform(img)

        return (img, y_label)

    import torch
from torchvision import models
from torchcam.methods import SmoothGradCAMpp

# Lade dein Modell
model = CNN()
model.load_state_dict(torch.load("model2_params.pt"))

# Gib die letzte Faltungsschicht an
target_layer = model.conv3

# Setze requires_grad auf True für das Modell
model.requires_grad_(True)

# Definiere deine Eingabe
img, label = testset[0]

# Setze requires_grad auf True für die Eingabe
img.requires_grad_(True)

# Erstelle einen CAM-Extraktor mit SmoothGradCAMpp
with SmoothGradCAMpp(model, target_layer) as cam_extractor:
  # Mache eine Vorhersage mit deinem Modell
  out = model(img.unsqueeze(0))
  # Berechne die Aktivierungskarte mit dem CAM-Extraktor
  activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)

# Ersetze torch.tensor durch torch.stack, um deine Liste von Tensoren in einen Tensor zu konvertieren
activation_map = torch.stack(activation_map)

# Plotte die Aktivierungskarte mit dem Originalbild
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
img_permuted = img.permute(1, 2, 0)
plt.imshow(img_permuted.detach().numpy())
plt.title(f'Label: {label}')
plt.subplot(1, 2, 2)
# Überprüfen Sie die Anzahl der Dimensionen von activation_map
if activation_map.dim() == 2:
    activation_map = activation_map.permute(1,0)
elif activation_map.dim() == 3:
    activation_map = activation_map.permute(2,1,0)
else:
    print("Unexpected number of dimensions in activation_map")

activation_map_squeezed = activation_map.squeeze()
plt.imshow(activation_map_squeezed, cmap='jet')
plt.title('Activation Map')
plt.show()





# Importiere die Bibliotheken
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchcam.methods import SmoothGradCAMpp
from torch.autograd import Variable
# Definiere die Architektur des Modells
class CNN(nn.Module):
  def __init__(self):
    super(CNN, self).__init__()
    # Convolutional part
    self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
    self.pool1 = nn.MaxPool2d(2, 2) # output: 32x89x109
    self.conv2 = nn.Conv2d(32, 64, 3, padding=1) # output: 64x89x109
    self.pool2 = nn.MaxPool2d(2, 2) # output: 64x44x54
    self.conv3 = nn.Conv2d(64, 128, 3, padding=1) # output: 128x44x54
    self.pool3 = nn.MaxPool2d(2, 2) # output: 128x22x27
    self.relu = nn.ReLU()
    # Linear part
    self.fc1 = nn.Linear(128*22*27, 256) # output: 256
    # self.fc1 = nn.Linear(100352, 256)
    self.fc2 = nn.Linear(256, 32)
    # self.fc2 = nn.Linear(100352, 40)
    self.sigmoid = nn.Sigmoid() # <- add the sigmoid function

  def forward(self, x):
    # Convolutional part
    x = self.pool1(self.relu(self.conv1(x)))
    x = self.pool2(self.relu(self.conv2(x)))
    x = self.pool3(self.relu(self.conv3(x)))
    # Flatten the output
    x = torch.flatten(x, 1)
    # Linear part
    x = self.relu(self.fc1(x))
    x = self.sigmoid(self.fc2(x)) # <- apply the sigmoid function
    return x

# Erstelle ein Modell-Objekt
model = CNN()

# Erstelle ein Dataset-Objekt
transform = transforms.Compose([transforms.Resize((178, 218)), transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# Ändere den folgenden Code, um ImageFolder zu verwenden
# trainset = torchvision.datasets.CelebA(root='./data', split='train', download=True, transform=transform)
# testset = torchvision.datasets.CelebA(root='./data', split='test', download=True, transform=transform)
trainset = torchvision.datasets.ImageFolder(root=r'C:\Users\busse\Bachelorarbeit\CICD-Pipeline-Gender-Recognition\data\output2\train', transform=transform) # <- use ImageFolder for trainset
testset = torchvision.datasets.ImageFolder(root=r'C:\Users\busse\Bachelorarbeit\CICD-Pipeline-Gender-Recognition\data\output2\val', transform=transform) # <- use ImageFolder for testset

# Erstelle einen DataLoader-Objekt
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)

# Definiere den Verlust und den Optimierer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Trainiere das Modell
num_epochs = 1
for epoch in range(num_epochs):
  running_loss = 0.0
  for i, data in enumerate(trainloader, 0):
    # Hole die Eingaben, die Labels und die Pfade
    inputs, labels = data # <- use only two variables
    labels = labels[:40] # <- remove the colon from the index
    paths = trainset.imgs[i][0] # <- use the imgs attribute to get the path
    b_x = Variable(inputs) # batch x (image)
    b_y = Variable(labels).float() # batch y (target)
    output = model(b_x)[0].float()
    loss = criterion(output, b_y).float()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # Test -> this is where I have no clue
   
print('Finished Training')


