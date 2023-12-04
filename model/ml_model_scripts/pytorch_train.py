#!/usr/bin/env python
# coding: utf-8

# # 1. Importieren der benötigten Bibliotheken für das ML-Training

# In[1]:


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
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.optim import Adam
import torch.optim as optim
from tqdm import tqdm

# # 2. ML-Training 

# ## 2.1 Festlegen der epochen und der Batchsize sowie die Transformierung der Daten in einen Tensor und dann noch normalisieren

# In[ ]:


epochs = 50
batch_size = 64

# Epochen speichern um diese für das Testen zu verwenden
with open("test/epochs/epochs.txt", "w") as f:
    f.write(str(epochs))

# Transformation der Daten für das Training und Testen  
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Dateipfade für das Training und Testen festlegen
train_dataset = datasets.ImageFolder(root='data/output/train',transform=transform)
test_dataset = datasets.ImageFolder(root= 'data/output/val',transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# ## 2.2 Erstellen eines eigenen Datasets (falls notwendig)

# In[ ]:



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









# ## 2.3 Überprüfen ob die Bilder richtig angezeigt werden

# In[ ]:


# Display image and label.
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
img = train_features[0].permute(1, 2, 0)
plt.imshow(img, cmap="gray")

# ## 2.4 Erstellen eines CNN-Modells

# ### Das CNN-Modell besteht aus 2 Convolutional Layers, 2 Pooling Layers, 2 Max Pooling Layers, 3 Dense Layers

# In[ ]:


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

# # 3. ML-Modell trainieren

# # Dokumentation für PyTorch Trainingsskript
# 
# Dieses Skript trainiert ein Convolutional Neural Network (CNN) mit PyTorch.
# 
# ## Variablen
# 
# - `model`: Eine Instanz des `SimpleCNN` Modells.
# - `criterion`: Die Verlustfunktion, die während des Trainings verwendet wird. In diesem Fall wird die Cross-Entropy-Loss-Funktion verwendet.
# - `optimizer`: Der Optimierer, der zur Aktualisierung der Modellparameter verwendet wird. Hier wird der Stochastic Gradient Descent (SGD) Optimierer verwendet.
# - `patience`: Die Anzahl der Epochen, die auf eine Verbesserung der Genauigkeit gewartet wird, bevor das Training gestoppt wird.
# - `best_accuracy`: Die beste Genauigkeit, die während des Trainings erreicht wurde. Initialisiert auf 0.
# - `early_stopping_counter`: Zählt die Anzahl der Epochen ohne Verbesserung der Genauigkeit.
# 
# ## Trainingsschleife
# 
# Das Modell wird für eine bestimmte Anzahl von Epochen trainiert. In jeder Epoche wird das Modell mit den Trainingsdaten trainiert und dann mit den Testdaten validiert.
# 
# Während des Trainings werden die Modellparameter aktualisiert, um den Verlust zu minimieren. Der Verlust wird berechnet, indem die Ausgabe des Modells und die tatsächlichen Labels verglichen werden.
# 
# Nach jeder Epoche wird die Genauigkeit des Modells auf den Testdaten berechnet. Wenn die Genauigkeit über 90% liegt, wird der aktuelle Zustand des Modells gespeichert. Wenn die Genauigkeit nicht besser ist als die bisher beste Genauigkeit, wird der `early_stopping_counter` erhöht. Wenn der `early_stopping_counter` den Wert von `patience` erreicht, wird das Training gestoppt.
# 
# ## Ausgabe
# 
# Das Skript gibt den Verlust und die Genauigkeit nach jeder Epoche aus. Wenn das Training aufgrund von Early Stopping gestoppt wird, wird eine entsprechende Nachricht ausgegeben. Am Ende des Trainings wird eine Nachricht ausgegeben, dass das Training abgeschlossen ist.

# In[ ]:



model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
patience = 10 
best_accuracy = 0.0  
early_stopping_counter = 0  

for epoch in range(epochs): 
    running_loss = 0.0
    for i, data in enumerate(tqdm(train_dataloader), 0):
      
        inputs, labels = data

   
        optimizer.zero_grad()

   
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()  

   
        running_loss += loss.item()

  
    correct = 0
    total = 0
    if i % 10 == 9: 
        print('[%d, %5d] loss: %.3f' %
              (epoch + 1, i + 1, running_loss / 100))
        running_loss = 0.0
        
    with torch.no_grad():
        for val_data in test_dataloader:
            val_images, val_labels = val_data
            val_outputs = model(val_images)
            _, predicted = torch.max(val_outputs.data, 1)
            total += val_labels.size(0)
            correct += (predicted == val_labels).sum().item()
    accuracy = correct / total


    if accuracy > 0.9:  
        torch.save(model.state_dict(), f'model/PyTorch_Trained_Models/model_epoch_{epoch+1}_accuracy_{accuracy:.2f}.pth')
   
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        early_stopping_counter = 0
        print(f"Genauigkeit: {accuracy:.2f}")
    else:
        early_stopping_counter += 1

    if early_stopping_counter >= patience:
        print('Early stopping')
        break

print('Finished Training')

model_path_git = f'model/PyTorch_Trained_Models/'
torch.save(model.state_dict(), f'{model_path_git}model_git{batch_size}' + '-' + f'{epochs}' + '.pth')

# In[ ]:


model_path = f'{model_path_git}model_git{batch_size}' + '-' + f'{epochs}' + '.pth'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SimpleCNN()
# model.load_state_dict(torch.load(f'model/model{batch_size}' + '-' + f'{epochs}' + '.pth'))
model.load_state_dict(torch.load(model_path))

# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

for inputs, _ in test_dataloader:
    inputs = inputs.to(device) 
    output = model(inputs)
    probabilities = torch.nn.functional.softmax(output, dim=1)
    predictions = torch.argmax(probabilities, dim=1)
    predictions_list = predictions.cpu().numpy().tolist()
    print(predictions_list)

# In[ ]:


# Load the model
model = SimpleCNN()
# model.load_state_dict(torch.load(f'model/model{batch_size}' + '-' + f'{epochs}' + '.pth'))
model.load_state_dict(torch.load(model_path))
# Assume `test_dataset` is your ImageFolder dataset
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Use the model to make predictions and calculate accuracy
correct = 0
total = 0

for inputs, labels in test_dataloader:
    inputs = inputs.to(device)
    labels = labels.to(device)
    
    # Forward pass
    outputs = model(inputs)
    _, predicted = torch.max(outputs.data, 1)
    
    # Update total and correct counts
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

# Calculate accuracy
accuracy = correct / total
print('Genauigkeit des Modells auf Testbilder: {}%'.format(100 * accuracy))

# In[ ]:


# Setze das Modell in den Evaluierungsmodus
model.eval() 
predictions = []
true_labels = []
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        predictions.extend(predicted)
        true_labels.extend(labels)


# Konvertiere die Listen in numpy-Arrays
predictions = np.array(predictions)
true_labels = np.array(true_labels)

# Berechne die Metriken
accuracy = accuracy_score(true_labels, predictions)
precision = precision_score(true_labels, predictions, average='weighted')
recall = recall_score(true_labels, predictions, average='weighted')
f1 = f1_score(true_labels, predictions, average='weighted')

print(f'Genauigkeit: {accuracy}, Präzision: {precision}, Recall: {recall}, F1-Score: {f1}')

with open('model/metrics/metrics.txt', 'w') as outfile:
    outfile.write(f'Modellmetriken: Genauigkeit: {accuracy}, Präzision: {precision}, Recall: {recall}, F1-Score: {f1}')


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

model.eval()  # Set the model to evaluation mode
predictions = []
labels_list = []
with torch.no_grad():
    for data, labels in test_loader:
        outputs = model(data)
        
        # Flatten the outputs and convert to numpy array
        predictions.extend(outputs.view(-1).cpu().numpy())
        labels_list.extend(labels.cpu().numpy())
        
# Flatten the arrays
predictions = np.array(predictions).ravel()
labels_list = np.array(labels_list).ravel()
print(predictions)
print(labels_list)

# Ensure both arrays have the same length
min_length = min(len(predictions), len(labels_list))
predictions = predictions[:min_length]
labels_list = labels_list[:min_length]

predictions = np.array(predictions).ravel()
labels_list = np.array(labels_list).ravel()

# Plot true labels against predictions
# plt.scatter(labels_list,predictions)
plt.scatter(labels_list,predictions)
plt.grid(True)
plt.xlabel('True Labels')
plt.ylabel('Predictions')
plt.xlabel('True Labels',color='blue')
plt.ylabel('Predictions',color='red')
plt.title('True Labels vs Predictions')

plt.savefig("model/plots/plot_scatter.jpg",dpi=100)
plt.show()


plt.plot([labels_list.min(), labels_list.max()], [predictions.min(), predictions.max()], 'k--', lw=4)
plt.grid(True)
plt.xlabel('True Labels')
plt.ylabel('Predictions')
plt.xlabel('True Labels',color='blue')
plt.ylabel('Predictions',color='red')
plt.title('True Labels vs Predictions')
plt.savefig("model/plots/plot_plt.jpg",dpi=100)
plt.show()

# Create a 2D histogram from the data
heatmap_data, xedges, yedges = np.histogram2d(labels_list, predictions, bins=50)


# Plot the heatmap
plt.imshow(heatmap_data, origin='lower', cmap='hot', interpolation='nearest')
plt.colorbar(label='Anzahl')
plt.xlabel('True Labels')
plt.ylabel('Predictions')
plt.title('Heatmap of True Labels vs Predictions')
plt.savefig("model/plots/heatmap.jpg", dpi=100)
plt.show()
