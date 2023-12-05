#!/usr/bin/env python
# coding: utf-8

# # 1. Importieren der benötigten Bibliotheken

# # 2. Testdataset erstellen

# In[ ]:


import pytorch_train as pt
import torch
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
from pytorch_train import SimpleCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

model = SimpleCNN()  # Instantiate the model
def get_model_path(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".pth"):
            return os.path.join(directory, filename)
    return None

model_path = get_model_path("test/model_to_be_tested/")
print(model_path)

# model_path = f'model/PyTorch_Trained_Models/model64-91acc-100.pth'
model.load_state_dict(torch.load(model_path, map_location=device))  # Load the state_dict
test_dataset = datasets.ImageFolder("data/output/val", transform=transform)

test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

for inputs, _ in test_dataloader:
    inputs = inputs.to(device) 
    output = model(inputs)
    probabilities = torch.nn.functional.softmax(output, dim=1)
    predictions = torch.argmax(probabilities, dim=1)
    predictions_list = predictions.cpu().numpy().tolist()
    print(predictions_list)

   

# ## 2.1 Robustheit des ML-Models Testen auf Testbilder

# In[ ]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def test_model_robustness(model, test_dataloader, device):
    model.eval()  # Set the model to evaluation mode
    true_labels = []
    predicted_labels = []

    with torch.no_grad():  # Do not calculate gradients to speed up computation
        for inputs, labels in test_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            output = model(inputs)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            predictions = torch.argmax(probabilities, dim=1)

            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predictions.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    f1 = f1_score(true_labels, predicted_labels, average='weighted')

    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')
    return accuracy, precision, recall, f1


test_model_robustness(model, test_dataloader, device)

# ## 2.2 Testdaten verrauschen

# ### `add_noise` Funktion
# Die Funktion `add_noise` fügt einem gegebenen Bild Rauschen hinzu.
# 
# #### Parameter
# 
# - `images`: Die Bilder, zu denen Rauschen hinzugefügt werden soll.
# - `noise_factor`: Der Faktor, der bestimmt, wie viel Rauschen hinzugefügt wird. Standardmäßig ist dieser Wert 0.5.
# 
# #### Rückgabewert
# 
# Die Funktion gibt die verrauschten Bilder zurück.
# 
# ### `add_noise_and_test` Funktion
# 
# Die Funktion `add_noise_and_test` fügt den Testbildern Rauschen hinzu und testet dann das Modell mit diesen verrauschten Bildern.
# 
# #### Parameter
# 
# - `model`: Das zu testende Modell.
# - `test_dataloader`: Ein DataLoader, der die Testdaten bereitstellt.
# - `device`: Das Gerät, auf dem das Modell ausgeführt wird (z.B. 'cpu' oder 'cuda').
# - `noise_factor`: Der Faktor, der bestimmt, wie viel Rauschen hinzugefügt wird. Standardmäßig ist dieser Wert 0.5.
# 
# #### Funktionsweise
# 
# Die Funktion fügt den Testbildern Rauschen hinzu und erstellt dann einen neuen DataLoader mit den verrauschten Bildern. Anschließend wird das Modell mit diesen verrauschten Bildern getestet.
# 
# ### Beispiel
# 
# ```python
# add_noise_and_test(model, test_dataloader, device)
# ```
# 
# In diesem Beispiel wird die Funktion `add_noise_and_test` aufgerufen, um Rauschen zu den Testbildern hinzuzufügen und dann das Modell `model` mit diesen verrauschten Bildern zu testen. Die Testdaten werden vom `test_dataloader` bereitgestellt und das Modell wird auf dem `device` ausgeführt. Der Rauschfaktor beträgt 0.5.

# In[ ]:


def add_noise(images, noise_factor=0.5):
    noise = torch.randn_like(images) * noise_factor
    noisy_images = images + noise
    noisy_images = torch.clamp(noisy_images, 0., 1.)
    return noisy_images

def add_noise_and_test(model, test_dataloader, device, noise_factor=0.5):
    noisy_images = []
    labels_list = []
    
    for inputs, labels in test_dataloader:
        inputs_noisy = add_noise(inputs, noise_factor)
        noisy_images.append(inputs_noisy)
        labels_list.append(labels)

    # Stack images and labels separately
    noisy_images = torch.cat(noisy_images)
    labels_list = torch.cat(labels_list)

    noisy_dataloader = torch.utils.data.DataLoader(list(zip(noisy_images, labels_list)), batch_size=test_dataloader.batch_size)

    test_model_robustness(model, noisy_dataloader, device)

# Verwendung der Funktion
add_noise_and_test(model, test_dataloader, device)



# ## 2.3 Model auf verrauschte Bilder testen
# 

# ### Dokumentation für `test_noise_robustness` Funktion
# 
# Die Funktion `test_noise_robustness` testet die Robustheit eines Modells gegenüber Bildrauschen. Sie nimmt ein Modell, einen DataLoader für Testdaten, ein Gerät und optionale Parameter für den Start-, End- und Schrittwert des Rauschfaktors an.
# 
# ### Parameter
# 
# - `model`: Das zu testende Modell.
# - `test_dataloader`: Ein DataLoader, der die Testdaten bereitstellt.
# - `device`: Das Gerät, auf dem das Modell ausgeführt wird (z.B. 'cpu' oder 'cuda').
# - `start_noise`: Der Startwert für den Rauschfaktor. Standardmäßig ist dieser Wert 0.0.
# - `end_noise`: Der Endwert für den Rauschfaktor. Standardmäßig ist dieser Wert 1.0.
# - `step`: Der Schrittwert, um den der Rauschfaktor bei jedem Durchlauf erhöht wird. Standardmäßig ist dieser Wert 0.1.
# 
# ### Funktionsweise
# 
# Die Funktion fügt den Testbildern schrittweise Rauschen hinzu, beginnend mit dem `start_noise`-Wert und endend mit dem `end_noise`-Wert. Bei jedem Schritt wird das Modell mit den verrauschten Bildern getestet und die Genauigkeit, Precision, Recall und der F1-Score werden berechnet und neben einem Beispielbild angezeigt.
# 
# Wenn die Genauigkeit des Modells unter 70% fällt, wird der Test gestoppt.
# 
# ### Beispiel
# 
# ```python
# test_noise_robustness(model, test_dataloader, device)
# ```
# 
# In diesem Beispiel wird die Funktion `test_noise_robustness` aufgerufen, um die Robustheit des Modells `model` gegenüber Bildrauschen zu testen. Die Testdaten werden vom `test_dataloader` bereitgestellt und das Modell wird auf dem `device` ausgeführt. Der Rauschfaktor startet bei 0.0 und endet bei 1.0, wobei er bei jedem Durchlauf um 0.1 erhöht wird.
# 
# 
# 
# Die Funktion `test_noise_robustness` testet die Robustheit eines Modells gegenüber Bildrauschen. Sie nimmt ein Modell, einen DataLoader für Testdaten, ein Gerät und optionale Parameter für den Start-, End- und Schrittwert des Rauschfaktors an.
# 
# ### Parameter
# 
# - `model`: Das zu testende Modell.
# - `test_dataloader`: Ein DataLoader, der die Testdaten bereitstellt.
# - `device`: Das Gerät, auf dem das Modell ausgeführt wird (z.B. 'cpu' oder 'cuda').
# - `start_noise`: Der Startwert für den Rauschfaktor. Standardmäßig ist dieser Wert 0.0.
# - `end_noise`: Der Endwert für den Rauschfaktor. Standardmäßig ist dieser Wert 1.0.
# - `step`: Der Schrittwert, um den der Rauschfaktor bei jedem Durchlauf erhöht wird. Standardmäßig ist dieser Wert 0.1.
# 
# ### Funktionsweise
# 
# Die Funktion fügt den Testbildern schrittweise Rauschen hinzu, beginnend mit dem `start_noise`-Wert und endend mit dem `end_noise`-Wert. Bei jedem Schritt wird das Modell mit den verrauschten Bildern getestet und die Genauigkeit, Precision, Recall und der F1-Score werden berechnet und neben einem Beispielbild angezeigt.
# 
# Wenn die Genauigkeit des Modells unter 70% fällt, wird der Test gestoppt.
# 
# ### Beispiel
# 
# ```python
# test_noise_robustness(model, test_dataloader, device)
# ```
# 
# In diesem Beispiel wird die Funktion `test_noise_robustness` aufgerufen, um die Robustheit des Modells `model` gegenüber Bildrauschen zu testen. Die Testdaten werden vom `test_dataloader` bereitgestellt und das Modell wird auf dem `device` ausgeführt. Der Rauschfaktor startet bei 0.0 und endet bei 1.0, wobei er bei jedem Durchlauf um 0.1 erhöht wird.

# In[ ]:


def test_noise_robustness(model, test_dataloader, device, start_noise=0.0, end_noise=1.0, step=0.1):
    noise_factor = start_noise
    i = 0
    while noise_factor <= end_noise:
        noisy_images = []
        labels_list = []

        for inputs, labels in test_dataloader:
            inputs_noisy = add_noise(inputs, noise_factor)
            noisy_images.append(inputs_noisy)
            labels_list.append(labels)

        noisy_images = torch.cat(noisy_images)
        labels_list = torch.cat(labels_list)

        noisy_dataloader = torch.utils.data.DataLoader(list(zip(noisy_images, labels_list)), batch_size=test_dataloader.batch_size)

        print(f'Test mit noise factor: {noise_factor}')
        accuracy, precision,recall, f1 = test_model_robustness(model, noisy_dataloader, device)
        

        # Show a sample noisy image
        plt.figure(figsize=(12, 6))
        plt.title(f'Verauschtes Bild mit noise factor: {noise_factor}.', fontsize=10)
        plt.imshow(noisy_images[0].permute(1, 2, 0))
        plt.text(1.2, 0.6, f'Genauigkeit: {accuracy}', horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)
        plt.text(1.2, 0.5, f'Precision: {precision}', horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)
        plt.text(1.2, 0.4, f'Recall: {recall}', horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)
        plt.text(1.2, 0.3, f'F1 Score: {f1}', horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)
        plt.savefig(f"test/test-plots-rauschen/{i}.png")
        print(f"test/test-plots-verzerrung/{i}.png")
        plt.show()

        if accuracy < 0.7:
            print(f'Genauigkeit unter 70% mit noise factor: {noise_factor}. Test wird gestoppt.')
            break
        i += 1
        noise_factor += step

# Verwendung der Funktion
test_noise_robustness(model, test_dataloader, device)

# ## 2.4 Testdaten mit Verzerrungen erstellen 
# 

# In[ ]:


import torchvision.transforms.functional as TF


def add_distortion(image, distortion_factor=0.5):
    # Create the distortion matrix
    startpoints = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    endpoints = startpoints + torch.tensor([[0.0, distortion_factor], [0.0, -distortion_factor], [0.0, 0.0], [0.0, 0.0]])

    # Apply the distortion to the image
    distorted_image = TF.perspective(image, startpoints, endpoints)

    return distorted_image

# def add_distortion_and_test(model, test_dataloader, device, distortion_factor=0.0):
#     distorted_images = []
#     labels_list = []
#     while distortion_factor <= 1.0:
#         for inputs, labels in test_dataloader:
#             inputs_distorted = add_distortion(inputs, distortion_factor)
#             distorted_images.append(inputs_distorted)  # Append tensor directly
#             labels_list.append(labels)  # Append tensor directly

#         distorted_images = torch.stack(distorted_images)  # Convert list of tensors to tensor
#         labels_list = torch.cat(labels_list)

#         distorted_dataloader = torch.utils.data.DataLoader(list(zip(distorted_images, labels_list)), batch_size=test_dataloader.batch_size)

#         accuracy, recall, precision, f1 = test_model_robustness(model, distorted_dataloader, device)

#         # Show a sample distorted image
#         plt.figure(figsize=(2, 2))
#         plt.imshow(inputs_distorted[0].permute(1, 2, 0))
#         plt.title(f'Distorted Image with Distortion Factor: {distortion_factor}')
#         plt.text(1.2, 0.6, f'Genauigkeit: {accuracy}', horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)
#         plt.text(1.2, 0.5, f'Precision: {precision}', horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)
#         plt.text(1.2, 0.4, f'Recall: {recall}', horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)
#         plt.text(1.2, 0.3, f'F1 Score: {f1}', horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)
#         plt.savefig(f"test/test-plots-verzerrung/plot{distortion_factor}-img.png")
#         plt.show()

#         if accuracy < 0.7:
#             print(f'Genauigkeit unter 70% mit noise factor: {distortion_factor}. Test wird gestoppt.')
#             break
#         distortion_factor += 0.0001

# Verwendung der Funktion
# add_distortion_and_test(model, test_dataloader, device)
        

# In[ ]:


def test_distortion_robustness(model, test_dataloader, device, start_distortion=0.0, end_distortion=1.0, step=0.0001):
    distortion_factor = start_distortion
    i = 0
    while distortion_factor <= end_distortion:
        distorted_images = []
        labels_list = []

        for inputs, labels in test_dataloader:
            inputs_distorted = add_distortion(inputs, distortion_factor)
            distorted_images.append(inputs_distorted)
            labels_list.append(labels)

        distorted_images = torch.cat(distorted_images)
        labels_list = torch.cat(labels_list)

        distorted_dataloader = torch.utils.data.DataLoader(list(zip(distorted_images, labels_list)), batch_size=test_dataloader.batch_size)

        print(f'Test mit distortion factor: {distortion_factor}')
        accuracy, precision, recall, f1 = test_model_robustness(model, distorted_dataloader, device)
        
        # Show a sample distorted image
        plt.figure(figsize=(12, 6))
        plt.title(f'Verzerrtes Bild mit distortion factor: {distortion_factor}.', fontsize=10)
        plt.imshow(distorted_images[0].permute(1, 2, 0))
        plt.text(1.2, 0.6, f'Genauigkeit: {accuracy}', horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)
        plt.text(1.2, 0.5, f'Precision: {precision}', horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)
        plt.text(1.2, 0.4, f'Recall: {recall}', horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)
        plt.text(1.2, 0.3, f'F1 Score: {f1}', horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)
        plt.savefig(f"test/test-plots-verzerrung/{i}.png")
        print(f"test/test-plots-verzerrung/{i}.png")
        plt.show()

        if accuracy < 0.7:
            print(f'Genauigkeit unter 70% mit distortion factor: {distortion_factor}. Test wird gestoppt.')
            break
        i += 1
        distortion_factor += step

test_distortion_robustness(model, test_dataloader, device)

# In[13]:



