import torch
from torchvision import transforms
import streamlit as st
from PIL import Image
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


class GenderRecognitionPredictor:
    def __init__(self):
        self.model_dir = 'model/PyTorch_Trained_Models'
        self.models = os.listdir(self.model_dir)

    def predict(self, image, model_path):
        # Define the transformation
        transform = transforms.Compose([
            transforms.Resize((178, 218)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        image = image.convert("RGB")
        # Apply the transformation and add an extra dimension
        image = transform(image)
        image = image.unsqueeze(0)

        # Load the model
        model = SimpleCNN()
        model.load_state_dict(torch.load(model_path))
        model.eval()

        # Make the prediction
        with torch.no_grad():
            outputs = model(image)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            _, predicted = torch.max(outputs.data, 1)

        return predicted.item(), probabilities.numpy()




class MainDeploy(GenderRecognitionPredictor):
    def deploy(self):  
        st.title("Gender Recognition CI/CD Pipeline")

        uploaded_files = st.file_uploader("Bilder hochladen...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file)
            st.image(image, caption='Hochgeladenes Bild.', use_column_width=True)
            st.write("Bild erfolgreich hochgeladen.")

            # Use the list of models as options for the selectbox
            model_name = st.selectbox("Wählen Sie ein Modell aus:", GenderRecognitionPredictor.models)
            model_path = os.path.join(GenderRecognitionPredictor.model_dir, model_name)

            if st.button('Prediction Starten!'):
                prediction, probabilities = GenderRecognitionPredictor.predict(image, model_path)
                st.write(f"Prediction: {prediction}")
                st.write(f"Wahrscheinlichkeit das auf dem Bilde ein Mann ist: {probabilities[0]*100}%")
                st.write(f"Wahrscheinlichkeit das auf dem Bilde eine Frau ist: {probabilities[1]*100}%")

