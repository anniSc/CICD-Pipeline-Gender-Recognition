import os
import sys
from model_train import SimpleCNN,SimpleCNN2,SimpleCNN3
import streamlit as st
import torch
from model_train import SimpleCNN as SCNN
from PIL import Image
from torchvision import transforms
import re



class GenderRecognitionPredictor:
    def __init__(self):
        """
        Initialisiert eine neue Instanz der Deploy-Klasse.
        Die Methode lädt die verfügbaren Modelle aus dem angegebenen Verzeichnis.
        """
        self.model_dir = "model/PyTorch_Trained_Models"
        self.models = os.listdir(self.model_dir)

    def predict(image, model_path, model_name):
            """
            Führt eine Vorhersage für ein gegebenes Bild mit einem bestimmten Modell durch.

            Args:
                image (PIL.Image): Das Eingangsbild, für das die Vorhersage gemacht werden soll.
                model_path (str): Der Pfad zum gespeicherten Modell.
                model_name (str): Der Name des Modells.

            Returns:
                tuple: Ein Tupel bestehend aus der vorhergesagten Klasse (int) und den Wahrscheinlichkeiten (numpy.ndarray).
            """
            # Define the transformation
            transform = transforms.Compose(
                [
                    transforms.Resize((178, 218)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
          
            image = image.convert("RGB")
            # st.write(image.size)
            image = image.resize((178, 218))
            # Apply the transformation and add an extra dimension
            image = transform(image)
            image = image.unsqueeze(0)
          
            # Load the model
            if model_name == "SimpleCNN": 
                model = SimpleCNN()
            elif model_name == "SimpleCNN2":
                model = SimpleCNN2()
            elif model_name == "SimpleCNN3":
                model = SimpleCNN3()

            model.load_state_dict(torch.load(model_path))
            model.eval()

            # Make the prediction
            with torch.no_grad():
                outputs = model(image)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                _, predicted = torch.max(outputs.data, 1)

            return predicted.item(), probabilities.numpy()


class MainDeploy(GenderRecognitionPredictor):
    """
    Hauptklasse für die Bereitstellung des Gender Recognition CI/CD-Pipelines.

    Diese Klasse erbt von der GenderRecognitionPredictor-Klasse und enthält Methoden zum Hochladen von Bildern,
    zur Auswahl eines Modells und zur Durchführung von Vorhersagen.

    Attributes:
        model_dir (str): Der Verzeichnispfad, in dem die Modelle gespeichert sind.
        models (list): Eine Liste der Modellnamen, die im model_dir gefunden wurden.
    """

    model_dir = "model/PyTorch_Trained_Models/"
    models = os.listdir(model_dir)

    def extract_model_name(filename):
        """
        Extrahiert den Modellnamen aus dem Dateinamen.

        Args:
            filename (str): Der Dateiname des Modells.

        Returns:
            str: Der extrahierte Modellname.
            None: Wenn kein Modellname gefunden wurde.
        """
        match = re.match(r"(\w+)_model.*\.pth", filename)
        if match:
            return match.group(1)
        else:
            return None


    def deploy():
        """
        Startet die Gender Recognition CI/CD-Pipeline.

        Diese Methode zeigt ein Upload-Feld für Bilder an, ermöglicht die Auswahl eines Modells
        und führt die Vorhersage für das hochgeladene Bild mit dem ausgewählten Modell durch.
        """
        st.title("Gender Recognition CI/CD Pipeline")

        uploaded_files = st.file_uploader(
            "Bilder hochladen...",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
        )

        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file)
            st.image(
                image,
                caption="Hochgeladenes Bild.",
                use_column_width=True)
            st.write('<style>img { max-width: 50%; height: auto; }</style>', unsafe_allow_html=True)

            # Use the list of models as options for the selectbox
            model_name = st.selectbox(
                "Wählen Sie ein Modell aus:",
                MainDeploy.models)
            model_path = os.path.join(MainDeploy.model_dir, model_name)
            
            model_name = MainDeploy.extract_model_name(model_name)  

            if st.button("Vorhersage Starten!"):
                prediction, probabilities = GenderRecognitionPredictor.predict(
                    image, model_path, model_name= model_name
                )
                st.write(f"Prediction: {prediction}")
                st.write(
                    f"Wahrscheinlichkeit das auf dem Bilde ein Mann ist: {probabilities[0]*100}%"
                )
                st.write(
                    f"Wahrscheinlichkeit das auf dem Bilde eine Frau ist: {probabilities[1]*100}%"
                )


MainDeploy.deploy()
