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
        self.model_dir = "model/PyTorch_Trained_Models"
        self.models = os.listdir(self.model_dir)

    def predict(image, model_path, model_name):
        # Define the transformation
        transform = transforms.Compose(
            [
                transforms.Resize((178, 218)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        
        image = image.convert("RGB")
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
    model_dir = "model/PyTorch_Trained_Models/"
    models = os.listdir(model_dir)

    def extract_model_name(filename):
        match = re.match(r"(\w+)_model.*\.pth", filename)
        if match:
            return match.group(1)
        else:
            return None


    def deploy():
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
            st.write("Bild erfolgreich hochgeladen.")

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
