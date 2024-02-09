import os
import sys

import streamlit as st
import torch
from model_train import SimpleCNN as SCNN
from PIL import Image
from torchvision import transforms

sys.path.insert(0, "model/model_script/")


class GenderRecognitionPredictor:
    def __init__(self):
        self.model_dir = "model/PyTorch_Trained_Models"
        self.models = os.listdir(self.model_dir)

    def predict(image, model_path):
        # Define the transformation
        transform = transforms.Compose(
            [
                transforms.Resize((178, 218)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        image = image.convert("RGB")
        # Apply the transformation and add an extra dimension
        image = transform(image)
        image = image.unsqueeze(0)

        # Load the model
        model = SCNN()
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

            if st.button("Vorhersage Starten!"):
                prediction, probabilities = GenderRecognitionPredictor.predict(
                    image, model_path
                )
                st.write(f"Prediction: {prediction}")
                st.write(
                    f"Wahrscheinlichkeit das auf dem Bilde ein Mann ist: {probabilities[0]*100}%"
                )
                st.write(
                    f"Wahrscheinlichkeit das auf dem Bilde eine Frau ist: {probabilities[1]*100}%"
                )


MainDeploy.deploy()
