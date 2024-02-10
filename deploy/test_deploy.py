import unittest
from streamlit.testing.v1 import AppTest

from PIL import Image
import os
import sys
sys.path.insert(0, "deploy/deploy.py")
sys.path.insert(0, "model/model_script/model_train.py")
sys.path.append("model/model_script/model_train.py")
sys.path.append("deploy/deploy.py")
from deploy import MainDeploy, GenderRecognitionPredictor
from model_train import SimpleCNN as SCNN
import streamlit as st  





class TestMainDeploy(unittest.TestCase):

    """
    Diese Klasse enthält Unittests für die MainDeploy-Klasse.
    """
    def _init_(self):
        at = AppTest.from_file("deploy/deploy.py")
   
        at.selectbox[0].select("default.txt").run()
        at.button[0].click().run()
        assert at.warning[0].value == "Try again!"
        at.run()
        assert   at.exception
       
    # def test_deploy(self):
    #     """
    #     Testen Sie die Bereitstellungsmethode der GenderRecognitionDeploy-Klasse.

    #     Diese Methode simuliert die Bereitstellung des Gender Klassifikations-Modells, indem sie die erforderlichen Abhängigkeiten
    #     nachahmt und verschiedene Aspekte des Bereitstellungsprozesses testet.

    #     In diesem Test durchgeführte Schritte:
    #     1. Testen des Datei-Uploaders
    #     2. Testen der Bildanzeige
    #     3. Testen der Modellauswahl
    #     4. Testen der Vorhersage
    #     5. Ausführen der Bereitstellungsmethode

    #     Nach dem Ausführen der Bereitstellungsmethode können Assertions hinzugefügt werden, um das erwartete Verhalten zu überprüfen.

    #     """
    #     container = st.container()
    #     # Testen des Datei-Uploaders
    #     uploaded_files = [Image.open("deploy/test.jpg")]
    #     container.file_uploader = lambda label, type, accept_multiple_files: uploaded_files

    #     # Test image display
    #     container.image = lambda image, caption, use_column_width: None

    #     # Test model selection
    #     container.selectbox = lambda label, options: options[0]
    #     # self.deploy.path.join = lambda *args: os.path.join(*args)


    #     # # Test prediction
    #     # container.button = lambda label: True
    #     # deploy.GenderRecognitionPredictor.predict = lambda image, model_path: ("Male", [0.8, 0.2])

    #     # # Run the deploy method
    #     # self.deploy.deploy()

    #     # Add assertions here

if __name__ == '__main__':
    unittest.main()