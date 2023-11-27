# GenderRecognition CI/CD-Pipline

Dieses Projekt ist eine CI/CD-Pipline für ein GenderRecognition-Modell, das das Geschlecht einer Person anhand eines Gesichtsbildes erkennen kann. Das Modell wurde mit dem Datensatz CelebA trainiert, der über 200.000 Bilder von Prominenten mit verschiedenen Attributen enthält, darunter auch das Geschlecht. Das Modell ist ein Convolutional Neural Network (CNN), das mit Tensorflow implementiert wurde. Die Pipline verwendet GithubActions, um das Modell automatisch zu testen, zu bauen und zu deployen.

## Anforderungen

Um dieses Projekt auszuführen, benötigen Sie:

- Python 3.8 oder höher
- Tensoflow 1.9.0 oder höher
- Keras am besten aktuelles Version
- matplotlib
- seaborn
- Numpy 1.21.0 oder höher
- PIL 8.2.0 oder höher
- Pandas am besten aktuellste Version
- Github Account
- Github Personal Access Token
- GitHub Pro falls Ihre Tests die 2000 minuten Github Actions Serverzeit überschreiten
- ggf. einen eignen Testserver um Github Actions ohne zeitliche Limitierung verwenden zu können

## Installation

Um dieses Projekt zu installieren, folgen Sie diesen Schritten:

- Klonen Sie dieses Repository auf Ihren lokalen Rechner oder forken Sie es auf Github.
- Erstellen Sie eine virtuelle Umgebung mit Python 3.8 oder höher und aktivieren Sie sie.
- Installieren Sie die erforderlichen Pakete mit dem Befehl `pip install -r requirements.txt`.
- Erstellen Sie eine Datei namens `.env` im Projektverzeichnis und fügen Sie Ihren Github Personal Access Token als Umgebungsvariable hinzu, z.B. `GITHUB_TOKEN=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`.
- Erstellen Sie eine Datei namens `.github/workflows/cicd.yml` im Projektverzeichnis und fügen Sie den folgenden Inhalt hinzu:

#Beispiel.yaml
Diese Datei ist nur ein Beispiel wie Ihr CI/CD Workflow aussehen könnte: 
```yaml
name: CI/CD Pipline

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.8
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
      - name: Run tests
        run: |
          python -m unittest discover tests

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.8
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
      - name: Download CelebA dataset
        run: |
          wget https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/celeba.zip
          unzip celeba.zip
      - name: Train model
        run: |
          python train.py
      - name: Save model
        run: |
          mkdir model
          mv gender_recognition.pth model/
      - name: Upload model artifact
        uses: actions/upload-artifact@v2
        with:
          name: model
          path: model

  deploy:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.8
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
      - name: Download model artifact
        uses: actions/download-artifact@v2
        with:
          name: model
      - name: Deploy model to Heroku
        uses: akhileshns/heroku-deploy@v3.12.12
        with:
          heroku_api_key: ${{secrets.HEROKU_API_KEY}}
          heroku_app_name: "gender-recognition-app"
          heroku_email: "your_email@example.com"
```

- Ersetzen Sie `your_email@example.com` mit Ihrer eigenen E-Mail-Adresse, die Sie für Heroku verwenden.
- Erstellen Sie einen Heroku Account, falls Sie noch keinen haben, und erstellen Sie eine neue App mit dem Namen `gender-recognition-app`.
- Erstellen Sie eine Datei namens `Procfile` im Projektverzeichnis und fügen Sie den folgenden Inhalt hinzu:

```
web: gunicorn app:app
```

- Erstellen Sie eine beliebige Datei z. B. `app.py` im Projektverzeichnis und fügen Sie den folgenden Inhalt hinzu:
```python
import tensorflow as tf
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np

app = Flask(__name__)

model = tf.keras.models.load_model("model/gender_recognition.h5")

def preprocess_image(image):
  image = image.resize((178, 218))
  image = np.array(image) / 255.0
  image = np.expand_dims(image, axis=0)
  return image

@app.route("/predict", methods=["POST"])
def predict():
  if request.files.get("image"):
    image = Image.open(request.files["image"])
    image = preprocess_image(image)
    output = model.predict(image)
    gender = "Male" if output[0][0] < 0.5 else "Female"
    return jsonify({"gender": gender})
  else:
    return jsonify({"error": "No image provided"})
```

- Führen Sie einen Push oder einen Pull Request auf den main Branch aus, um die Pipline auszulösen.
- Überprüfen Sie den Status der Pipline auf GithubActions und warten Sie, bis alle Jobs erfolgreich abgeschlossen sind.
- Besuchen Sie die URL Ihrer Heroku App, z.B. https://gender-recognition-app.herokuapp.com/, um das Modell zu testen. Sie können eine POST-Anfrage an die Route `/predict` mit einem Bild als Parameter senden und die Antwort als JSON erhalten, z.B. `{"gender": "Female"}`.

## Referenzen

: [CelebFaces Attributes Dataset (CelebA)]: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
