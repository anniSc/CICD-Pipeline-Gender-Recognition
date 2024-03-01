# GenderRecognition CI/CD-Pipline

Dieses Projekt ist eine CI/CD-Pipline für ein GenderRecognition-Modell, das das Geschlecht einer Person anhand eines Gesichtsbildes erkennen kann. Das Modell wurde mit dem Datensatz CelebA trainiert, der über 200.000 Bilder von Prominenten mit verschiedenen Attributen enthält, darunter auch das Geschlecht. Das Modell ist ein Convolutional Neural Network (CNN), das mit Tensorflow implementiert wurde. Die Pipline verwendet GithubActions, um das Modell automatisch zu testen, zu bauen und zu deployen.

## Anforderungen
Um dieses Projekt selbst verwenden zu können, benötigst du folgende Bibliotheken:
- fairlearn
- matplotlib
- numpy
- openpyxl
- pandas
- psutil
- pydqc
- pytest
- radon
- scikit-image
- scikit-learn
- scipy
- seaborn
- tqdm
- torch
- torchvision
- torchcam

Des Weiteren benötigst du:
- einen Github Account
- Github Personal Access Token
- GitHub Pro falls deine Tests die 2000 minuten Github Actions Serverzeit überschreiten
- ggf. einen eignen Testserver um Github Actions ohne zeitliche Limitierung verwenden zu können

## Installation
Um dieses Projekt zu installieren, folge diesen Schritten:
- Klone dieses Repository auf deinen lokalen Rechner oder forke es auf Github.
- Erstelle eine virtuelle Umgebung mit Python 3.8 oder höher und aktiviere sie.
- Installiere die erforderlichen Pakete mit dem Befehl `pip install -r requirements.txt`.
- Erstelle eine Workflowdatei z. B. `.github/workflows/cicd.yml` im Projektverzeichnis un

# Der Workflow

Der Workflow wird in der folgenden Grafik kompakt dargestellt, um einen Überblick über die relevanten Schritte zu erhalten. 

![Workflow Diagram](bpmn/Workflow.png?raw=true "Workflow")




# Klassendiagramme der Python-Skripte

# UML-Datenaufbereitungsskript

Im folgenden Bild wird das Datenaufbereitungsskript aufgezeigt mit allen Funktionen, Klassen und Vererbungen. Unter folgenden Link hast du die Möglichkeit die komplette Dokumentation zu diesem Skript hier anzuschauen: [Datenaufbereitungsskript](documentation/dataprep_docu.md)

![plot](./uml_diagrams/classes_dataprep_uml.png?raw=true "Datenaufbereitungsskript")


# UML-Trainingsskript

Im folgenden Bild wird das Trainingsskript aufgezeigt mit allen Funktionen, Klassen und Vererbungen. Unter folgenden Link hast du die Möglichkeit die komplette Dokumentation zu diesem Skript hier anzuschauen: [Trainingsskript](documentation/model_train_docu.md)

![plot](./uml_diagrams/classes_model_train_uml.jpg?raw=true "Trainingsskript")

# UML-Testskript

Im folgenden Bild wird das Trainingsskript aufgezeigt mit allen Funktionen, Klassen und Vererbungen. Unter folgenden Link hast du die Möglichkeit die komplette Dokumentation zu diesem Skript hier anzuschauen: [Testskript](documentation/model_test_docu.md)

![plot](./uml_diagrams/classes_model_test_uml.jpg?raw=true "Testskript")

# UML-Deployskript

Im folgenden Bild wird das Deployskript aufgezeigt mit allen Funktionen und Vererbungen. Unter folgenden Link hast du die Möglichkeit die komplette Dokumentation zu diesem Skript hier anzuschauen: [Deployskript](documentation/deploy_docu.md)

![plot](./uml_diagrams/classes_deploy_uml.png?raw=true "Deployskript")
