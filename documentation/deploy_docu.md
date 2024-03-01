## Inhaltsverzeichnis für das Skript deploy.py 

- [Python-Klasse: `MainDeploy`](#python-klasse-maindeploy)
  - [Attribute](#attribute)
  - [Methoden](#methoden)
    - [`extract_model_name`](#extract_model_name)
    - [`deploy`](#deploy)
- [Python-Klasse: `GenderRecognitionPredictor`](#python-klasse-genderrecognitionpredictor)
  - [Methoden](#methoden-1)
    - [`__init__`](#1)
    - [`predict`](#predict)



# Python-Klasse: `MainDeploy`

Diese Klasse ist verantwortlich für die Bereitstellung der Gender Recognition CI/CD-Pipeline. Sie erbt von der `GenderRecognitionPredictor`-Klasse und enthält Methoden zum Hochladen von Bildern, zur Auswahl eines Modells und zur Durchführung von Vorhersagen.

## Attribute

- `model_dir` (`str`): Der Verzeichnispfad, in dem die Modelle gespeichert sind.
- `models` (`list`): Eine Liste der Modellnamen, die im `model_dir` gefunden wurden.

## Methoden

### `extract_model_name`

Diese Methode extrahiert den Modellnamen aus dem Dateinamen.

#### Parameter

- `filename` (`str`): Der Dateiname des Modells.

#### Rückgabewert

- `str`: Der extrahierte Modellname.
- `None`: Wenn kein Modellname gefunden wurde.

### `deploy`

Diese Methode startet die Gender Recognition CI/CD-Pipeline. Sie zeigt ein Upload-Feld für Bilder an, ermöglicht die Auswahl eines Modells und führt die Vorhersage für das hochgeladene Bild mit dem ausgewählten Modell durch.

#### Ablauf

1. Zeigt einen Titel und ein Upload-Feld für Bilder an.
2. Für jede hochgeladene Datei öffnet sie das Bild und zeigt es an.
3. Zeigt eine Auswahlbox mit den verfügbaren Modellen an.
4. Extrahiert den Modellnamen aus dem ausgewählten Modell.
5. Wenn der Benutzer auf den Vorhersage-Button klickt, führt sie eine Vorhersage mit dem ausgewählten Modell durch und zeigt das Ergebnis an.


# Python-Klasse: `GenderRecognitionPredictor`

Diese Klasse ist verantwortlich für das Laden und Vorhersagen von Geschlechtserkennungsmodellen.

## Methoden

### `__init__` {#1}

Diese Methode initialisiert eine neue Instanz der `GenderRecognitionPredictor`-Klasse. Sie lädt die verfügbaren Modelle aus dem angegebenen Verzeichnis.

### `predict`

Diese Methode führt eine Vorhersage für ein gegebenes Bild mit einem bestimmten Modell durch.

#### Parameter

- `image` (`PIL.Image`): Das Eingangsbild, für das die Vorhersage gemacht werden soll.
- `model_path` (`str`): Der Pfad zum gespeicherten Modell.
- `model_name` (`str`): Der Name des Modells.

#### Rückgabewert

Ein Tupel bestehend aus der vorhergesagten Klasse (`int`) und den Wahrscheinlichkeiten (`numpy.ndarray`).

#### Ablauf

1. Definiert die Transformation, die auf das Bild angewendet wird.
2. Konvertiert das Bild in RGB und ändert seine Größe.
3. Wendet die Transformation an und fügt eine zusätzliche Dimension hinzu.
4. Lädt das Modell basierend auf dem Modellnamen.
5. Lädt den gespeicherten Zustand des Modells und setzt das Modell in den Evaluierungsmodus.
6. Macht eine Vorhersage mit dem Modell und berechnet die Wahrscheinlichkeiten.
7. Gibt die vorhergesagte Klasse und die Wahrscheinlichkeiten zurück.

