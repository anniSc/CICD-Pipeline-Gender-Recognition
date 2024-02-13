# Klassen: SimpleCNN, SimpleCNN2 und SimpleCNN3

Diese Klassen definieren drei einfache Convolutional Neural Network (CNN) Modelle zur Klassifizierung von Bildern.

## SimpleCNN

### Attribute

- `conv1` (nn.Conv2d): Erste Faltungsoperationsschicht.
- `pool` (nn.MaxPool2d): Max-Pooling-Schicht.
- `conv2` (nn.Conv2d): Zweite Faltungsoperationsschicht.
- `fc1` (nn.Linear): Erste vollständig verbundene Schicht.
- `fc2` (nn.Linear): Zweite vollständig verbundene Schicht.
- `fc3` (nn.Linear): Dritte vollständig verbundene Schicht.

### Methoden

- `forward(x)`: Führt die Vorwärtsberechnung des Modells durch.

## SimpleCNN2

### Attribute

- `conv1` (nn.Conv2d): Erste Faltungsoperationsschicht.
- `pool` (nn.MaxPool2d): Max-Pooling-Schicht.
- `conv2` (nn.Conv2d): Zweite Faltungsoperationsschicht.
- `fc1` (nn.Linear): Erste vollständig verbundene Schicht.
- `fc2` (nn.Linear): Zweite vollständig verbundene Schicht.
- `fc3` (nn.Linear): Dritte vollständig verbundene Schicht.
- `dropout` (nn.Dropout): Dropout-Schicht zur Vermeidung von Overfitting.

### Methoden

- `forward(x)`: Führt die Vorwärtsberechnung des Modells durch.

## SimpleCNN3

### Attribute

- `conv1` (nn.Conv2d): Erste Faltungsoperationsschicht.
- `pool` (nn.MaxPool2d): Max-Pooling-Schicht.
- `conv2` (nn.Conv2d): Zweite Faltungsoperationsschicht.
- `fc1` (nn.Linear): Erste vollständig verbundene Schicht.
- `fc2` (nn.Linear): Zweite vollständig verbundene Schicht.
- `fc3` (nn.Linear): Dritte vollständig verbundene Schicht.
- `dropout` (nn.Dropout): Dropout-Schicht zur Vermeidung von Overfitting.
- `bn1` (nn.BatchNorm2d): Batch-Normalisierung für die erste Faltungsschicht.
- `bn2` (nn.BatchNorm2d): Batch-Normalisierung für die zweite Faltungsschicht.

### Methoden

- `forward(x)`: Führt die Vorwärtsberechnung des Modells durch.

# Klasse: Trainer

Diese Klasse verwaltet den Trainingsprozess für ein PyTorch-Modell.

## Attribute

- `model` (torch.nn.Module): Das zu trainierende PyTorch-Modell.
- `train_dataloader` (torch.utils.data.DataLoader): Der Datenlader für den Trainingsdatensatz.
- `test_dataloader` (torch.utils.data.DataLoader): Der Datenlader für den Test-/Validierungsdatensatz.
- `epochs` (int): Die Anzahl der Epochen, um das Modell zu trainieren.
- `batch_size` (int): Die Batch-Größe für das Training.
- `criterion` (torch.nn.Module): Die Verlustfunktion für das Training.
- `optimizer` (torch.optim.Optimizer): Der Optimierer zum Aktualisieren der Modellparameter.
- `patience` (int): Die Anzahl der Epochen, um auf eine Verbesserung der Validierungsgenauigkeit zu warten, bevor das Training abgebrochen wird.
- `best_accuracy` (float): Die beste Validierungsgenauigkeit, die während des Trainings erreicht wurde.
- `early_stopping_counter` (int): Der Zähler zur Verfolgung der Anzahl der Epochen ohne Verbesserung der Validierungsgenauigkeit.

## Methoden

- `__init__(model, train_dataloader, test_dataloader, epochs, optimizer, criterion, batch_size, patience=10, best_accuracy=0.0, early_stopping_counter=0)`: Initialisiert die Trainer-Klasse mit dem gegebenen Modell, Datenladern, Anzahl der Epochen, Optimierer, Verlustfunktion, Batch-Größe und optionalen Parametern für die Geduld, die beste Genauigkeit und den Zähler für das frühe Stoppen.

# Methoden: `train` und `plot_cpu_memory_usage`

## Methode: `train`

Diese Methode trainiert ein Modell für eine festgelegte Anzahl von Epochen. Sie führt die folgenden Schritte aus:

1. Initialisiert Listen für die CPU- und Speichernutzung sowie Zeitstempel.
2. Führt das Training für eine festgelegte Anzahl von Epochen durch.
3. In jeder Epoche werden die Daten durch das Modell geführt, der Verlust berechnet und die Gewichte des Modells aktualisiert.
4. Nach jeder Epoche wird die Genauigkeit des Modells auf den Test-/Validierungsdaten berechnet.
5. Die CPU- und Speichernutzung sowie der Zeitstempel werden nach jeder Epoche aufgezeichnet.
6. Wenn die Genauigkeit 90% oder 95% übersteigt, wird der aktuelle Zustand des Modells gespeichert.
7. Wenn die Genauigkeit nicht innerhalb einer festgelegten Anzahl von Epochen verbessert wird, wird das Training frühzeitig beendet ("early stopping").
8. Am Ende des Trainings wird der Zustand des Modells gespeichert und die CPU- und Speichernutzung über die Zeit geplottet.

### Parameter

Keine

### Gibt zurück

None

## Methode: `plot_cpu_memory_usage`

Diese Methode zeichnet die CPU- und Speichernutzung über die Zeit. Sie nimmt drei Parameter: eine Liste von CPU-Nutzungsprozenten, eine Liste von Speichernutzungsprozenten und eine Liste von Zeitstempeln, die den CPU- und Speichernutzungsprozenten entsprechen. Sie erstellt ein Diagramm mit der Zeit auf der x-Achse und der Nutzung in Prozent auf der y-Achse und speichert das Diagramm als Bilddatei.

### Parameter

- `cpu_percentages` (Liste): Eine Liste von CPU-Nutzungsprozenten. Jedes Element in der Liste repräsentiert die CPU-Nutzung zu einem bestimmten Zeitpunkt.
- `memory_percentages` (Liste): Eine Liste von Speichernutzungsprozenten. Jedes Element in der Liste repräsentiert die Speichernutzung zu einem bestimmten Zeitpunkt.
- `time_stamps` (Liste): Eine Liste von Zeitstempeln, die den CPU- und Speichernutzungsprozenten entsprechen. Jedes Element in der Liste repräsentiert den Zeitpunkt, zu dem die entsprechenden CPU- und Speichernutzungsprozentsätze aufgezeichnet wurden.

### Gibt zurück

None

# Klasse: `DataLoaderModelTrain`

Diese Klasse ist verantwortlich für das Laden von Trainings- und Testdaten und das Erstellen von DataLoader-Objekten.

## Methode: `__init__`

Diese Methode initialisiert die `DataLoaderModelTrain`-Klasse.

### Parameter

- `batch_size` (int): Die Batch-Größe für den DataLoader.
- `transform` (torchvision.transforms): Die Transformationen, die auf die Daten angewendet werden sollen.

### Gibt zurück

None

## Methode: `load_data`

Diese Methode lädt die Trainings- und Testdaten und erstellt DataLoader-Objekte.

### Parameter

- `test_dir` (str): Der Pfad zum Verzeichnis mit den Testdaten.
- `train_dir` (str): Der Pfad zum Verzeichnis mit den Trainingsdaten.
- `transform` (torchvision.transforms): Die Transformationen, die auf die Daten angewendet werden sollen.
- `batch_size` (int): Die Batch-Größe für den DataLoader.

### Gibt zurück

- `tuple`: Ein Tupel bestehend aus den Trainings- und Test-DataLoader-Objekten.

# Klasse: `DataLoaderModelTrain`

Diese Klasse ist verantwortlich für das Laden von Trainings- und Testdaten und das Erstellen von DataLoader-Objekten.

## Methode: `__init__`

Diese Methode initialisiert die `DataLoaderModelTrain`-Klasse.

### Parameter

- `batch_size` (int): Die Batch-Größe für den DataLoader.
- `transform` (torchvision.transforms): Die Transformationen, die auf die Daten angewendet werden sollen.

### Gibt zurück

None

## Methode: `load_data`

Diese Methode lädt die Trainings- und Testdaten und erstellt DataLoader-Objekte.

### Parameter

- `test_dir` (str): Der Pfad zum Verzeichnis mit den Testdaten.
- `train_dir` (str): Der Pfad zum Verzeichnis mit den Trainingsdaten.
- `transform` (torchvision.transforms): Die Transformationen, die auf die Daten angewendet werden sollen.
- `batch_size` (int): Die Batch-Größe für den DataLoader.

### Gibt zurück

- `tuple`: Ein Tupel bestehend aus den Trainings- und Test-DataLoader-Objekten.



# Klasse: `Main`

Diese Klasse erbt von `DataLoaderModelTrain` und ist die Hauptklasse zum Trainieren und Speichern eines PyTorch-Modells.

## Methode: `__init__`

Diese Methode initialisiert die `Main`-Klasse.

### Parameter

- `batch_size` (int): Die Batch-Größe für das Training und Testen.
- `epochs` (int): Die Anzahl der Epochen für das Training.
- `test_dir` (str): Der Verzeichnispfad für die Testdaten.
- `transform` (torchvision.transforms.Compose): Die Daten-Transformationspipeline.
- `train_dir` (str): Der Verzeichnispfad für die Trainingsdaten.
- `train_dataloader` (torch.utils.data.DataLoader): Der Datenlader für die Trainingsdaten.
- `test_dataloader` (torch.utils.data.DataLoader): Der Datenlader für die Testdaten.
- `model` (SimpleCNN): Das Modell für die Geschlechtererkennung.
- `model_save_path` (str): Der Verzeichnispfad zum Speichern des trainierten Modells.
- `model_test_path` (str): Der Verzeichnispfad für das zu testende Modell.

### Gibt zurück

None

## Methode: `clean_up_pth`

Diese Methode löscht .pth-Dateien im angegebenen Verzeichnis.

### Parameter

- `directory` (str): Der Verzeichnispfad.

### Gibt zurück

- `None` wenn das Verzeichnis nicht existiert und eine Datei namens "default.txt" gefunden wird.
- `"Modell gelöscht!"` wenn .pth-Dateien erfolgreich gelöscht wurden.

## Methode: `train_and_save`

Diese Methode trainiert das Modell und speichert das trainierte Modell.

### Parameter

- `model_name` (str): Der Name des Modells.
- `model_test_path` (str, optional): Der Verzeichnispfad zum Speichern des zu testenden Modells.

### Gibt zurück

None


# Python-Script: `model_train.py`

Dieses Skript initialisiert und trainiert ein `SimpleCNN3` Modell und speichert es dann.

## Hauptablauf

Wenn dieses Skript als Hauptprogramm ausgeführt wird, führt es die folgenden Schritte aus:

1. Initialisiert ein `SimpleCNN3` Modell und setzt einige Parameter wie `batch_size`, `epochs`, `test_dir`, `model_save_path`, `model_test_path`, `train_dir`, `optimizer`, `criterion`, `patience`, `best_accuracy`, `early_stopping_counter`.

2. Definiert eine Transformationspipeline, die die Größe der Bilder ändert, sie in Tensoren umwandelt und sie normalisiert.

3. Lädt die Trainings- und Testdaten mit der `load_data` Methode der `DataLoaderModelTrain` Klasse.

4. Initialisiert ein `Trainer` Objekt mit dem Modell, den Datenladern, den Epochen, der Batch-Größe, dem Optimierer, der Geduld, der besten Genauigkeit, dem Frühstoppzähler und dem Kriterium.

5. Initialisiert ein `Main` Objekt mit der Batch-Größe, den Epochen, dem Testverzeichnis, der Transformationspipeline, dem Trainingsverzeichnis, den Datenladern, dem Modell und den Pfaden zum Speichern des Modells.

6. Liest den Code des aktuellen Skripts und speichert ihn in den Verzeichnissen `deploy` und `test/model_test_scripts`.

7. Trainiert das Modell mit der `train` Methode des `Trainer` Objekts.

8. Trainiert das Modell und speichert es mit der `train_and_save` Methode des `Main` Objekts.