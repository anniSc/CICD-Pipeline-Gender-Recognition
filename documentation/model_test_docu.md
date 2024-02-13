# Klasse ModelTester

Die `ModelTester` Klasse stellt Methoden zum Testen und Evaluieren eines Modells bereit.

## Methode: test_model_robustness

Die Methode `test_model_robustness` testet die Robustheit des Modells, indem es auf den Testdaten Vorhersagen trifft und Metriken berechnet.

### Parameter

- `model` (torch.nn.Module): Das trainierte Modell.
- `test_dataloader` (torch.utils.data.DataLoader): Der DataLoader für die Testdaten.
- `device` (torch.device): Das Gerät (CPU oder GPU), auf dem die Berechnungen durchgeführt werden sollen.

### Rückgabewert

Ein Tupel mit den berechneten Metriken (Genauigkeit, Präzision, Recall, F1-Score).

### Beispiel

```python
model = ...  # Ein trainiertes Modell
test_dataloader = ...  # DataLoader für Testdaten
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Gerät

tester = ModelTester()
accuracy, precision, recall, f1 = tester.test_model_robustness(model, test_dataloader, device)

print(f"Genauigkeit: {accuracy}")
print(f"Präzision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")
``` 

# Methode: evaluate_model

Die Methode `evaluate_model` bewertet das Modell anhand des Testdatensatzes und gibt die Metriken zurück.

## Parameter

- `model` (torch.nn.Module): Das trainierte Modell.
- `test_dataloader` (torch.utils.data.DataLoader): Der DataLoader für den Testdatensatz.

## Rückgabewert

- `float`: Die Genauigkeit des Modells.
- `float`: Die Präzision des Modells.
- `float`: Der Recall des Modells.
- `float`: Der F1-Score des Modells.

## Beispiel

```python
model = ...  # Ein trainiertes Modell
test_dataloader = ...  # DataLoader für Testdaten

accuracy, precision, recall, f1 = evaluate_model(model, test_dataloader)

print(f"Genauigkeit: {accuracy}")
print(f"Präzision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")
``` 

# Methode: test_predicts

Die Methode `test_predicts` führt Vorhersagen für das gegebene Modell auf dem Testdatensatz durch.

## Parameter

- `model` (torch.nn.Module): Das Modell, das für die Vorhersagen verwendet werden soll.
- `test_dataloader` (torch.utils.data.DataLoader): Der DataLoader, der den Testdatensatz enthält.
- `device` (torch.device): Das Gerät (GPU oder CPU), auf dem die Vorhersagen durchgeführt werden sollen.

## Rückgabewert

Die Methode gibt nichts zurück (`None`). Sie druckt jedoch die Vorhersagen des Modells auf der Konsole aus.

## Beispiel

```python
model = ...  # Ein trainiertes Modell
test_dataloader = ...  # DataLoader für Testdaten
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Gerät

test_predicts(model, test_dataloader, device)
``` 

# Methode: test_noise_robustness

Die Methode `test_noise_robustness` führt einen Test der Rauschrobustheit des Modells durch.

## Parameter

- `model` (torch.nn.Module): Das zu testende Modell.
- `test_dataloader` (torch.utils.data.DataLoader): Der DataLoader für den Testdatensatz.
- `device` (torch.device): Das Gerät, auf dem der Test ausgeführt werden soll.
- `start_noise` (float, optional): Der Startwert für den Rauschfaktor. Standardmäßig 0.0.
- `end_noise` (float, optional): Der Endwert für den Rauschfaktor. Standardmäßig 1.0.
- `step` (float, optional): Der Schrittweite für den Rauschfaktor. Standardmäßig 0.1.
- `savefig_path` (str, optional): Der Pfad zum Speichern der generierten Plots. Standardmäßig "test/test-plots-rauschen".

## Rückgabewert

Die Methode gibt nichts zurück (`None`). Sie führt jedoch einen Test der Rauschrobustheit des Modells durch und speichert die generierten Plots.

## Beispiel

```python
model = ...  # Ein trainiertes Modell
test_dataloader = ...  # DataLoader für Testdaten
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Gerät

test_noise_robustness(model, test_dataloader, device)
``` 
# Methode: add_distortion

Die Methode `add_distortion` fügt eine Verzerrung zum Bild hinzu.

## Parameter

- `image` (torch.Tensor): Das Eingangsbild.
- `distortion_factor` (float, optional): Der Verzerrungsfaktor. Standardwert ist 0.5.

## Rückgabewert

- `torch.Tensor`: Das verzerrte Bild.

## Beispiel

```python
image = ...  # Ein Bild als Tensor
distorted_image = add_distortion(image, distortion_factor=0.7)
``` 
# Methode: test_distortion_robustness

Die Methode `test_distortion_robustness` testet die Robustheit des Modells gegenüber Verzerrungen.

## Parameter

- `model` (torch.nn.Module): Das zu testende Modell.
- `test_dataloader` (torch.utils.data.DataLoader): Der DataLoader für den Testdatensatz.
- `device` (torch.device): Das Gerät, auf dem das Modell ausgeführt wird.
- `start_distortion` (float, optional): Der Startwert für den Verzerrungsfaktor. Standardmäßig 0.0.
- `end_distortion` (float, optional): Der Endwert für den Verzerrungsfaktor. Standardmäßig 1.0.
- `step` (float, optional): Der Schrittweite für den Verzerrungsfaktor. Standardmäßig 0.0001.
- `savefig_path` (str, optional): Der Pfad zum Speichern der generierten Plots. Standardmäßig "test/test-plots-verzerrung".

## Rückgabewert

- `None`

## Beschreibung

Die Methode führt einen Test durch, um die Robustheit des Modells gegenüber Verzerrungen zu bewerten. Sie verändert den Verzerrungsfaktor schrittweise von `start_distortion` bis `end_distortion` und testet das Modell mit jedem Schritt. Die Testergebnisse (Genauigkeit, Präzision, Recall und F1-Score) werden zusammen mit einem Beispielbild für jede Verzerrungsstufe ausgegeben und gespeichert. Der Test wird abgebrochen, wenn die Genauigkeit unter 70% fällt.

# Methode: rotate_and_convert

Die Methode `rotate_and_convert` dreht das Bild um den angegebenen Winkel und konvertiert es in ein anderes Format.

## Parameter

- `image` (PIL.Image): Das Bild, das gedreht und konvertiert werden soll.
- `angle` (float): Der Winkel, um den das Bild gedreht werden soll.
- `test_images` (str, optional): Der Pfad zum Ordner mit den Testbildern. Standardmäßig "data/train-test-data/test".

## Rückgabewert

- `None`

## Beschreibung

Die Methode durchläuft alle Bilder im angegebenen Ordner. Für jedes Bild, das die Endung ".jpg" oder ".png" hat, wird es um den angegebenen Winkel gedreht. Die gedrehten Bilder werden im gleichen Format gespeichert.

## Beispiel

```python
from PIL import Image

image = Image.open('path_to_image.jpg')
rotate_and_convert(image, 45, 'path_to_test_images')
``` 

# Methode: test_model_robustness_rotation

Die Methode `test_model_robustness_rotation` testet die Robustheit des Modells gegenüber Rotationen.

## Parameter

- `model` (torch.nn.Module): Das zu testende Modell.
- `test_dataloader` (torch.utils.data.DataLoader): Der DataLoader für den Testdatensatz.
- `device` (torch.device): Das Gerät, auf dem die Berechnungen durchgeführt werden sollen.

## Rückgabewert

- `tuple`: Ein Tupel mit den Metriken Accuracy, Precision, Recall und F1-Score.

## Beschreibung

Die Methode führt einen Test durch, um die Robustheit des Modells gegenüber Rotationen zu bewerten. Sie setzt das Modell in den Evaluationsmodus und deaktiviert die Berechnung der Gradienten, um die Berechnung zu beschleunigen. Für jedes Bild im Testdatensatz berechnet sie die Ausgabe des Modells, wendet eine Softmax-Funktion an, um Wahrscheinlichkeiten zu erhalten, und wählt die Klasse mit der höchsten Wahrscheinlichkeit als Vorhersage. Die wahren und vorhergesagten Labels werden gesammelt und zur Berechnung der Metriken (Accuracy, Precision, Recall und F1-Score) verwendet.

## Beispiel

```python
model = ...  # Ein trainiertes Modell
test_dataloader = ...  # Ein DataLoader für den Testdatensatz
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

accuracy, precision, recall, f1 = test_model_robustness_rotation(model, test_dataloader, device)
``` 
# Methode: rotate_tensor_image

Die Methode `rotate_tensor_image` rotiert ein Bildtensor um den angegebenen Winkel.

## Parameter

- `image_tensor` (Tensor): Der Bildtensor, der rotiert werden soll.
- `angle` (float): Der Rotationswinkel in Grad.

## Rückgabewert

- `Tensor`: Der rotierte Bildtensor.

## Beschreibung

Die Methode nimmt einen Bildtensor und einen Winkel als Eingabe und gibt den rotierten Bildtensor zurück. Die Rotation wird mit der Funktion `TF.rotate` aus dem `torchvision.transforms.functional` Modul durchgeführt.

## Beispiel

```python
image_tensor = ...  # Ein Bildtensor
angle = 45  # Der Rotationswinkel in Grad

rotated_image = rotate_tensor_image(image_tensor, angle)
``` 
# Methode: test_rotation_robustness

Die Methode `test_rotation_robustness` testet die Robustheit des Modells gegenüber Rotationen.

## Parameter

- `model` (nn.Module): Das zu testende Modell.
- `test_dataloader` (DataLoader): Der DataLoader für den Testdatensatz.
- `device` (str): Das Gerät, auf dem die Berechnungen durchgeführt werden sollen.
- `start_angle` (float, optional): Der Startwinkel für die Rotation. Standardmäßig 0.0.
- `end_angle` (float, optional): Der Endwinkel für die Rotation. Standardmäßig 270.0.
- `step` (float, optional): Der Schritt für die Rotation. Standardmäßig 90.0.
- `savefig_path` (str, optional): Der Pfad zum Speichern der generierten Plots. Standardmäßig "test/test-plots-rotation".

## Rückgabewert

- `None`

## Beschreibung

Die Methode führt einen Test durch, um die Robustheit des Modells gegenüber Rotationen zu bewerten. Sie rotiert die Bilder im Testdatensatz um verschiedene Winkel und testet das Modell auf den rotierten Bildern. Die Ergebnisse des Tests (Genauigkeit, Präzision, Recall und F1-Score) werden für jeden Winkel berechnet und in einem Plot dargestellt. Der Test wird abgebrochen, wenn die Genauigkeit unter 70% fällt.

## Beispiel

```python
model = ...  # Ein trainiertes Modell
test_dataloader = ...  # Ein DataLoader für den Testdatensatz
device = 'cuda' if torch.cuda.is_available() else 'cpu'

test_rotation_robustness(model, test_dataloader, device)
``` 
# Klasse: TestFairness

Die Klasse `TestFairness` stellt Methoden zur Durchführung von Fairness-Tests für ein Modell bereit.

## Attribute

- `train_men` (str): Der Pfad zum Ordner mit den Trainingsdaten für Männer.
- `train_women` (str): Der Pfad zum Ordner mit den Trainingsdaten für Frauen.
- `sensitive_features` (list): Eine Liste der sensiblen Merkmale.
- `train` (str): Der Pfad zum Trainingsdatensatz.
- `test` (str): Der Pfad zum Testdatensatz.
- `merged_csv` (str): Der Pfad zur kombinierten CSV-Datei.
- `transform` (transforms.Compose): Eine Sequenz von Transformationen, die auf die Bilder angewendet werden.

## Methoden

- `get_sensitive_features(merged_csv, label_men=1, label_women=-1)`: Gibt die sensiblen Merkmale zurück.
- `get_fairness_metrics(merged_csv, train_dataloader, model, transform, sensitive_features)`: Berechnet die Fairness-Metriken.
- `create_gender_labelled_csv(men_folder, women_folder, output_csv, label_men=1, label_women=-1)`: Erstellt eine CSV-Datei mit den gelabelten Daten.
- `plot_bar_fairnesscheck(groups, accuracies, metrics)`: Erstellt ein Balkendiagramm der Fairness-Metriken.
- `analyze_metrics(sensitive_features, y_test, y_pred)`: Analysiert die Fairness-Metriken.
- `clear_file(file_path)`: Löscht den Inhalt einer Datei.
- `run_fairness_tests(train_dataloader, model, transform)`: Führt die Fairness-Tests aus.

## Beschreibung

Die Klasse `TestFairness` ist dafür konzipiert, Fairness-Tests auf einem Modell durchzuführen. Sie bietet Methoden zum Erhalten sensibler Merkmale, Berechnen von Fairness-Metriken, Erstellen von gelabelten CSV-Dateien, Plotten von Fairness-Checks, Analysieren von Metriken und Durchführen von Fairness-Tests.

# Methode: get_sensitive_features

Die Methode `get_sensitive_features` gibt die sensiblen Merkmale zurück.

## Parameter

- `merged_csv` (str): Der Pfad zur kombinierten CSV-Datei.
- `label_men` (int, optional): Der Label-Wert für Männer. Standardmäßig 1.
- `label_women` (int, optional): Der Label-Wert für Frauen. Standardmäßig -1.

## Rückgabewert

- Eine Liste der sensiblen Merkmale (list).

## Beschreibung

Die Methode liest eine kombinierte CSV-Datei ein, die die sensiblen Merkmale enthält. Sie konvertiert die Labels für Männer und Frauen in die entsprechenden Strings ("Mann" und "Frau") und gibt eine Liste dieser sensiblen Merkmale zurück.

## Beispiel

```python
merged_csv = "test/csv/gender_labelled.csv"
sensitive_features = get_sensitive_features(merged_csv)
```
# Methode: get_fairness_metrics

Die Methode `get_fairness_metrics` berechnet die Fairness-Metriken.

## Parameter

- `merged_csv` (str): Der Pfad zur kombinierten CSV-Datei.
- `train_dataloader` (DataLoader): Der Trainings-Dataloader.
- `model` (nn.Module): Das Modell.
- `transform` (torchvision.transforms.Compose): Die Transformationen für die Eingabedaten.
- `sensitive_features` (list): Die sensiblen Merkmale.

## Rückgabewert

- Ein Tupel, das die berechneten Metriken, die wahren Labels und die vorhergesagten Labels enthält (tuple).

## Beschreibung

Die Methode `get_fairness_metrics` berechnet die Fairness-Metriken für ein gegebenes Modell. Sie führt Vorhersagen auf dem Trainingsdatensatz durch und berechnet dann die Genauigkeit der Vorhersagen, aufgeteilt nach den sensiblen Merkmalen. Die berechneten Metriken, die wahren Labels und die vorhergesagten Labels werden zurückgegeben.

## Beispiel

```python
merged_csv = "test/csv/gender_labelled.csv"
train_dataloader = ...  # Ein DataLoader für den Trainingsdatensatz
model = ...  # Ein trainiertes Modell
transform = ...  # Die Transformationen für die Eingabedaten
sensitive_features = ["men", "women"]

metrics, y_test, y_pred = get_fairness_metrics(merged_csv, train_dataloader, model, transform, sensitive_features)
``` 

# Methode: create_gender_labelled_csv

Die Methode `create_gender_labelled_csv` erstellt eine CSV-Datei mit den gelabelten Daten.

## Parameter

- `men_folder` (str): Der Pfad zum Ordner mit den Trainingsdaten für Männer.
- `women_folder` (str): Der Pfad zum Ordner mit den Trainingsdaten für Frauen.
- `output_csv` (str): Der Pfad zur Ausgabedatei.
- `label_men` (int, optional): Der Label-Wert für Männer. Standardmäßig 1.
- `label_women` (int, optional): Der Label-Wert für Frauen. Standardmäßig -1.

## Rückgabewert

- Der absolute Pfad zur Ausgabedatei (str).

## Beschreibung

Die Methode `create_gender_labelled_csv` erstellt eine CSV-Datei mit den gelabelten Daten. Sie liest die Dateien aus den angegebenen Ordnern für Männer und Frauen ein, erstellt für jede Gruppe ein DataFrame mit den Dateinamen und den entsprechenden Labels und fügt diese dann zusammen. Das kombinierte DataFrame wird dann als CSV-Datei gespeichert. Der absolute Pfad zur Ausgabedatei wird zurückgegeben.

## Beispiel

```python
men_folder = "data/train/men"
women_folder = "data/train/women"
output_csv = "data/train/gender_labelled.csv"

output_path = create_gender_labelled_csv(men_folder, women_folder, output_csv)
```
# Methode: plot_bar_fairnesscheck

Die Methode `plot_bar_fairnesscheck` erstellt ein Balkendiagramm der Fairness-Metriken.

## Parameter

- `groups` (list): Die Gruppen.
- `accuracies` (list): Die Genauigkeiten.
- `metrics` (MetricFrame): Die Metriken.

## Beschreibung

Die Methode `plot_bar_fairnesscheck` erstellt ein Balkendiagramm, das die Genauigkeit der Vorhersagen für jede Gruppe darstellt. Die Gruppen und Genauigkeiten werden aus dem übergebenen `MetricFrame` extrahiert. Das Diagramm wird mit Titel, Achsenbeschriftungen und einer festen Auflösung von 100 dpi erstellt und als JPEG-Datei gespeichert. Nach dem Speichern des Diagramms wird der Plotbereich mit `plt.clf()` gelöscht, um sicherzustellen, dass zukünftige Plots nicht auf demselben Diagramm dargestellt werden.

## Beispiel

```python
groups = ["men", "women"]
accuracies = [0.8, 0.7]
metrics = ...  # Ein MetricFrame-Objekt

plot_bar_fairnesscheck(groups, accuracies, metrics)
```
# Methode: analyze_metrics

Die Methode `analyze_metrics` analysiert die Fairness-Metriken.

## Parameter

- `sensitive_features` (list): Die sensiblen Merkmale.
- `y_test` (list): Die wahren Labels.
- `y_pred` (list): Die vorhergesagten Labels.

## Rückgabewert

- Das Metriken-Framework (MetricFrame).

## Beschreibung

Die Methode `analyze_metrics` analysiert die Fairness-Metriken für die gegebenen wahren und vorhergesagten Labels, aufgeteilt nach den sensiblen Merkmalen. Sie berechnet eine Reihe von Metriken, darunter Genauigkeit, Präzision, Falsch-Positiv-Rate, Falsch-Negativ-Rate, Auswahlrate und Anzahl. Diese Metriken werden dann in einem `MetricFrame` zusammengefasst.

Die Methode erstellt auch ein Balkendiagramm für jede Metrik, aufgeteilt nach den sensiblen Merkmalen. Das Diagramm wird als JPEG-Datei gespeichert.

## Beispiel

```python
sensitive_features = ["men", "women"]
y_test = [1, -1, 1, 1, -1, -1]
y_pred = [1, 1, -1, 1, -1, -1]

metric_frame = analyze_metrics(sensitive_features, y_test, y_pred)
```
# Methode: clear_file

Die Methode `clear_file` löscht den Inhalt einer Datei.

## Parameter

- `file_path` (str): Der Pfad zur Datei, deren Inhalt gelöscht werden soll.

## Beschreibung

Die Methode `clear_file` öffnet die angegebene Datei im Schreibmodus und überschreibt ihren Inhalt mit einem leeren String. Dadurch wird der gesamte Inhalt der Datei gelöscht.

## Beispiel

```python
file_path = "data/test.txt"

clear_file(file_path)
```

# Methode: run_fairness_tests

Die Methode `run_fairness_tests` führt die Fairness-Tests aus.

## Parameter

- `train_dataloader` (DataLoader): Der Trainings-Dataloader.
- `model` (nn.Module): Das Modell.
- `transform` (torchvision.transforms.Compose): Die Transformationen für die Eingabedaten.

## Beschreibung

Die Methode `run_fairness_tests` führt eine Reihe von Fairness-Tests auf einem gegebenen Modell aus. Sie erstellt zunächst eine CSV-Datei mit Geschlechtslabels, berechnet dann die Fairness-Metriken und speichert die Genauigkeit für jede Gruppe in einer Textdatei. Anschließend erstellt sie ein Balkendiagramm der Fairness-Metriken und analysiert die Metriken. Schließlich erstellt sie zwei Diagramme: ein Balkendiagramm aller Metriken mit zugewiesenem y-Achsenbereich und ein Kuchendiagramm der Metriken. Beide Diagramme werden als JPEG-Dateien gespeichert.

## Beispiel

```python
train_dataloader = ...  # Ein DataLoader-Objekt
model = ...  # Ein nn.Module-Objekt
transform = ...  # Ein torchvision.transforms.Compose-Objekt

run_fairness_tests(train_dataloader, model, transform)
```
# Klasse: ModelExplainability

Die Klasse `ModelExplainability` dient zur Erklärung von Modellvorhersagen.

## Methoden

### `__init__(self, model_path, target_layer, model)`

Initialisiert das `ModelExplainability`-Objekt.

#### Parameter

- `model_path` (str): Der Pfad zum Modell.
- `target_layer` (str): Die Ziel-Ebene des Modells. Dies ist die Ebene, auf der die Grad-CAM++-Visualisierung basiert.
- `model` (Model): Das Modellobjekt.

#### Beschreibung

Der Konstruktor der Klasse `ModelExplainability` initialisiert das Modellobjekt und erstellt einen Grad-CAM++-Extraktor für das gegebene Modell und die Ziel-Ebene. Grad-CAM++ ist eine Methode zur Visualisierung der Wichtigkeit jeder Position in der Ziel-Ebene für die Vorhersage des Modells.

## Beispiel

```python
model_path = "path/to/model"
target_layer = "layer_name"
model = ...  # Ein Modellobjekt

model_explainability = ModelExplainability(model_path, target_layer, model)
```

# Methoden: get_image_paths, select_images, process_image

## Methode: get_image_paths

Die Methode `get_image_paths` gibt eine Liste der Bildpfade im angegebenen Verzeichnis zurück.

### Parameter

- `dir_path` (str): Der Pfad zum Verzeichnis.

### Rückgabewert

- Eine Liste der Bildpfade (list[str]).

## Methode: select_images

Die Methode `select_images` wählt zufällig ein Bild aus dem Männerverzeichnis und ein Bild aus dem Frauenverzeichnis aus.

### Parameter

- `men_dir` (str): Der Pfad zum Männerverzeichnis.
- `women_dir` (str): Der Pfad zum Frauenverzeichnis.

### Rückgabewert

- Eine Liste der ausgewählten Bildpfade (list[str]).

## Methode: process_image

Die Methode `process_image` verarbeitet das Bild anhand des angegebenen Bildpfads.

### Parameter

- `img_path` (str): Der Pfad zum Bild.

### Rückgabewert

- Das verarbeitete Bild (torch.Tensor). Das Bild wird gelesen, auf die Größe (178, 218) skaliert, normalisiert und dann zurückgegeben.

## Beispiel

```python
men_dir = "path/to/men_images"
women_dir = "path/to/women_images"
img_path = "path/to/image.jpg"

image_paths = get_image_paths(men_dir)
selected_images = select_images(men_dir, women_dir)
processed_image = process_image(img_path)
```
# Methode: visualize_model_grad

Die Methode `visualize_model_grad` visualisiert das Modell anhand ausgewählter Bilder aus den Männer- und Frauenverzeichnissen.

## Parameter

- `men_dir` (str): Der Pfad zum Männerverzeichnis. Standardmäßig ist dies "data/train-test-data/train/men".
- `women_dir` (str): Der Pfad zum Frauenverzeichnis. Standardmäßig ist dies "data/train-test-data/train/women".

## Beschreibung

Die Methode `visualize_model_grad` visualisiert die Gradienten des Modells für ausgewählte Bilder. Sie wählt zufällig ein Bild aus dem Männerverzeichnis und ein Bild aus dem Frauenverzeichnis aus, verarbeitet diese Bilder und führt sie durch das Modell. Dann extrahiert sie die Aktivierungskarte aus der Ausgabe des Modells und visualisiert diese Aktivierungskarte. Die Aktivierungskarte zeigt, welche Bereiche des Bildes das Modell für seine Vorhersage verwendet hat. Die Aktivierungskarten werden als PNG-Dateien im Verzeichnis "test/activation_map" gespeichert.

## Beispiel

```python
men_dir = "path/to/men_images"
women_dir = "path/to/women_images"

visualize_model_grad(men_dir, women_dir)
```

# Klasse: Main_Model_Test

Die Klasse `Main_Model_Test` erbt von den Klassen `ModelTester`, `TestFairness` und `ModelExplainability`.

## Beschreibung

Die Klasse `Main_Model_Test` ist eine Testklasse, die verschiedene Tests auf einem Modell ausführt. Sie erbt Methoden von den Klassen `ModelTester`, `TestFairness` und `ModelExplainability`, die jeweils Methoden zum Testen der Modellleistung, zur Überprüfung der Fairness des Modells und zur Erklärung der Modellvorhersagen bereitstellen.

Die Klasse `Main_Model_Test` kann verwendet werden, um ein Modell auf verschiedene Aspekte zu testen, einschließlich seiner Leistung auf einem Testdatensatz, seiner Robustheit gegenüber verschiedenen Arten von Störungen, seiner Fairness in Bezug auf verschiedene Gruppen und seiner Erklärbarkeit in Bezug auf die Vorhersagen, die es macht.

## Verwendung

Um die Klasse `Main_Model_Test` zu verwenden, erstellen Sie eine Instanz der Klasse und rufen Sie die gewünschten Testmethoden auf. Die spezifischen Methoden, die aufgerufen werden, hängen von den spezifischen Tests ab, die Sie auf Ihrem Modell ausführen möchten.


# Funktion: extract_model_name

Die Funktion `extract_model_name` extrahiert den Modellnamen aus dem angegebenen vollständigen Pfad.

## Parameter

- `full_path` (str): Der vollständige Pfad, aus dem der Modellname extrahiert werden soll.

## Rückgabewert

- `str`: Der extrahierte Modellname.

## Beschreibung

Die Funktion `extract_model_name` nimmt einen vollständigen Pfad als Eingabe und extrahiert den Modellnamen aus diesem Pfad. Der Modellname wird angenommen, um der Teil des Dateinamens vor dem ersten Unterstrich zu sein. Der Dateiname wird aus dem vollständigen Pfad extrahiert und dann am Unterstrich aufgeteilt. Der erste Teil der Aufteilung wird als Modellname zurückgegeben.

## Beispiel

```python
full_path = "path/to/model_name_other_info.pth"
model_name = extract_model_name(full_path)
```


# Funktion: run_tests

Die Funktion `run_tests` führt verschiedene Tests auf dem Modell aus.

## Parameter

- Keine

## Rückgabewert

- Keine

## Beschreibung

Die Funktion `run_tests` lädt ein Modell, führt verschiedene Tests auf dem Modell durch und visualisiert die Ergebnisse. Es werden Tests zur Modellbewertung, Robustheit, Fairness und Erklärbarkeit durchgeführt.

Die Funktion lädt das Modell aus dem angegebenen Pfad, extrahiert den Modellnamen und lädt das entsprechende Modell. Sie liest die Batch-Größe aus einer Datei und lädt die Trainings- und Testdaten. Anschließend führt sie verschiedene Tests auf dem Modell aus, einschließlich der Bewertung des Modells, der Überprüfung der Robustheit des Modells gegenüber Rauschen, Verzerrungen und Rotationen, der Durchführung von Fairness-Tests und der Visualisierung der Modellgradienten.

## Beispiel

```python
run_tests()
```

