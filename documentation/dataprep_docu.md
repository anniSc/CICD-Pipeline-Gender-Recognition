
# Inhaltsverzeichnis für die Dokumentation des Datenaufbereitungsskriptes

---
- [Dokumentation Main Klasse](#dokumentation-für-die-main-klasse)
    - [Funktion init](#funktion-init)
    - [collect_usage(self)](#funktion-collect_usage)
    - [get_usage_collection()](#funktion-get_usage_collection)
    - [plot_usage(self)](#funktion-plot_usage)
---
- [Dokumentation DataBalancing Klasse](#dokumentation-für-die-databalancing-klasse)
    - [balance_column(csv_path, column_name)](#funktion-balance_columncsv_path-column_name)
---
- [Dokumentation Datatest Klasse](#dokumentation-für-die-datatest-klasse)
    - [run_datatest()](#funktion-run_datatest())
    - [check_data_completeness(csv1, csv2)](#funktion-check_data_completenesscsv1-csv2)
    - [is_numeric(column)](#funktion-is_numericcolumn)
    - [test_image_extensions(directory)](#funktion-test_image_extensionsdirectory)
    - [check_csv_extension(csv_path)](#funktion-check_csv_extensioncsv_path)
    - [test_image_extensions(directory)](#funktion-test_image_extensionsdirectory)
    - [is_numeric(column)](#funktion-is_numericcolumn)
    - [check_required_directories_data_exists(directories)`](#funktion-check_required_directories_data_existsdirectories)
    - [test_quality_of_csv(csv_path, column_name_of_image_paths="image_id")](#funktion-test_quality_of_csvcsv_path-column_name_of_image_pathsimage_id)
    - [test_outliers_zscore](#funktion-test_outliers_zscore)
    - [test_balance_all_columns](#funktion-test_balance_all_columns)
    - [test_outliers_IQR](#funktion-test_outliers_iqr)
    - [detect_outliers](#funktion-detect_outliers)
    - [detect_anomaly](#funktion-detect_anomaly)
    - [test_normal_distribution](#funktion-test_normal_distribution)
    - [test_uniform_distribution](#funktion-test_uniform_distribution)
    - [test_binomial_distribution](#funktion-test_binomial_distribution)
    - [test_exponential_distribution](#funktion-test_exponential_distribution)
    - [test_image_brightness](#funktion-test_image_brightness)
    ---
- [Dokumentation DataVisualization Klasse](#klasse-datavisualization)
    - [run_datavis](#funktion-run_datavis)
    - [histogram_all_columns](#funktion-histogram_all_columns)
    - [plot_histogram](#funktion-plot_histogram)


# Dokumentation für die `Main` Klasse

Die `Main` Klasse ist die Hauptklasse, die die verschiedenen Funktionen zur Datenverarbeitung, Datenprüfung, Datenbalancierung und Datenvisualisierung enthält. Sie erbt von den Klassen `DataPreparation`, `DataTest`, `DataBalancing` und `DataVisualization`.

## Funktionen der Klasse 'Main' 

### Funktion init

Die `__init__` Methode initialisiert eine Instanz der Klasse `Main`. Sie setzt die Attribute der Klasse, einschließlich der Anzahl der Gesamtbilder, der Pfade zu den ausbalancierten CSV-Dateien, der Spaltennamen für das Alter, der Pfade zu den Ausgabedateien für die Verteilungsberichte, der Listen für den Speicher- und CPU-Verbrauch, der Zeitstempel und der Thread-Steuerungsvariablen.

### `collect_usage`

Die `collect_usage` Methode sammelt die CPU- und Speicherauslastung in regelmäßigen Abständen. Diese Methode läuft in einer Schleife, bis der stop_thread-Flag gesetzt ist. In jedem Schleifendurchlauf werden die CPU-Auslastung, die Speicherauslastung und der Zeitstempel erfasst und in den entsprechenden Listen gespeichert.

### `run_all`

Die `run_all` Methode führt alle Schritte der Datenverarbeitung aus, einschließlich der Ausführung von DataTest, DataVisualization und DataPreparation.

### `get_usage_collection`

Die `get_usage_collection` Methode gibt die CPU-Auslastung, den Speicherverbrauch und die Zeitstempel zurück.

### `plot_usage`

Die `plot_usage` Methode plottet die CPU- und Speichernutzung während der Datenvorbereitung.
### `run_all`

Die `run_all` Methode führt alle Schritte der Datenverarbeitung aus, einschließlich der Ausführung von `DataTest`, `DataVisualization` und `DataPreparation`.

**Parameter**: Keine

**Rückgabewert**: Keine

Die Methode führt die folgenden Schritte aus:

1. Startet den Thread, der die CPU- und Speicherauslastung sammelt.
2. Speichert die aktuelle Zeit in `start_time`.
3. Führt `DataTest.run_datatest` aus, um verschiedene statistische Tests auf den Daten durchzuführen. Die Pfade zu den Ausgabedateien für die Testergebnisse werden als Argumente übergeben.
4. Führt `DataVisualization.run_datavis` aus, um die Daten zu visualisieren. Die Pfade zu den ausbalancierten CSV-Dateien, der Spaltenname für das Alter und die Feature-Spalte werden als Argumente übergeben.
5. Führt `DataPreparation.run_dataprep` aus, um die Daten vorzubereiten. Die Gesamtzahl der Bilder wird als Argument übergeben.
6. Setzt `stop_thread` auf `True`, um den Thread, der die CPU- und Speicherauslastung sammelt, zu stoppen.
7. Wartet auf das Ende des Threads mit `self.thread.join()`.
8. Ruft `self.plot_usage()` auf, um die CPU- und Speicherauslastung zu plotten.

```python
def run_all(self):
    self.thread.start()
    start_time = time.time()
    DataTest.run_datatest(
        self.save_binomial_distribution_path_txt,
        self.save_uniform_distribution_path_txt,
        self.save_exponential_distribution_path_txt,
        self.save_norm_distribution_path_txt,
    )
    DataVisualization.run_datavis(
        balanced_gender_path=self.balanced_gender_path,
        balanced_young_path=self.balanced_young_path,
        column_name=self.young_column,
        feature_column=DataPreparation.feature_column,
    )
    DataPreparation.run_dataprep(total_images=self.total_images)
    self.stop_thread = True
    self.thread.join()
    self.plot_usage()
```

### init

Die `__init__` Methode initialisiert eine Instanz der Klasse `DataPreparation`.

**Parameter**: Keine

**Rückgabewert**: Keine

Diese Methode initialisiert die Attribute der Klasse, einschließlich:

- `total_images`: Die Anzahl der Gesamtbilder. Standardmäßig auf 10 gesetzt.
- `balanced_gender_path`: Der Pfad zur ausbalancierten CSV-Datei für das Geschlecht.
- `balanced_young_path`: Der Pfad zur ausbalancierten CSV-Datei für das Alter.
- `young_column`: Der Spaltenname für das Alter in der CSV-Datei.
- `save_norm_distribution_path_txt`: Der Pfad zur Ausgabedatei für den Normalverteilungsbericht.
- `save_binomial_distribution_path_txt`: Der Pfad zur Ausgabedatei für den Binomialverteilungsbericht.
- `save_uniform_distribution_path_txt`: Der Pfad zur Ausgabedatei für den Gleichverteilungsbericht.
- `save_exponential_distribution_path_txt`: Der Pfad zur Ausgabedatei für den Exponentialverteilungsbericht.
- `memory_usage`: Eine Liste zur Speicherung der Speicherauslastung während der Datenvorbereitung.
- `cpu_usage`: Eine Liste zur Speicherung der CPU-Auslastung während der Datenvorbereitung.
- `timestamps`: Eine Liste zur Speicherung der Zeitstempel während der Datenvorbereitung.
- `stop_thread`: Ein Flag zur Steuerung des Threads, der die CPU- und Speicherauslastung sammelt.
- `thread`: Ein Thread, der die Methode `collect_usage` ausführt, um die CPU- und Speicherauslastung zu sammeln.
- `time`: Eine Variable zur Speicherung der Gesamtlaufzeit der Datenvorbereitung.

```python
def __init__(self):
    self.total_images = 10
    self.balanced_gender_path = "data/balanced_source_csv/gender_balanced.csv"
    self.balanced_young_path = "data/balanced_source_csv/young_balanced.csv"
    self.young_column = "Young"
    self.save_norm_distribution_path_txt = "data/reports_data/norm_distribution.txt"
    self.save_binomial_distribution_path_txt = "data/reports_data/binomial_distribution.txt"
    self.save_uniform_distribution_path_txt = "data/reports_data/uniform_distribution.txt"
    self.save_exponential_distribution_path_txt = "data/reports_data/exponential_distribution.txt"
    self.memory_usage = []
    self.cpu_usage = []
    self.timestamps = []
    self.stop_thread = False
    self.thread = threading.Thread(target=self.collect_usage)
    self.time = 0
```

### Funktion `collect_usage`

Die `collect_usage` Methode sammelt die CPU- und Speicherauslastung in regelmäßigen Abständen.

**Parameter**: Keine

**Rückgabewert**: Keine

Diese Methode läuft in einer Schleife, bis der `stop_thread`-Flag gesetzt ist. In jedem Schleifendurchlauf werden die CPU-Auslastung, die Speicherauslastung und der Zeitstempel erfasst und in den entsprechenden Listen gespeichert. Die CPU-Auslastung wird mit der `psutil.cpu_percent(interval=1)` Funktion erfasst, die Speicherauslastung mit der `psutil.virtual_memory().percent` Funktion und der Zeitstempel mit der `time.time()` Funktion. Nachdem diese Werte erfasst wurden, wartet die Methode eine Sekunde (`time.sleep(1)`) bevor sie den nächsten Durchlauf startet. Am Ende der Methode werden die Listen mit den gesammelten Werten ausgegeben.

```python
def collect_usage(self):
    while not self.stop_thread:
        self.memory_usage.append(psutil.virtual_memory().percent)
        self.cpu_usage.append(psutil.cpu_percent(interval=1))
        self.timestamps.append(time.time())
        time.sleep(1)
    print(self.cpu_usage, self.memory_usage, self.timestamps)
```


### Funktion `run_all`

Die `run_all` Methode führt alle Schritte der Datenverarbeitung aus, einschließlich der Ausführung von `DataTest`, `DataVisualization` und `DataPreparation`.

**Parameter**: Keine

**Rückgabewert**: Keine

Die Methode führt die folgenden Schritte aus:

1. Startet den Thread, der die CPU- und Speicherauslastung sammelt.
2. Speichert die aktuelle Zeit in `start_time`.
3. Führt `DataTest.run_datatest` aus, um verschiedene statistische Tests auf den Daten durchzuführen. Die Pfade zu den Ausgabedateien für die Testergebnisse werden als Argumente übergeben.
4. Führt `DataVisualization.run_datavis` aus, um die Daten zu visualisieren. Die Pfade zu den ausbalancierten CSV-Dateien, der Spaltenname für das Alter und die Feature-Spalte werden als Argumente übergeben.
5. Führt `DataPreparation.run_dataprep` aus, um die Daten vorzubereiten. Die Gesamtzahl der Bilder wird als Argument übergeben.
6. Setzt `stop_thread` auf `True`, um den Thread, der die CPU- und Speicherauslastung sammelt, zu stoppen.
7. Wartet auf das Ende des Threads mit `self.thread.join()`.
8. Ruft `self.plot_usage()` auf, um die CPU- und Speicherauslastung zu plotten.

```python
def run_all(self):
    self.thread.start()
    start_time = time.time()
    DataTest.run_datatest(
        self.save_binomial_distribution_path_txt,
        self.save_uniform_distribution_path_txt,
        self.save_exponential_distribution_path_txt,
        self.save_norm_distribution_path_txt,
    )
    DataVisualization.run_datavis(
        balanced_gender_path=self.balanced_gender_path,
        balanced_young_path=self.balanced_young_path,
        column_name=self.young_column,
        feature_column=DataPreparation.feature_column,
    )
    DataPreparation.run_dataprep(total_images=self.total_images)
    self.stop_thread = True
    self.thread.join()
    self.plot_usage()
```

### Funktion `get_usage_collection`

Die `get_usage_collection` Methode gibt die CPU-Auslastung, den Speicherverbrauch und die Zeitstempel zurück.

**Parameter**: Keine

**Rückgabewert**: Ein Tupel bestehend aus der CPU-Auslastung, dem Speicherverbrauch und den Zeitstempeln.

Die Methode gibt ein Tupel zurück, das die Listen `cpu_usage`, `memory_usage` und `timestamps` enthält, die während der Ausführung der `collect_usage` Methode gefüllt wurden.

```python
def get_usage_collection(self):
    return self.cpu_usage, self.memory_usage, self.timestamps
```

### Funktion `plot_usage`

Die `plot_usage` Methode plottet die CPU- und Speichernutzung während der Datenvorbereitung.

**Parameter**: Keine

**Rückgabewert**: Keine

Die Methode führt die folgenden Schritte aus:

1. Ruft `self.get_usage_collection()` auf, um die CPU-Auslastung, den Speicherverbrauch und die Zeitstempel zu erhalten.
2. Setzt den Titel des Plots auf "CPU und Speichernutzung während der Datenvorbereitung".
3. Setzt die Beschriftungen der x- und y-Achsen auf "Zeit (s)" bzw. "Nutzung (%)".
4. Fügt eine Legende mit den Einträgen "CPU-Nutzung" und "Speichernutzung" hinzu.
5. Plottet die CPU-Auslastung gegen die Zeitstempel mit einer roten Linie und der Beschriftung "CPU Auslastung".
6. Plottet den Speicherverbrauch gegen die Zeitstempel mit einer blauen Linie und der Beschriftung "Speicher Nutzung".
7. Speichert den Plot als Bild unter dem Pfad "data/cpu_memory_usage_on_dataprep.png".
8. Zeigt den Plot an.

```python
def plot_usage(self):
    cpu_usage, memory_usage, timestamps = self.get_usage_collection()
    plt.title("CPU und Speichernutzung während der Datenvorbereitung")
    plt.xlabel("Zeit (s)")
    plt.ylabel("Nutzung (%)")
    plt.legend(["CPU-Nutzung", "Speichernutzung"])
    plt.plot(timestamps, cpu_usage, label="CPU Auslastung", color="red", linewidth=3)
    plt.plot(timestamps, memory_usage, label="Speicher Nutzung", color="blue",linewidth=3)
    plt.savefig("data/cpu_memory_usage_on_dataprep.png")
    plt.show()
```









# Dokumentation für die `DataBalancing` Klasse

Die `DataBalancing` Klasse bietet Methoden zum Ausgleichen von Daten in einer CSV-Datei.

## Methoden

### Funktion `balance_column(csv_path, column_name)`

Diese Methode gleicht die Daten in der angegebenen Spalte der CSV-Datei aus.

#### Parameter

- `csv_path` (str): Der Pfad zur CSV-Datei.
- `column_name` (str): Der Name der auszugleichenden Spalte.

#### Rückgabewert

- `df_balanced` (pandas.DataFrame): Der ausgeglichene DataFrame.

#### Beschreibung

Die Methode liest die Daten aus der CSV-Datei in einen DataFrame ein und zählt die Anzahl der Werte in der angegebenen Spalte. Sie bestimmt die minimale Anzahl von -1 und 1 Werten in der Spalte und erstellt einen neuen DataFrame, der eine gleiche Anzahl von -1 und 1 Werten enthält, indem sie zufällige Zeilen aus dem ursprünglichen DataFrame auswählt. Der ausgeglichene DataFrame wird zurückgegeben.

# Dokumentation für die `DataTest` Klasse


Die `DataTest` Klasse ist verantwortlich für die Durchführung verschiedener Tests und Überprüfungen auf den Daten, die für die Datenanalyse vorbereitet wurden.

## Methoden

### Funktion `run_datatest()`

Diese Methode führt eine Reihe von Tests und Überprüfungen auf den vorbereiteten Daten durch. Sie nimmt vier Parameter entgegen, die Pfade zu den Textdateien sind, in denen die Ergebnisse der Verteilungstests gespeichert werden sollen.

Die Methode führt die folgenden Aktionen aus:

1. Erstellt die benötigten Verzeichnisse mit `DataPreparation.create_directories()`.
2. Extrahiert die IDs aus den Quelldaten und speichert sie mit `DataPreparation.extract_ids_source_data_and_save()`.
3. Extrahiert alle IDs mit `DataPreparation.extract_all_ids()`.

Die Ergebnisse der Verteilungstests werden in den angegebenen Textdateien gespeichert.

## Klassenvariablen

- `save_norm_distribution_path_txt`: Pfad zur Textdatei, in der die Ergebnisse des Normalverteilungstests gespeichert werden.
- `save_binomial_distribution_path_txt`: Pfad zur Textdatei, in der die Ergebnisse des Binomialverteilungstests gespeichert werden.
- `save_uniform_distribution_path_txt`: Pfad zur Textdatei, in der die Ergebnisse des Gleichverteilungstests gespeichert werden.
- `save_exponential_distribution_path_txt`: Pfad zur Textdatei, in der die Ergebnisse des Exponentialverteilungstests gespeichert werden.

---

### Funktionen der Klasse `DataTest`

### Funktion `check_data_completeness(csv1, csv2)`

Diese Methode überprüft die Vollständigkeit der Daten in zwei CSV-Dateien.

#### Parameter

- `csv1` (str): Der Pfad zur ersten CSV-Datei.
- `csv2` (str): Der Pfad zur zweiten CSV-Datei.

#### Rückgabewert

- `is_equal` (bool): Ein Boolean-Wert, der angibt, ob die Daten in beiden Dateien vollständig sind.

#### Beschreibung

Die Methode liest die Daten aus beiden CSV-Dateien in separate DataFrames ein und vergleicht die Einträge in der ersten Spalte beider DataFrames. Sie identifiziert die Einträge, die in der einen, aber nicht in der anderen Datei vorhanden sind, und gibt Warnungen aus, wenn solche Einträge gefunden werden. Wenn keine fehlenden Einträge gefunden werden, gibt sie eine Bestätigung aus, dass die Daten vollständig sind. Andernfalls gibt sie eine Warnung aus und wirft eine AssertionError-Ausnahme. Der Boolean-Wert, der angibt, ob die Daten vollständig sind, wird zurückgegeben.

---
### Funktion `is_numeric(column)`

Diese Methode überprüft, ob eine Spalte numerische Werte enthält.

#### Parameter

- `column` (pandas.Series): Die zu überprüfende Spalte.

#### Rückgabewert

- `is_numeric` (bool): Ein Boolean-Wert, der angibt, ob die Spalte numerische Werte enthält.

#### Beschreibung

Die Methode versucht, die Werte in der Spalte in numerische Werte umzuwandeln. Wenn dies erfolgreich ist, gibt sie True zurück. Wenn ein ValueError auftritt, gibt sie False zurück.

---

### Funktion `test_image_extensions(directory)`

Diese Methode überprüft die Dateierweiterungen aller Dateien in einem Verzeichnis und stellt sicher, dass sie gültige Bild-Dateierweiterungen haben.

#### Parameter

- `directory` (str): Der Pfad zum Verzeichnis, das die zu überprüfenden Dateien enthält.

#### Beschreibung

Die Methode listet alle Dateien in dem angegebenen Verzeichnis auf und überprüft ihre Dateierweiterungen. Sie erstellt eine Liste der Dateien, deren Erweiterungen nicht in der Liste der gültigen Bild-Dateierweiterungen enthalten sind. Wenn solche Dateien gefunden werden, gibt sie eine Warnung aus und wirft eine AssertionError-Ausnahme.

---

### Funktion `test_image_extensions(directory)`

Diese Methode überprüft die Dateierweiterungen aller Dateien in einem Verzeichnis und stellt sicher, dass sie gültige Bild-Dateierweiterungen haben.

#### Parameter

- `directory` (str): Der Pfad zum Verzeichnis, das die zu überprüfenden Dateien enthält.

#### Beschreibung

Die Methode listet alle Dateien in dem angegebenen Verzeichnis auf und überprüft ihre Dateierweiterungen. Sie erstellt eine Liste der Dateien, deren Erweiterungen nicht in der Liste der gültigen Bild-Dateierweiterungen enthalten sind. Wenn solche Dateien gefunden werden, gibt sie eine Warnung aus und wirft eine AssertionError-Ausnahme.

---

### Funktion `check_csv_extension(csv_path)`

Diese Methode überprüft die Dateierweiterung einer Datei und stellt sicher, dass sie eine .csv-Erweiterung hat.

#### Parameter

- `csv_path` (str): Der Pfad zur zu überprüfenden Datei.

#### Beschreibung

Die Methode extrahiert die Dateierweiterung der angegebenen Datei und überprüft, ob sie .csv ist. Wenn dies nicht der Fall ist, wirft sie eine AssertionError-Ausnahme.

---

### Funktion `is_numeric(column)`

Diese Methode überprüft, ob eine Spalte numerische Werte enthält.

#### Parameter

- `column` (pandas.Series): Die zu überprüfende Spalte.

#### Rückgabewert

- `is_numeric` (bool): Ein Boolean-Wert, der angibt, ob die Spalte numerische Werte enthält.

#### Beschreibung

Die Methode versucht, die Werte in der Spalte in numerische Werte umzuwandeln. Wenn dies erfolgreich ist, gibt sie True zurück. Wenn ein ValueError auftritt, gibt sie False zurück.

---
### Funktion `check_required_directories_data_exists(directories)`

Diese Methode überprüft, ob die angegebenen Verzeichnisse existieren.

#### Parameter

- `directories` (list): Eine Liste von Pfaden zu den zu überprüfenden Verzeichnissen.

#### Beschreibung

Die Methode durchläuft jedes Verzeichnis in der Liste und überprüft, ob es existiert. Wenn ein Verzeichnis nicht existiert, wirft sie eine AssertionError-Ausnahme.

---

### Funktion `test_quality_of_csv(csv_path, column_name_of_image_paths="image_id")`

Diese Methode überprüft die Qualität der Daten in einer CSV-Datei.

#### Parameter

- `csv_path` (str): Der Pfad zur CSV-Datei.
- `column_name_of_image_paths` (str, optional): Der Name der Spalte, die die Bildpfade enthält. Standardmäßig ist dies "image_id".

#### Beschreibung

Die Methode liest die Daten aus der CSV-Datei in einen DataFrame ein und führt dann zwei Überprüfungen durch: Sie überprüft, ob es fehlende Werte in der Spalte gibt, die die Bildpfade enthält, und ob es Duplikate in den Daten gibt. Wenn eine dieser Überprüfungen fehlschlägt, wirft sie eine AssertionError-Ausnahme.

---

### Funktion `test_outliers_zscore`

Diese Methode überprüft auf Ausreißer in den numerischen Spalten einer CSV-Datei.

#### Parameter

- `csv_path` (str): Der Pfad zur CSV-Datei.

#### Beschreibung

Die Methode liest die Daten aus der CSV-Datei in einen DataFrame ein und durchläuft dann jede Spalte. Wenn eine Spalte numerische Daten enthält, berechnet sie die Z-Scores der Werte in dieser Spalte und überprüft, ob es Werte gibt, die mehr als drei Standardabweichungen vom Mittelwert entfernt sind. Wenn solche Werte gefunden werden, gibt sie eine Warnung aus.

---

## Funktion `test_balance_all_columns`

Die Methode `test_balance_all_columns(csv_path)` ist eine statische Methode, die das Gleichgewicht aller Spalten in einer CSV-Datei überprüft.

## Parameter

- `csv_path` (str): Der Pfad zur CSV-Datei.

## Rückgabewert

- Kein Rückgabewert.

## Beschreibung

Die Methode liest eine CSV-Datei vom angegebenen Pfad und überprüft das Gleichgewicht aller Spalten. Wenn der absolute Unterschied zwischen den Zählungen von -1 und 1 in einer numerischen Spalte größer oder gleich 10% der Gesamtzahl der Zeilen ist, fügt sie eine Nachricht zum Ungleichgewichtsbericht hinzu. Wenn es unausgeglichene Spalten gibt, druckt sie eine Nachricht, die besagt: "Es gibt unausgeglichene Spalten", gefolgt vom Ungleichgewichtsbericht.

## Beispiel

Angenommen, wir haben eine CSV-Datei mit dem Pfad "data.csv", die unausgeglichene Spalten enthält. Die Methode kann wie folgt aufgerufen werden:

```python
DataTest.test_balance_all_columns("data.csv")
```

Wenn es unausgeglichene Spalten gibt, wird eine Ausgabe ähnlich der folgenden erzeugt:

```
Es gibt unausgeglichene Spalten:
Die Spalte 'column1' ist unausgeglichen. Anzahl von -1: 100, Anzahl von 1: 200
Die Spalte 'column2' ist unausgeglichen. Anzahl von -1: 150, Anzahl von 1: 300
```

Dies bedeutet, dass in den Spalten 'column1' und 'column2' ein Ungleichgewicht zwischen den Zählungen von -1 und 1 besteht.

---

## Funktion `test_outliers_IQR`

Die Methode `test_outliers_IQR(df)` ist eine statische Methode, die den Prozentsatz der Ausreißer für numerische Spalten eines DataFrame unter Verwendung des IQR-Verfahrens berechnet.

## Parameter

- `df` (pandas.DataFrame): Der DataFrame, für den die Ausreißer berechnet werden sollen.

## Rückgabewert

- `dict`: Ein Wörterbuch, das den Prozentsatz der Ausreißer für jede numerische Spalte enthält.

## Beschreibung

Die Methode durchläuft jede Spalte des DataFrame. Wenn die Spalte numerisch ist, berechnet sie das erste (Q1) und dritte Quartil (Q3) sowie den Interquartilbereich (IQR). Sie definiert dann die untere und obere Grenze für Ausreißer als Q1 - 1,5*IQR und Q3 + 1,5*IQR. Sie identifiziert Ausreißer als Werte, die kleiner als die untere Grenze oder größer als die obere Grenze sind. Der Prozentsatz der Ausreißer wird berechnet und zum Wörterbuch `outliers_percentage` hinzugefügt. Sie druckt auch den Prozentsatz der Ausreißer für jede Spalte.

## Beispiel

Angenommen, wir haben einen DataFrame `df`, der numerische Spalten enthält. Die Methode kann wie folgt aufgerufen werden:

```python
outliers_percentage = DataTest.test_outliers_IQR(df)
```

Die Ausgabe könnte so aussehen:

```
Ausreißerprozentwert für Spalte 'column1': 2.5%
Ausreißerprozentwert für Spalte 'column2': 1.0%
```

Und `outliers_percentage` wäre ein Wörterbuch, das so aussieht:

```python
{
    'column1': 2.5,
    'column2': 1.0
}
```

Dies bedeutet, dass 2,5% der Werte in 'column1' und 1,0% der Werte in 'column2' als Ausreißer identifiziert wurden.

---

## Funktion `detect_outliers`

Die Methode `detect_outliers(df, column_name)` ist eine statische Methode, die Ausreißer in einer gegebenen Spalte eines DataFrame erkennt.

## Parameter

- `df` (pandas.DataFrame): Der DataFrame, in dem die Ausreißer erkannt werden sollen.
- `column_name` (str): Der Name der Spalte, in der die Ausreißer erkannt werden sollen.

## Rückgabewert

- `pandas.DataFrame`: Ein DataFrame, der nur die Ausreißer enthält.

## Beschreibung

Die Methode berechnet das erste (Q1) und dritte Quartil (Q3) sowie den Interquartilbereich (IQR) der angegebenen Spalte. Sie definiert dann die untere und obere Grenze für Ausreißer als Q1 - 1,5*IQR und Q3 + 1,5*IQR. Sie identifiziert Ausreißer als Werte, die kleiner als die untere Grenze oder größer als die obere Grenze sind. Diese Ausreißer werden in einem neuen DataFrame zurückgegeben.

## Beispiel

Angenommen, wir haben einen DataFrame `df`, der eine Spalte 'column1' enthält. Die Methode kann wie folgt aufgerufen werden:

```python
outliers = DataTest.detect_outliers(df, 'column1')
```

`outliers` wäre dann ein DataFrame, der nur die Ausreißer in 'column1' enthält.

---

## Funktion `detect_anomaly`

Die Methode `detect_anomaly(csv_path, id_column)` ist eine statische Methode, die Anomalien in den Daten erkennt.

## Parameter

- `csv_path` (str): Der Pfad zur CSV-Datei mit den Daten.
- `id_column` (str): Der Name der Spalte, die die IDs enthält.

## Ausnahmen

- `ValueError`: Wenn Anomalien in den Daten gefunden werden.

## Beschreibung

Die Methode liest eine CSV-Datei vom angegebenen Pfad und entfernt die ID-Spalte. Sie verwendet dann den IsolationForest-Algorithmus aus der sklearn-Bibliothek, um Anomalien in den Daten zu erkennen. Wenn in den vorhergesagten Werten keine -1 oder 1 vorhanden ist, wird eine ValueError-Ausnahme ausgelöst, die besagt: "Anomalie gefunden!". Wenn keine Anomalien gefunden werden, druckt sie eine Nachricht, die besagt: "Keine Anomalien gefunden."

## Beispiel

Angenommen, wir haben eine CSV-Datei mit dem Pfad "data.csv", die eine ID-Spalte "id" enthält. Die Methode kann wie folgt aufgerufen werden:

```python
DataTest.detect_anomaly("data.csv", "id")
```

Wenn Anomalien gefunden werden, wird eine ValueError-Ausnahme ausgelöst. Wenn keine Anomalien gefunden werden, wird die Nachricht "Keine Anomalien gefunden." ausgegeben.

---

## Funktion `test_normal_distribution`

Die Methode `test_normal_distribution(data, save_distribution_path_txt)` ist eine statische Methode, die überprüft, ob die Daten in einem DataFrame einer Normalverteilung folgen.

## Parameter

- `data` (str): Der Pfad zur CSV-Datei, die die Daten enthält.
- `save_distribution_path_txt` (str, optional): Der Pfad zur Textdatei, in der die Ergebnisse gespeichert werden sollen. Standardmäßig "data/reports_data/norm_distribution.txt".

## Rückgabewert

- `None`

## Beschreibung

Die Methode liest eine CSV-Datei vom angegebenen Pfad und durchläuft jede Spalte. Wenn die Spalte numerisch ist, führt sie den Shapiro-Wilk-Test durch, um zu überprüfen, ob die Daten in der Spalte einer Normalverteilung folgen. Wenn der p-Wert größer als 0,05 ist, wird angenommen, dass die Daten wahrscheinlich einer Normalverteilung folgen. Andernfalls wird angenommen, dass die Daten wahrscheinlich nicht einer Normalverteilung folgen. Die Ergebnisse werden sowohl ausgegeben als auch in einer Textdatei am angegebenen Pfad gespeichert.

## Beispiel

Angenommen, wir haben eine CSV-Datei mit dem Pfad "data.csv". Die Methode kann wie folgt aufgerufen werden:

```python
DataTest.test_normal_distribution("data.csv")
```

Die Ausgabe könnte so aussehen:

```
Die Daten in der Spalte 'column1' folgen wahrscheinlich einer Normalverteilung.
Die Daten in der Spalte 'column2' folgen wahrscheinlich nicht einer Normalverteilung.
```

Und eine Textdatei mit dem Pfad "data/reports_data/norm_distribution.txt" wird erstellt, die die gleichen Ergebnisse enthält.

---

## Funktion `test_uniform_distribution`

Die Methode `test_uniform_distribution(data, save_distribution_path_txt)` ist eine statische Methode, die überprüft, ob die Daten in einem DataFrame einer gleichmäßigen Verteilung folgen.

## Parameter

- `data` (str): Der Pfad zur CSV-Datei, die die Daten enthält.
- `save_distribution_path_txt` (str, optional): Der Pfad zur Textdatei, in der die Ergebnisse gespeichert werden sollen. Standardmäßig "data/reports_data/uniform_distribution.txt".

## Rückgabewert

- `None`

## Beschreibung

Die Methode liest eine CSV-Datei vom angegebenen Pfad und durchläuft jede Spalte. Wenn die Spalte numerisch ist, führt sie den Kolmogorov-Smirnov-Test durch, um zu überprüfen, ob die Daten in der Spalte einer gleichmäßigen Verteilung folgen. Wenn der p-Wert größer als 0,05 ist, wird angenommen, dass die Daten wahrscheinlich einer gleichmäßigen Verteilung folgen. Andernfalls wird angenommen, dass die Daten wahrscheinlich nicht einer gleichmäßigen Verteilung folgen. Die Ergebnisse werden sowohl ausgegeben als auch in einer Textdatei am angegebenen Pfad gespeichert.

## Beispiel

Angenommen, wir haben eine CSV-Datei mit dem Pfad "data.csv". Die Methode kann wie folgt aufgerufen werden:

```python
DataTest.test_uniform_distribution("data.csv")
```

Die Ausgabe könnte so aussehen:

```
Die Daten in der Spalte 'column1' folgen wahrscheinlich einer Uniformverteilung.
Die Daten in der Spalte 'column2' folgen wahrscheinlich nicht einer Uniformverteilung.
```

Und eine Textdatei mit dem Pfad "data/reports_data/uniform_distribution.txt" wird erstellt, die die gleichen Ergebnisse enthält.

---

## Funktion `test_binomial_distribution`

Die Methode `test_binomial_distribution(csv_path, save_distribution_path_txt, p)` ist eine statische Methode, die überprüft, ob die Daten in einer CSV-Datei einer Binomialverteilung folgen.

## Parameter

- `csv_path` (str): Der Pfad zur CSV-Datei.
- `save_distribution_path_txt` (str, optional): Der Pfad zur Textdatei, in der die Ergebnisse gespeichert werden sollen. Standardmäßig "data/reports_data/binomial_distribution.txt".
- `p` (float, optional): Der Erfolgswahrscheinlichkeitsparameter der Binomialverteilung. Standardmäßig 0.5.

## Rückgabewert

- `None`

## Beschreibung

Die Methode liest eine CSV-Datei vom angegebenen Pfad und durchläuft jede Spalte. Wenn die Spalte numerisch ist, ersetzt sie -1 durch 0 und zählt die Werte. Sie berechnet dann die beobachteten Werte und die erwarteten Werte. Wenn der absolute Unterschied zwischen den beobachteten und erwarteten Werten für beide Werte kleiner als 5% ist, wird angenommen, dass die Daten wahrscheinlich einer Binomialverteilung folgen. Andernfalls wird angenommen, dass die Daten wahrscheinlich nicht einer Binomialverteilung folgen. Die Ergebnisse werden sowohl ausgegeben als auch in einer Textdatei am angegebenen Pfad gespeichert.

## Beispiel

Angenommen, wir haben eine CSV-Datei mit dem Pfad "data.csv". Die Methode kann wie folgt aufgerufen werden:

```python
DataTest.test_binomial_distribution("data.csv")
```

Die Ausgabe könnte so aussehen:

```
Die Daten in der Spalte 'column1' folgen wahrscheinlich einer Binomial-Verteilung.
Die Daten in der Spalte 'column2' folgen wahrscheinlich nicht einer Binomial-Verteilung.
```

Und eine Textdatei mit dem Pfad "data/reports_data/binomial_distribution.txt" wird erstellt, die die gleichen Ergebnisse enthält.

---

## Funktion `test_exponential_distribution`

Die Methode `test_exponential_distribution(csv_path, save_distribution_path_txt)` ist eine statische Methode, die überprüft, ob die Daten in einer CSV-Datei einer Exponentialverteilung folgen.

## Parameter

- `csv_path` (str): Der Pfad zur CSV-Datei.
- `save_distribution_path_txt` (str, optional): Der Pfad zur Textdatei, in der die Ergebnisse gespeichert werden sollen. Standardmäßig "data/reports_data/exponential_distribution.txt".

## Rückgabewert

- `None`

## Beschreibung

Die Methode liest eine CSV-Datei vom angegebenen Pfad und durchläuft jede Spalte. Wenn die Spalte numerisch ist, entfernt sie nicht-numerische Werte und führt den Kolmogorov-Smirnov-Test durch, um zu überprüfen, ob die Daten in der Spalte einer Exponentialverteilung folgen. Wenn der p-Wert größer als 0,05 ist, wird angenommen, dass die Daten wahrscheinlich einer Exponentialverteilung folgen. Andernfalls wird angenommen, dass die Daten wahrscheinlich nicht einer Exponentialverteilung folgen. Die Ergebnisse werden sowohl ausgegeben als auch in einer Textdatei am angegebenen Pfad gespeichert.

## Beispiel

Angenommen, wir haben eine CSV-Datei mit dem Pfad "data.csv". Die Methode kann wie folgt aufgerufen werden:

```python
DataTest.test_exponential_distribution("data.csv")
```

Die Ausgabe könnte so aussehen:

```
Die Daten in der Spalte 'column1' folgen wahrscheinlich einer Exponentialverteilung.
Die Daten in der Spalte 'column2' folgen wahrscheinlich nicht einer Exponentialverteilung.
```

Und eine Textdatei mit dem Pfad "data/reports_data/exponential_distribution.txt" würde erstellt, die die gleichen Ergebnisse enthält.

---
## Funktion `test_image_brightness`

Die Methode `test_image_brightness(source_directory, num_images, num_pixels)` ist eine statische Methode, die die Helligkeit von zufällig ausgewählten Bildern in einem angegebenen Verzeichnis berechnet.

## Parameter

- `source_directory` (str): Das Verzeichnis, in dem die Bilder gespeichert sind.
- `num_images` (int, optional): Die Anzahl der zufällig ausgewählten Bilder, die verwendet werden sollen. Standardwert ist 3.
- `num_pixels` (int, optional): Die Anzahl der zufällig ausgewählten Pixel pro Bild, die zur Berechnung der Helligkeit verwendet werden sollen. Standardwert ist 1000.

## Rückgabewert

- Ein Tupel bestehend aus dem Statistikwert und dem p-Wert des Kruskal-Wallis-Tests.

## Beschreibung

Die Methode liest alle .jpg-Bilddateien aus dem angegebenen Verzeichnis und wählt zufällig eine bestimmte Anzahl von Bildern aus. Für jedes ausgewählte Bild wird es in Graustufen konvertiert und ein Array erstellt. Dann werden zufällig eine bestimmte Anzahl von Pixeln ausgewählt und ihre Helligkeitswerte gespeichert. Schließlich wird der Kruskal-Wallis-Test auf die Helligkeitswerte angewendet und das Ergebnis zurückgegeben.

## Beispiel

Angenommen, wir haben ein Verzeichnis mit dem Pfad "images/" und es enthält .jpg-Bilder. Die Methode kann wie folgt aufgerufen werden:

```python
stat, p = DataPreparation.test_image_brightness("images/")
```

Die Ausgabe könnte so aussehen:

```python
(2.345, 0.309)
```

Das bedeutet, dass der Statistikwert des Kruskal-Wallis-Tests 2.345 und der p-Wert 0.309 ist.

---
# Klasse `DataVisualization`

Die Klasse `DataVisualization` ist für die Datenvisualisierung zuständig. Sie hat zwei Attribute: `balanced_gender_path` und `balanced_young_path`, die die Pfade zu den ausgeglichenen Gender- und Young-Dateien speichern.

---

## Funktion `run_datavis`

Die Methode `run_datavis` führt die Datenvisualisierung aus. Sie nimmt vier Parameter: `balanced_gender_path`, `balanced_young_path`, `column_name` und `feature_column`. Die Methode liest die Daten aus den CSV-Dateien, balanciert die Daten und speichert die balancierten Daten in den angegebenen Pfaden. Anschließend erstellt sie Histogramme für die balancierten Daten und speichert diese.

## Funktion `plot_histogram`

Die Methode `plot_histogram` erstellt ein Histogramm für eine gegebene Spalte und speichert es. Sie nimmt fünf Parameter: `df`, `column_name`, `title`, `save_path` und `save_name`. Die Methode zählt die Werte in der angegebenen Spalte, erstellt ein Histogramm und speichert es im angegebenen Pfad.

## Beispiel

Angenommen, wir haben zwei CSV-Dateien: "balanced_gender.csv" und "balanced_young.csv". Die Methode `run_datavis` kann wie folgt aufgerufen werden:

```python
DataVisualization.run_datavis("balanced_gender.csv", "balanced_young.csv", "age", "gender")
```

Die Methode `plot_histogram` kann wie folgt aufgerufen werden:

```python
df = pd.read_csv("balanced_gender.csv")
DataVisualization.plot_histogram(df, "gender", "Gender Distribution", "plots", "gender_distribution")
```

Die Ausgabe wäre ein Histogramm, das die Verteilung der Geschlechter in der Datei "balanced_gender.csv" zeigt und im Verzeichnis "plots" mit dem Namen "gender_distribution.png" gespeichert wird.

---

## Funktion `histogram_all_columns`

Die Funktion `histogram_all_columns(csv_path, save_path)` erstellt Histogramme für alle numerischen Spalten in der angegebenen CSV-Datei und speichert sie im angegebenen Pfad.

## Parameter

- `csv_path` (str): Der Pfad zur CSV-Datei.
- `save_path` (str): Der Pfad zum Speichern der Histogramme.

## Beschreibung

Die Funktion liest die Daten aus der CSV-Datei in einen DataFrame. Dann geht sie durch jede Spalte im DataFrame. Wenn die Spalte numerisch ist, zählt sie die Werte in der Spalte und erstellt ein Histogramm. Das Histogramm wird dann im angegebenen Pfad gespeichert.

## Beispiel

Angenommen, wir haben eine CSV-Datei mit dem Pfad "data.csv" und wir möchten die Histogramme im Verzeichnis "histograms" speichern. Die Funktion kann wie folgt aufgerufen werden:

```python
histogram_all_columns("data.csv", "histograms")
```

Die Ausgabe wären Histogramme für alle numerischen Spalten in der Datei "data.csv", die im Verzeichnis "histograms" gespeichert sind.



