# Importiere die Great Expectations Bibliothek
import great_expectations as ge
# Erstelle eine Klasse, die von PandasDataset erbt
class CustomPandasDataset(ge.dataset.PandasDataset):

    # Definiere deine eigenen Erwartungen
    def expect_column_values_to_be_even(self, column):
        # Diese Erwartung überprüft, ob die Werte in einer Spalte gerade sind
        return self.expect_column_values_to_be_in_set(column, list(range(0, self[column].max() + 2, 2)))

    def expect_column_values_to_be_odd(self, column):
        # Diese Erwartung überprüft, ob die Werte in einer Spalte ungerade sind
        return self.expect_column_values_to_be_in_set(column, list(range(1, self[column].max() + 2, 2)))
    # Erstelle ein Datenobjekt aus der CSV-Datei
data = ge.read_csv(r"C:\Users\busse\Bachelorarbeit\CICD-Pipeline-Gender-Recognition\data\source_csv\list_attr_celeba.csv")
data = ge.from_pandas(data, dataset_class=CustomPandasDataset)
# Erstelle eine neue Erwartungssuite
suite = data.create_expectation_suite("my_suite")

# Füge Erwartungen hinzu, die die Datenqualität überprüfen
suite.expect_column_to_exist("image_id") # Erwarte, dass die Spalte "name" existiert

# Validiere die Daten mit der Erwartungssuite
results = data.validate(expectation_suite=suite)

# Zeige die Ergebnisse der Validierung an
print(results)