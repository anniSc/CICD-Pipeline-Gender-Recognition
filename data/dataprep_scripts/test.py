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

# Erstelle einen DataContext aus der Konfigurationsdatei
context = ge.data_context.DataContext()

# Erstelle eine neue Erwartungssuite mit dem DataContext
suite = context.create_expectation_suite("my_suite")

