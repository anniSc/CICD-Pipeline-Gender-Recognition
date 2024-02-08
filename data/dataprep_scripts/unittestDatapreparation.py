import unittest
from unittest.mock import patch
from datapreparation import Main, DataPreparation, DataTest, DataBalancing, DataVisualization
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


class TestDataBalancing(unittest.TestCase):
    @patch.object(pd, 'read_csv')
    def test_balance_column(self, mock_read_csv):
        # Erstelle ein DataFrame mit Testdaten
        df = pd.DataFrame({
            'column': [1, -1, 1, -1, 1, 1, -1, -1, -1, 1]
        })

        # Konfigurieren des Mock-Objekts, um die Testdaten zurückzugeben
        mock_read_csv.return_value = df

        # Rufen Sie die Methode mit den Testdaten auf
        df_balanced = DataBalancing.balance_column('fake_path', 'column')

        # Überprüfen, ob das resultierende DataFrame die erwartete Größe hat
        self.assertEqual(len(df_balanced), 10)

        # Überprüfen, ob das resultierende DataFrame die erwartete Balance hat
        counts = df_balanced['column'].value_counts()
        self.assertEqual(counts[1], counts[-1])


class TestMain(unittest.TestCase):
    @patch.object(DataTest, 'run_datatest')
    @patch.object(DataVisualization, 'run_datavis')
    @patch.object(DataPreparation, 'run_dataprep')
    def setUp(self, mock_run_datatest, mock_run_datavis, mock_run_dataprep):
        self.main = Main()

    def test_variables(self):
        # self.assertEqual(self.main.total_images, 1000)
        self.assertEqual(self.main.balanced_gender_path, "data/balanced_source_csv/gender_balanced.csv")
        self.assertEqual(self.main.balanced_young_path, "data/balanced_source_csv/young_balanced.csv")
        self.assertEqual(self.main.young_column, "Young")
        self.assertEqual(self.main.save_norm_distribution_path_txt, "data/reports_data/norm_distribution.txt")
        self.assertEqual(self.main.save_binomial_distribution_path_txt, "data/reports_data/binomial_distribution.txt")
        self.assertEqual(self.main.save_uniform_distribution_path_txt, "data/reports_data/uniform_distribution.txt")
        self.assertEqual(self.main.save_exponential_distribution_path_txt, "data/reports_data/exponential_distribution.txt")

    def test_methods(self):
        self.assertTrue(hasattr(DataTest, 'run_datatest'))
        self.assertTrue(hasattr(DataVisualization, 'run_datavis'))
        self.assertTrue(hasattr(DataPreparation, 'run_dataprep'))


class TestDataVisualization(unittest.TestCase):
    @patch.object(plt, 'bar')
    @patch.object(plt, 'text')
    @patch.object(plt, 'title')
    @patch.object(plt, 'xlabel')
    @patch.object(plt, 'ylabel')
    @patch.object(plt, 'savefig')
    @patch.object(plt, 'clf')
    def test_plot_histogram(self, mock_clf, mock_savefig, mock_ylabel, mock_xlabel, mock_title, mock_text, mock_bar):
        # Erstellen Sie ein DataFrame mit Testdaten
        df = pd.DataFrame({
            'column': [1, -1, 1, -1, 1]
        })

        # Rufen Sie die Methode mit den Testdaten auf
        DataVisualization.plot_histogram(df, 'column', 'Test Title', '/path/to/save', 'test_name')

        # Überprüfen Sie, ob die erwarteten Methoden aufgerufen wurden
        mock_bar.assert_called_once()
        self.assertEqual(mock_text.call_count, 2)
        mock_title.assert_called_once_with('Test Title')
        mock_xlabel.assert_called_once_with('column or not column')
        mock_ylabel.assert_called_once_with('Count')
        mock_savefig.assert_called_once_with('/path/to/save/test_name.png')
        mock_clf.assert_called_once()




if __name__ == '__main__':
    unittest.main()