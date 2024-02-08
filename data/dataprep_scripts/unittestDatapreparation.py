import unittest
from unittest.mock import patch
from datapreparation import Main, DataPreparation, DataTest, DataBalancing, DataVisualization
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from unittest import mock

class TestDataTest(unittest.TestCase):
    def setUp(self):
        self.data_test = DataTest()
    @patch("datapreparation.DataTest.check_data_completeness")
    def test_check_data_completeness(self, mock_function_to_mock):
        csv1 = "data/IDs/data-ids.csv"
        csv2 = "data/labels/source_csv_all_ids.csv"
        expected_output = True

        with patch("pandas.read_csv") as mock_read_csv:
            mock_read_csv.side_effect = lambda _: pd.DataFrame({"id": [1, 2, 3]})
            result = DataTest.check_data_completeness(csv1, csv2)

        self.assertEqual(result, expected_output)

    @patch("datapreparation.DataTest.is_numeric")
    def test_is_numeric(self, mock_function_to_mock):
        column = pd.Series([1, 2, 3])
        expected_output = True

        result = DataTest.is_numeric(column)

        self.assertEqual(result, expected_output)
    @patch("datapreparation.DataTest.test_image_extensions")
    def test_test_image_extensions(self, mock_function_to_mock):
        directory = "path/to/images"
        expected_output = None

        with patch("os.listdir") as mock_listdir:
            mock_listdir.return_value = ["image1.jpg", "image2.png", "image3.bmp"]
            result = DataTest.test_image_extensions(directory)

        self.assertEqual(result, expected_output)
    @patch("datapreparation.DataTest.check_csv_extension")
    def test_check_csv_extension(self, mock_function_to_mock):
        csv_path = "path/to/csv_file.csv"
        expected_output = None

        with patch("os.path.splitext") as mock_splitext:
            mock_splitext.return_value = ("path/to/csv_file", ".csv")
            result = DataTest.check_csv_extension(csv_path)

        self.assertEqual(result, expected_output)
    @patch("datapreparation.DataTest.check_required_directories_data_exists")
    def test_check_required_directories_data_exists(self, mock_function_to_mock):
        directories = ["path/to/directory1", "path/to/directory2"]
        expected_output = None

        with patch("os.path.isdir") as mock_isdir:
            mock_isdir.side_effect = [True, True]
            result = DataTest.check_required_directories_data_exists(directories)

        self.assertEqual(result, expected_output)
    @patch("datapreparation.DataTest.test_quality_of_csv")
    def test_test_quality_of_csv(self):
        csv_path = "path/to/csv_file.csv"
        column_name_of_image_paths = "image_id"
        expected_output = None

        with patch("pandas.read_csv") as mock_read_csv:
            mock_read_csv.return_value = pd.DataFrame({"image_id": [1, 2, 3]})
            result = DataTest.test_quality_of_csv(csv_path, column_name_of_image_paths)

        self.assertEqual(result, expected_output)
    @patch("datapreparation.DataTest.test_outliers_zscore")
    def test_test_outliers_zscore(self, mock_function_to_mock):
        csv_path = "path/to/csv_file.csv"
        expected_output = None

        with mock.patch('module_under_test.test_outliers_zscore') as mock_test_outliers_zscore:
            mock_test_outliers_zscore.return_value = None
            result = DataTest.test_outliers_zscore(csv_path)

        self.assertEqual(result, expected_output)
    @patch("datapreparation.DataTest.test_balance_all_columns")
    def test_test_balance_all_columns(self, mock_function_to_mock):
        csv_path = "path/to/csv_file.csv"
        expected_output = None

        with patch("pandas.read_csv") as mock_read_csv:
            mock_read_csv.return_value = pd.DataFrame({"column1": [1, 1, 1, -1, -1], "column2": [1, 1, 1, 1, 1]})
            result = DataTest.test_balance_all_columns(csv_path)

        self.assertEqual(result, expected_output)
    @patch("datapreparation.DataTest.test_outliers_IQR")
    def test_test_outliers_IQR(self, mock_function_to_mock):
        with mock.patch('module_under_test.test_outliers_IQR') as mock_test_outliers_IQR:
            mock_test_outliers_IQR.return_value = {'column1': 0.0, 'column2': 0.0}
        
        df = pd.DataFrame({"column1": [1, 2, 3, 4, 5], "column2": [6, 7, 8, 9, 10]})
        expected_output = {"column1": 0.0, "column2": 0.0}

        result = DataTest.test_outliers_IQR(df)

        self.assertEqual(result, expected_output)
    @patch("datapreparation.DataTest.detect_anomaly")
    def test_detect_anomaly(self, mock_function_to_mock):
        csv_path = "path/to/csv_file.csv"
        id_column = "id"
        expected_output = None

        with patch("pandas.read_csv") as mock_read_csv:
            mock_read_csv.return_value = pd.DataFrame({"id": [1, 2, 3]})
            with patch("sklearn.ensemble.IsolationForest") as mock_IsolationForest:
                mock_IsolationForest.return_value.fit.return_value.predict.return_value = [-1, 1, -1]
                result = DataTest.detect_anomaly(csv_path, id_column)

        self.assertEqual(result, expected_output)

    @patch("datapreparation.DataTest.test_normal_distribution")
    def test_test_normal_distribution(self, mock_function_to_mock):
        data = "path/to/csv_file.csv"
        save_distribution_path_txt = "path/to/norm_distribution.txt"
        expected_output = None

        with patch("pandas.read_csv") as mock_read_csv:
            mock_read_csv.return_value = pd.DataFrame({"column1": [1, 2, 3], "column2": [4, 5, 6]})
            with patch("builtins.open"):
                result = DataTest.test_normal_distribution(data, save_distribution_path_txt)

        self.assertEqual(result, expected_output)
    @patch("datapreparation.DataTest.test_uniform_distribution")
    def test_test_uniform_distribution(self, mock_function_to_mock):
        data = "path/to/csv_file.csv"
        save_distribution_path_txt = "path/to/uniform_distribution.txt"
        expected_output = None

        with mock.patch('module_under_test.test_uniform_distribution') as mock_test_uniform_distribution:
            mock_test_uniform_distribution.return_value = None
            with patch("builtins.open"):
                result = DataTest.test_uniform_distribution(data, save_distribution_path_txt)

        self.assertEqual(result, expected_output)





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