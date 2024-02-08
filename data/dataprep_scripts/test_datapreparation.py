import unittest
from unittest.mock import patch
from datapreparation import DataTest
import pandas as pd
import os

class TestDataTest(unittest.TestCase):

    def test_check_data_completeness(self):
        csv1 = "data/IDs/data-ids.csv"
        csv2 = "data/labels/source_csv_all_ids.csv"
        expected_output = True

        with patch("pandas.read_csv") as mock_read_csv:
            mock_read_csv.side_effect = lambda csv_path: pd.DataFrame({"id": [1, 2, 3]})
            result = DataTest.check_data_completeness(csv1, csv2)

        self.assertEqual(result, expected_output)

    def test_is_numeric(self):
        column = pd.Series([1, 2, 3])
        expected_output = True

        result = DataTest.is_numeric(column)

        self.assertEqual(result, expected_output)

    def test_test_image_extensions(self):
        directory = "path/to/images"
        expected_output = None

        with patch("os.listdir") as mock_listdir:
            mock_listdir.return_value = ["image1.jpg", "image2.png", "image3.bmp"]
            result = DataTest.test_image_extensions(directory)

        self.assertEqual(result, expected_output)

    def test_check_csv_extension(self):
        csv_path = "path/to/csv_file.csv"
        expected_output = None

        with patch("os.path.splitext") as mock_splitext:
            mock_splitext.return_value = ("path/to/csv_file", ".csv")
            result = DataTest.check_csv_extension(csv_path)

        self.assertEqual(result, expected_output)

    def test_check_required_directories_data_exists(self):
        directories = ["path/to/directory1", "path/to/directory2"]
        expected_output = None

        with patch("os.path.isdir") as mock_isdir:
            mock_isdir.side_effect = [True, True]
            result = DataTest.check_required_directories_data_exists(directories)

        self.assertEqual(result, expected_output)

    def test_test_quality_of_csv(self):
        csv_path = "path/to/csv_file.csv"
        column_name_of_image_paths = "image_id"
        expected_output = None

        with patch("pandas.read_csv") as mock_read_csv:
            mock_read_csv.return_value = pd.DataFrame({"image_id": [1, 2, 3]})
            result = DataTest.test_quality_of_csv(csv_path, column_name_of_image_paths)

        self.assertEqual(result, expected_output)

    def test_test_outliers_zscore(self):
        csv_path = "path/to/csv_file.csv"
        expected_output = None

        with patch("pandas.read_csv") as mock_read_csv:
            mock_read_csv.return_value = pd.DataFrame({"column1": [1, 2, 3], "column2": [4, 5, 6]})
            result = DataTest.test_outliers_zscore(csv_path)

        self.assertEqual(result, expected_output)

    def test_test_balance_all_columns(self):
        csv_path = "path/to/csv_file.csv"
        expected_output = None

        with patch("pandas.read_csv") as mock_read_csv:
            mock_read_csv.return_value = pd.DataFrame({"column1": [1, 1, 1, -1, -1], "column2": [1, 1, 1, 1, 1]})
            result = DataTest.test_balance_all_columns(csv_path)

        self.assertEqual(result, expected_output)

    def test_test_outliers_IQR(self):
        df = pd.DataFrame({"column1": [1, 2, 3, 4, 5], "column2": [6, 7, 8, 9, 10]})
        expected_output = {"column1": 0.0, "column2": 0.0}

        result = DataTest.test_outliers_IQR(df)

        self.assertEqual(result, expected_output)

    def test_detect_anomaly(self):
        csv_path = "path/to/csv_file.csv"
        id_column = "id"
        expected_output = None

        with patch("pandas.read_csv") as mock_read_csv:
            mock_read_csv.return_value = pd.DataFrame({"id": [1, 2, 3]})
            with patch("sklearn.ensemble.IsolationForest") as mock_IsolationForest:
                mock_IsolationForest.return_value.fit.return_value.predict.return_value = [-1, 1, -1]
                result = DataTest.detect_anomaly(csv_path, id_column)

        self.assertEqual(result, expected_output)

    def test_test_normal_distribution(self):
        data = "path/to/csv_file.csv"
        save_distribution_path_txt = "path/to/norm_distribution.txt"
        expected_output = None

        with patch("pandas.read_csv") as mock_read_csv:
            mock_read_csv.return_value = pd.DataFrame({"column1": [1, 2, 3], "column2": [4, 5, 6]})
            with patch("builtins.open") as mock_open:
                result = DataTest.test_normal_distribution(data, save_distribution_path_txt)

        self.assertEqual(result, expected_output)

    def test_test_uniform_distribution(self):
        data = "path/to/csv_file.csv"
        save_distribution_path_txt = "path/to/uniform_distribution.txt"
        expected_output = None

        with patch("pandas.read_csv") as mock_read_csv:
            mock_read_csv.return_value = pd.DataFrame({"column1": [1, 2, 3], "column2": [4, 5, 6]})
            with patch("builtins.open") as mock_open:
                result = DataTest.test_uniform_distribution(data, save_distribution_path_txt)

        self.assertEqual(result, expected_output)

if __name__ == "__main__":
    unittest.main()