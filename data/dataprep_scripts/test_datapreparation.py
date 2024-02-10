
import unittest
from datapreparation import DataPreparation
import os
import sys
import pandas as pd
sys.path.insert(0, "data\dataprep_scripts\datapreparation.py")
sys.path.append("data\dataprep_scripts\datapreparation.py")    
    
import unittest
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datapreparation import DataPreparation

# class TestDataPreparation(unittest.TestCase):

#     def test_create_directories(self):
#         DataPreparation.create_directories()
#         self.assertTrue(os.path.exists(DataPreparation.men_image_source_path_train))
#         self.assertTrue(os.path.exists(DataPreparation.women_image_source_path_train))
#         self.assertTrue(os.path.exists(DataPreparation.women_image_source_path_test))
#         self.assertTrue(os.path.exists(DataPreparation.men_image_source_path_test))
#         self.assertTrue(os.path.exists(DataPreparation.IDs))

#     def test_extract_ids_source_data_and_save(self):
#         directory = "data/img_align_celeba"
#         csv_path = "data/IDs/data-ids.csv"
#         id_column = "image_id"
#         DataPreparation.extract_ids_source_data_and_save(directory, csv_path, id_column)
#         self.assertTrue(os.path.exists(csv_path))

#     def test_extract_ids(self):
#         csv_path = "data/IDs/data-ids.csv"
#         column = "Male"
#         id_column = "image_id"
#         directory = 'data/IDs/'


#             DataPreparation.extract_ids(csv_path,column=column, id_column=id_column)
#         self.assertTrue(os.path.exists("data/IDs/male_ids.csv"))
#         self.assertTrue(os.path.exists("data/IDs/female_ids.csv"))

#         male_df = pd.read_csv("data/IDs/male_ids.csv")
#         female_df = pd.read_csv("data/IDs/female_ids.csv")
#         self.assertEqual(male_df[id_column].nunique(), len(male_df))
#         self.assertEqual(female_df[id_column].nunique(), len(female_df))


# if __name__ == '__main__':
#     unittest.main()import unittest
import unittest
from datapreparation import DataTest


class TestDataTest(unittest.TestCase):
    def _init_(self):
        self.data = "data/source_csv/list_attr_celeba.csv"

    def test_check_data_completeness(self):
        csv1 = "data/IDs/data-ids.csv"
        csv2 = "data/IDs/source_csv_all_ids.csv"
        result = DataTest.check_data_completeness(csv1, csv2)
        self.assertTrue(result)

    def test_is_numeric(self):
        column1 = [1, 2, 3, 4, 5]
        column2 = ["a", "b", "c", "d", "e"]
        result1 = DataTest.is_numeric(column1)
        result2 = DataTest.is_numeric(column2)
        self.assertTrue(result1)
        self.assertFalse(result2)

    def test_test_image_extensions(self):
        directory = "data/img_align_celeba"
        DataTest.test_image_extensions(directory)
        # Add assertions here

    def test_check_csv_extension(self):
        DataTest.check_csv_extension(data)
        # Add assertions here

    def test_check_required_directories_data_exists(self):
        directories = [
            "data/img_align_celeba",
            "data/IDs",
            "data/reports_data"
        ]
        DataTest.check_required_directories_data_exists(directories)
        # Add assertions here

    def test_test_quality_of_csv(self):
        csv_path = "data/IDs/data-ids.csv"
        column_name_of_image_paths = "image_id"
        DataTest.test_quality_of_csv(csv_path, column_name_of_image_paths)
        # Add assertions here

    def test_test_outliers_zscore(self):
        csv_path = "data/IDs/data-ids.csv"
        DataTest.test_outliers_zscore(csv_path)
        # Add assertions here

    def test_test_balance_all_columns(self):
        csv_path = "data/IDs/data-ids.csv"
        DataTest.test_balance_all_columns(csv_path)
        # Add assertions here

    def test_test_outliers_IQR(self):
        csv_path = "data/IDs/data-ids.csv"
        df = pd.read_csv(csv_path)
        result = DataTest.test_outliers_IQR(df)
        # Add assertions here

    # def test_detect_outliers(self):
    #     csv_path = "data/IDs/data-ids.csv"
    #     column_name = "age"
    #     df = pd.read_csv(csv_path)
    #     outliers = DataTest.detect_outliers(df, column_name)
    #     Add assertions here

    def test_detect_anomaly(self):
        id_column = "image_id"
        DataTest.detect_anomaly(data, id_column)
        # Add assertions here

    def test_test_normal_distribution(self):
        save_distribution_path_txt = "data/plot_data/norm_distribution.txt"
        DataTest.test_normal_distribution(data, save_distribution_path_txt)
        # Add assertions here

    def test_test_uniform_distribution(self):
        
        save_distribution_path_txt = "data/reports_data/uniform_distribution.txt"
        DataTest.test_uniform_distribution(data, save_distribution_path_txt)
        # Add assertions here
data = "data/source_csv/list_attr_celeba.csv"
if __name__ == '__main__':
    unittest.main()


class TestDataPreparation(unittest.TestCase):

    def test_histogram_all_columns(self):
        csv_path = "data/source_csv/list_attr_celeba.csv"
        save_path = "data/histograms"
        DataPreparation.histogram_all_columns(csv_path, save_path)

        # Add assertions here
        # For example, you can check if the histogram images are saved correctly
        self.assertTrue(os.path.exists(f"{save_path}/column_name.png"))

if __name__ == '__main__':
    unittest.main()