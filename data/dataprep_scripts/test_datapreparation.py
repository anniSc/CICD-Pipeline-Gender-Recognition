
import unittest
from datapreparation import DataPreparation
import os

import pandas as pd

class TestDataPreparation(unittest.TestCase):

    def test_create_directories(self):
        DataPreparation.create_directories()
        self.assertTrue(os.path.exists(DataPreparation.men_image_source_path_train))
        self.assertTrue(os.path.exists(DataPreparation.women_image_source_path_train))
        self.assertTrue(os.path.exists(DataPreparation.women_image_source_path_test))
        self.assertTrue(os.path.exists(DataPreparation.men_image_source_path_test))
        self.assertTrue(os.path.exists(DataPreparation.IDs))

    def test_extract_ids_source_data_and_save(self):
        directory = "data/img_align_celeba"
        csv_path = "data/IDs/data-ids.csv"
        id_column = "image_id"
        DataPreparation.extract_ids_source_data_and_save(directory, csv_path, id_column)
        self.assertTrue(os.path.exists(csv_path))

    def test_extract_ids(self):
        csv_path = "data/IDs/data-ids.csv"
        column = "Male"
        id_column = "image_id"
        directory = 'data/IDs/'


            # DataPreparation.extract_ids(csv_path,column=column, id_column=id_column)
        self.assertTrue(os.path.exists("data/IDs/male_ids.csv"))
        self.assertTrue(os.path.exists("data/IDs/female_ids.csv"))

        male_df = pd.read_csv("data/IDs/male_ids.csv")
        female_df = pd.read_csv("data/IDs/female_ids.csv")
        self.assertEqual(male_df[id_column].nunique(), len(male_df))
        self.assertEqual(female_df[id_column].nunique(), len(female_df))


if __name__ == '__main__':
    unittest.main()