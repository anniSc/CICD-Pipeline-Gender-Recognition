import unittest
from unittest.mock import patch
from datapreparation import Main, DataPreparation, DataTest, DataBalancing, DataVisualization

class TestMain(unittest.TestCase):
    @patch.object(DataTest, 'run_datatest')
    @patch.object(DataVisualization, 'run_datavis')
    @patch.object(DataPreparation, 'run_dataprep')
    def setUp(self, mock_run_datatest, mock_run_datavis, mock_run_dataprep):
        self.main = Main()

    def test_variables(self):
        self.assertEqual(self.main.total_images, 10)
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

if __name__ == '__main__':
    unittest.main()