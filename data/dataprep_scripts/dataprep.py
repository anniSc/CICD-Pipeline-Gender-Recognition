import os
import subprocess
import pandas as pd
import shutil
import tqdm
import glob
import pytest
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.stats import shapiro
import seaborn as sns
from scipy.stats import shapiro, binom_test, kstest, uniform
import numpy as np
import pandas as pd
import shutil
import numpy as np
import os
import random
import os
import pandas as pd
from scipy.stats import shapiro
from scipy.stats import kstest, uniform
import pandas as pd
from scipy.stats import kstest
import pandas as pd
import numpy as np
from scipy.stats import shapiro, kstest, uniform
import os
import subprocess
import pandas as pd
import shutil
import tqdm
import glob
import pytest
import numpy as np
import pandas as pd
from scipy.stats import shapiro
import seaborn as sns
from scipy.stats import shapiro, binom_test, kstest, uniform
import numpy as np
import pandas as pd
import shutil
import numpy as np
import os
import random
import os
import pandas as pd
from scipy.stats import shapiro
from scipy.stats import kstest, uniform
import pandas as pd
from scipy.stats import kstest
import pandas as pd
import numpy as np
from scipy.stats import shapiro, kstest, uniform
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.image as mpimg



import matplotlib.pyplot as plt

class TestDataPreparation:
    @staticmethod
    def check_missing_values(csv_path):
        df = pd.read_csv(csv_path)
        missing_values = df.isnull().any()
        if missing_values.any():
            print(f"Es gibt fehlende Werte in den Daten:\n{missing_values}")
        else:   
            print("Es gibt keine fehlenden Werte in den Daten.")
        return missing_values
    
    @staticmethod
    def test_outliers_all_columns(csv_path):
        df = pd.read_csv(csv_path)
        for column_name in df.columns:
            if np.issubdtype(df[column_name].dtype, np.number):
                z_scores = np.abs((df[column_name] - df[column_name].mean()) / df[column_name].std())
                if any(z_scores > 3):
                    print(f"::warning::Es gibt Ausreißer in der Spalte '{column_name}'")
    
    @staticmethod
    def test_balance_all_columns(csv_path):
        df = pd.read_csv(csv_path)
        imbalance_report = []

        for column_name in df.columns:
            if np.issubdtype(df[column_name].dtype, np.number):
                counts = df[column_name].value_counts()
                if abs(counts.get(-1, 0) - counts.get(1, 0)) >= 0.1 * len(df):
                    imbalance_report.append(f"Die Spalte '{column_name}' ist unausgeglichen. Anzahl von -1: {counts.get(-1, 0)}, Anzahl von 1: {counts.get(1, 0)}")

        if imbalance_report:
            print("Es gibt unausgeglichene Spalten:/n" + "/n".join(imbalance_report))

    @staticmethod
    def is_numeric(column):
        try:
            pd.to_numeric(column)
            return True
        except ValueError:
            return False

    @staticmethod
    def filter_numeric_columns(csv_path):
        df = pd.read_csv(csv_path)
        numeric_columns = [col for col in df.columns if TestDataPreparation.is_numeric(df[col])]
        df = df[numeric_columns]
        df.to_csv(csv_path, index=False)


class TestDataVisualization:
    @staticmethod
    def plot_balance_all_columns(csv_path):
        df = pd.read_csv(csv_path)

        for column_name in df.columns:
            if np.issubdtype(df[column_name].dtype, np.number):
                counts = df[column_name].value_counts()
                counts.plot(kind='bar', title=f"Verteilung der Werte in der Spalte '{column_name}'")
                plt.savefig(f"../plot_data/{column_name}.png")
                plt.show()

    @staticmethod
    def plot_gender_histogram(df):
        counts = df['Male'].value_counts()
        plt.bar(['Female', 'Male'], [counts[-1], counts[1]], color=['#ff69b4', '#1f77b4'])
        for i, v in enumerate([counts[-1], counts[1]]):
            plt.text(i, v, str(v), fontsize=12, ha='center', va='bottom')
        plt.xlabel('Gender')
        plt.ylabel('Count')
        plt.show()

    @staticmethod
    def plot_young_histogram(df):
        counts = df["Young"].value_counts()
        plt.bar(['not Young', 'Young'], [counts[-1], counts[1]], color=['#ff69b4', '#1f77b4'])
        for i, v in enumerate([counts[-1], counts[1]]):
            plt.text(i, v, str(v), fontsize=12, ha='center', va='bottom')
        plt.title('Young oder nicht Young ohne Balanced Gender Data')
        plt.xlabel('Young or not Young')
        plt.ylabel('Count')
        plt.show()


class TestDataAnalysis:
    @staticmethod
    def detect_all_outliers(df):
        outliers_percentage = {}
        for column_name in df.columns:
            if pd.api.types.is_numeric_dtype(df[column_name]):
                Q1 = df[column_name].quantile(0.25)
                Q3 = df[column_name].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = df[(df[column_name] < lower_bound) | (df[column_name] > upper_bound)]
                outliers_percentage[column_name] = len(outliers) / len(df) * 100
        return outliers_percentage

    @staticmethod
    def detect_outliers(df, column_name):
        Q1 = df[column_name].quantile(0.25)
        Q3 = df[column_name].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[column_name] < lower_bound) | (df[column_name] > upper_bound)]
        return outliers

    @staticmethod
    def balance_column(csv_path, column_name):
        df = pd.read_csv(csv_path)
        counts = df[column_name].value_counts()
        min_count = min(counts.get(-1, 0), counts.get(1, 0))
        df_balanced = pd.concat([
            df[df[column_name] == -1].sample(min_count),
            df[df[column_name] == 1].sample(min_count)
        ], axis=0)
        return df_balanced

    @staticmethod
    def test_normal_distribution(data, column_name):
        stat, p = shapiro(data)
        if p > 0.05:
            result = f'::warning::Die Daten in der Spalte {column_name} folgen wahrscheinlich einer Normalverteilung.'
        else:
            result = f'::warning::Die Daten in der Spalte {column_name} folgen wahrscheinlich nicht einer Normalverteilung.'
        with open("../reports_data/norm_distribution.txt", "w") as f:
            f.write(result + "\n")

    @staticmethod
    def test_uniform_distribution(data, column_name):
        theoretical_values = uniform.rvs(size=len(data))
        stat, p = kstest(data, theoretical_values)
        if p > 0.05:
            result = f'::warning::Die Daten in der Spalte {column_name} folgen wahrscheinlich einer Uniformverteilung.'
        else:
            result = f'::warning::Die Daten in der Spalte {column_name} wahrscheinlich nicht einer Uniformverteilung.'
        with open("../reports_data/uniform_distribution.txt", "w") as f:
            f.write(result)

    @staticmethod
    def test_bernoulli_distribution(data, column_name, p):
        data = data.replace(-1, 0)
        value_counts = data.value_counts()
        observed_values = [value_counts.get(1, 0), value_counts.get(0, 0)]
        n = len(data)
        expected_values = [n*p, n*(1-p)]
        if abs(observed_values[0] - expected_values[0]) / n < 0.05 and abs(observed_values[1] - expected_values[1]) / n < 0.05:
            result = f'Die Daten in der Spalte {column_name} folgen wahrscheinlich einer Binomial-Verteilung.'
        else:
            result = f'Die Daten in der Spalte {column_name} folgen wahrscheinlich nicht einer Binomial-Verteilung.'
        with open("../reports_data/binomial_distribution.txt", "w") as f:
            f.write(result)

    @staticmethod
    def test_exponential_distribution(data, column_name):
        stat, p_value = kstest(data, 'expon')
        if p_value > 0.05:
            result = f'Die Daten in der Spalte {column_name} folgen wahrscheinlich einer Exponentialverteilung.'
        else:
            result = f'Die Daten in der Spalte {column_name} folgen wahrscheinlich nicht einer Exponentialverteilung.'
        with open("../reports_data/exponential_distribution.txt", "w") as f:
            f.write(result)

    @staticmethod
    def detect_outliers_iqr_all_columns(csv_path):
        df = pd.read_csv(csv_path)
        outliers_report = []
        for column_name in df.columns:
            if np.issubdtype(df[column_name].dtype, np.number):
                Q1 = df[column_name].quantile(0.25)
                Q3 = df[column_name].quantile(0.75)
                IQR = Q3 - Q1
                outliers = df[(df[column_name] < Q1 - 1.5 * IQR) | (df[column_name] > Q3 + 1.5 * IQR)]
                if not outliers.empty:
                    outliers_report.append(f"Die Spalte '{column_name}' hat Ausreißer. Anzahl: {len(outliers)}")
        return "/n".join(outliers_report)

    @staticmethod
    def check_duplicates(csv_path):
        df = pd.read_csv(csv_path)
        duplicates = df.duplicated()
        if duplicates.any():
            print(f"Es gibt {duplicates.sum()} Duplikate in den Daten.")
        else:
            print("Es gibt keine Duplikate in den Daten.")

    @staticmethod
    def check_null_values(csv_path):
        df = pd.read_csv(csv_path)
        null_values = df.isnull().sum()
        if null_values.any():
            print(f"Es gibt Nullwerte in den Daten:\n{null_values}")
        else:
            print("Es gibt keine Nullwerte in den Daten.")


class TestClass:
    @staticmethod
    def run_tests():
        csv_path="../source_csv/list_attr_celeba.csv"
        source_train_path = "../train-test-data/"
        men_image_source_path_train = "../train-test-data/train/men"
        women_image_source_path_train = "../train-test-data/train/women"
        men_image_source_path_test = "../train-test-data/test/men"
        women_image_source_path_test = "../train-test-data/test/women"
        merged_csv_test = "model/csv_sheets/merged_df_test.csv"
        merged_csv_train = "model/csv_sheets/merged_df_train.csv"
        required_directories = [source_train_path, women_image_source_path_test,men_image_source_path_test,men_image_source_path_train,women_image_source_path_train]
        # Hauptpfad zu den Bildern
        base_path = "../img_align_celeba"
        IDs = "../IDs"
        id_column = 'image_id'
        image_folder = "../img_align_celeba"
        total_images = 10
        male_csv = "../IDs/male_ids.csv"
        female_csv = "../IDs/female_ids.csv"
        df = pd.read_csv(r"C:\CICDPipeline\CICD-Pipeline-Gender-Recognition\data\dataprep_scripts\list_attr_celeba.csv")  
        TestDataPreparation.check_missing_values(r"C:\CICDPipeline\CICD-Pipeline-Gender-Recognition\data\dataprep_scripts\list_attr_celeba.csv")
        TestDataPreparation.test_outliers_all_columns(r"C:\CICDPipeline\CICD-Pipeline-Gender-Recognition\data\dataprep_scripts\list_attr_celeba.csv")
        TestDataPreparation.test_balance_all_columns(r"C:\CICDPipeline\CICD-Pipeline-Gender-Recognition\data\dataprep_scripts\list_attr_celeba.csv")
        TestDataPreparation.filter_numeric_columns(r"C:\CICDPipeline\CICD-Pipeline-Gender-Recognition\data\dataprep_scripts\list_attr_celeba.csv")


TestClass().run_tests()