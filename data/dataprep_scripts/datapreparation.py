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


class DataPreparation:
    total_images = 0
    source_csv = "data/source_csv/list_attr_celeba.csv"
    csv_path = "data/source_csv/list_attr_celeba.csv"
    source_train_path = "data/train-test-data/"
    image_source_path = "data/img_align_celeba"
    men_image_source_path_train = "data/train-test-data/train/men"
    women_image_source_path_train = "data/train-test-data/train/women"
    men_image_source_path_test = "data/train-test-data/test/men"
    women_image_source_path_test = "data/train-test-data/test/women"
    required_directories = [source_train_path, women_image_source_path_test, men_image_source_path_test, men_image_source_path_train, women_image_source_path_train]
    base_path = "data/img_align_celeba"
    IDs = "data/IDs"
    id_column = 'image_id'
    image_folder = "data/img_align_celeba"
    male_csv = "data/IDs/male_ids.csv"
    female_csv = "data/IDs/female_ids.csv"
    data_ids = "data/IDs/data-ids.csv"
    feature_column= "Male"
    data_vis_path = "data/plot_data"
    @staticmethod
    def create_directories():
        os.makedirs(DataPreparation.men_image_source_path_train, exist_ok=True)
        os.makedirs(DataPreparation.women_image_source_path_train, exist_ok=True)
        os.makedirs(DataPreparation.women_image_source_path_test, exist_ok=True)
        os.makedirs(DataPreparation.men_image_source_path_test, exist_ok=True)
        os.makedirs(DataPreparation.IDs, exist_ok=True)

    @staticmethod
    def save_filenames_to_csv(csv_path, csv_name=f"../IDs/data-ids.csv", id_column="image_id"):
        df = pd.read_csv(csv_path)
        df = df[[id_column]]
        df.to_csv(csv_name, index=False)

    @staticmethod
    def extract_all_ids(csv_path, column="Male", id_column="image_id"):
        df = pd.read_csv(csv_path)
        df[column] = df[column].replace(-1, 0)
        df.to_csv(f'data/IDs/source_csv_all_ids.csv', columns=[id_column], index=False)

    @staticmethod
    def compare_columns(csv1, csv2):
        df1 = pd.read_csv(csv1)
        df2 = pd.read_csv(csv2)
        column1 = df1.iloc[:, 0]
        column2 = df2.iloc[:, 0]
        is_equal = column1.equals(column2)
        if is_equal:
            print("::warning:: Daten sind vollständig! Die Bilddaten-IDs stimmen mit den IDs aus Attributliste überein! ")
        else:
            print("::error:: Die Bilddaten-IDs stimmen nicht mit den IDs aus Attributliste überein! ")
        return is_equal

    @staticmethod
    def extract_ids(csv_path, column="Male", id_column="image_id"):
        df = pd.read_csv(csv_path)
        df[column] = df[column].replace(-1, 0)
        male_df = df[df[column] == 1]
        female_df = df[df[column] == 0]
        male_df.to_csv(f'data/IDs/male_ids.csv', columns=[id_column], index=False)
        female_df.to_csv(f'data/IDs/female_ids.csv', columns=[id_column], index=False)

    @staticmethod
    def clear_directory(dir_path):
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

    @staticmethod
    def get_ids_from_csv(csv_file, id_column):
        df = pd.read_csv(csv_file)
        ids = df[id_column].tolist()
        return ids

    @staticmethod
    def split_data_random(image_folder, male_csv, female_csv, total_images, id_column, train_ratio=0.7):
        male_ids = DataPreparation.get_ids_from_csv(male_csv, id_column)
        female_ids = DataPreparation.get_ids_from_csv(female_csv, id_column)

        num_train = int(total_images * train_ratio)
        num_test = int(total_images - num_train)

        test_ids_male = random.sample(male_ids, num_test // 2)
        test_ids_female = random.sample(female_ids, num_test // 2)
        test_ids = test_ids_male + test_ids_female

        train_ids_male = set(male_ids) - set(test_ids)
        train_ids_female = set(female_ids) - set(test_ids)
        male_ids = sorted(train_ids_male)
        female_ids = sorted(train_ids_female)
        female_ids = random.sample(female_ids, num_train // 2)
        male_ids = random.sample(male_ids, num_train // 2)

        for id in test_ids_male:
            shutil.copy(os.path.join(image_folder, id), DataPreparation.men_image_source_path_test)

        for id in test_ids_female:
            shutil.copy(os.path.join(image_folder, id), DataPreparation.women_image_source_path_test)

        for id in male_ids:
            shutil.copy(os.path.join(image_folder, id), DataPreparation.men_image_source_path_train)

        for id in female_ids:
            shutil.copy(os.path.join(image_folder, id), DataPreparation.women_image_source_path_train)

    @staticmethod
    def get_image_paths(source_path):
        image_formats = ['*.jpg', '*.png', '*.gif', '*.jpeg']
        image_paths = []
        for format in image_formats:
            image_paths.extend(glob.glob(os.path.join(source_path, format)))
        return image_paths

    @staticmethod
    def test_image_extensions_in_csv(csv_path, column_name_of_image_paths="image_id"):
        df = pd.read_csv(csv_path)
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
        df['valid_extension'] = df[column_name_of_image_paths].apply(lambda x: os.path.splitext(x)[1].lower() in valid_extensions)
        invalid_rows = df[~df['valid_extension']].index
        if len(invalid_rows) > 0:
            print(f'Ungültige Dateierweiterungen gefunden in den Zeilen: {invalid_rows.tolist()}')

        assert all(df['valid_extension']), f'Nicht alle Werte in der Spalte {column_name_of_image_paths} verweisen auf Bilddateien./n {invalid_rows} /n Überprüfe die Dateierweiterungen.'

    @staticmethod
    def check_csv_extension(csv_path):
        _, ext = os.path.splitext(csv_path)
        assert ext.lower() == '.csv', f'Die Datei {csv_path} hat keine .csv Erweiterung'

    @staticmethod
    def check_required_directories_data_exists(directories):
        for directory in directories:
            assert os.path.isdir(directory), f'Das Verzeichnis {directory} existiert nicht'

    @staticmethod
    def test_quality_of_csv(csv_path,column_name_of_image_paths="image_id"):
        df = pd.read_csv(csv_path)
        assert df[column_name_of_image_paths].isnull().sum() == 0, f'Es gibt fehlende Werte in der Spalte {column_name_of_image_paths}'
        assert df.duplicated().sum() == 0, "Es gibt Duplikate in der Daten"
    @staticmethod
    def check_missing_values(csv_path):
        df = pd.read_csv(csv_path)
        missing_values = df.isnull().any()
        return missing_values
    
    @staticmethod
    def test_outliers_all_columns(csv_path):
        df = pd.read_csv(csv_path)
        for column_name in df.columns:
            if np.issubdtype(df[column_name].dtype, np.number):  # Überprüfe, ob die Spalte numerisch ist
                z_scores = np.abs((df[column_name] - df[column_name].mean()) / df[column_name].std())
                if any(z_scores > 3):
                    print(f"::warning::Es gibt Ausreißer in der Spalte '{column_name}'")
    @staticmethod
    def is_numeric(column):
        try:
            pd.to_numeric(column)
            return True
        except ValueError:
            return False
    @staticmethod
    def test_balance_all_columns(csv_path):
        df = pd.read_csv(csv_path)
        imbalance_report = []

        for column_name in df.columns:
            if DataPreparation.is_numeric(df[column_name]) == True:
                if np.issubdtype(df[column_name].dtype, np.number):  # Überprüfe, ob die Spalte numerisch ist
                    counts = df[column_name].value_counts()
                    if abs(counts.get(-1, 0) - counts.get(1, 0)) >= 0.1 * len(df):
                        imbalance_report.append(f"Die Spalte '{column_name}' ist unausgeglichen. Anzahl von -1: {counts.get(-1, 0)}, Anzahl von 1: {counts.get(1, 0)}")
        if imbalance_report:
            print("Es gibt unausgeglichene Spalten:\n" + "\n".join(imbalance_report))

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

class DataTest: 
        @staticmethod
        def get_image_paths(source_path):
            image_formats = ['*.jpg', '*.png', '*.gif', '*.jpeg']
            image_paths = []
            for format in image_formats:
                image_paths.extend(glob.glob(os.path.join(source_path, format)))
            return image_paths

        @staticmethod
        def test_image_extensions_in_csv(csv_path, column_name_of_image_paths="image_id"):
            df = pd.read_csv(csv_path)
            valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
            df['valid_extension'] = df[column_name_of_image_paths].apply(lambda x: os.path.splitext(x)[1].lower() in valid_extensions)
            invalid_rows = df[~df['valid_extension']].index
            if len(invalid_rows) > 0:
                print(f'Ungültige Dateierweiterungen gefunden in den Zeilen: {invalid_rows.tolist()}')

            assert all(df['valid_extension']), f'Nicht alle Werte in der Spalte {column_name_of_image_paths} verweisen auf Bilddateien./n {invalid_rows} /n Überprüfe die Dateierweiterungen.'

        @staticmethod
        def check_csv_extension(csv_path):
            _, ext = os.path.splitext(csv_path)
            assert ext.lower() == '.csv', f'Die Datei {csv_path} hat keine .csv Erweiterung'

        @staticmethod
        def check_required_directories_data_exists(directories):
            for directory in directories:
                assert os.path.isdir(directory), f'Das Verzeichnis {directory} existiert nicht'

        @staticmethod
        def test_quality_of_csv(csv_path,column_name_of_image_paths="image_id"):
            df = pd.read_csv(csv_path)
            assert df[column_name_of_image_paths].isnull().sum() == 0, f'Es gibt fehlende Werte in der Spalte {column_name_of_image_paths}'
            assert df.duplicated().sum() == 0, "Es gibt Duplikate in der Daten"
        @staticmethod
        def check_missing_values(csv_path):
            df = pd.read_csv(csv_path)
            missing_values = df.isnull().any()
            return missing_values
        
        @staticmethod
        def test_outliers_all_columns(csv_path):
            df = pd.read_csv(csv_path)
            for column_name in df.columns:
                if np.issubdtype(df[column_name].dtype, np.number):  # Überprüfe, ob die Spalte numerisch ist
                    z_scores = np.abs((df[column_name] - df[column_name].mean()) / df[column_name].std())
                    if any(z_scores > 3):
                        print(f"::warning::Es gibt Ausreißer in der Spalte '{column_name}'")
        @staticmethod
        def is_numeric(column):
            try:
                pd.to_numeric(column)
                return True
            except ValueError:
                return False
        @staticmethod
        def test_balance_all_columns(csv_path):
            df = pd.read_csv(csv_path)
            imbalance_report = []

            for column_name in df.columns:
                if DataPreparation.is_numeric(df[column_name]) == True:
                    if np.issubdtype(df[column_name].dtype, np.number):  # Überprüfe, ob die Spalte numerisch ist
                        counts = df[column_name].value_counts()
                        if abs(counts.get(-1, 0) - counts.get(1, 0)) >= 0.1 * len(df):
                            imbalance_report.append(f"Die Spalte '{column_name}' ist unausgeglichen. Anzahl von -1: {counts.get(-1, 0)}, Anzahl von 1: {counts.get(1, 0)}")
            if imbalance_report:
                print("Es gibt unausgeglichene Spalten:\n" + "\n".join(imbalance_report))

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
        def detect_anomaly(csv_path, id_column):
            from sklearn.ensemble import IsolationForest
            import numpy as np
            X = pd.read_csv(csv_path)
            X = X.drop(id_column, axis=1)
            clf = IsolationForest(random_state=0).fit(X)
            y_pred = clf.predict(X)
            if -1 not in y_pred or 1 not in y_pred:
                raise ValueError("Anomalie gefunden!")
            print("Keine Anomalien gefunden.")
        
        @staticmethod
        def test_normal_distribution(data, save_distribution_path_txt="data/plot_data/norm_distribution.txt"):
                df = pd.read_csv(data)
                results =[]
                for column_name in df.columns:
                    if pd.api.types.is_numeric_dtype(df[column_name]):
                        stat, p = shapiro(df[column_name])
                        if p > 0.05:
                            
                            result = f'Die Daten in der Spalte {column_name} folgen wahrscheinlich einer Normalverteilung.'
                            print(result)
                        else:
                            result = f'Die Daten in der Spalte {column_name} folgen wahrscheinlich nicht einer Normalverteilung.'
                            print(result)
                        results.append(result)
                with open(f"{save_distribution_path_txt}", "w") as f:
                    for result in results:
                        f.write(f"{result}\n")
        @staticmethod
        def test_uniform_distribution(data, save_distribution_path_txt="data/reports_data/uniform_distribution.txt"):
            df = pd.read_csv(data)
            results =[]
            for column_name in df.columns:
                if pd.api.types.is_numeric_dtype(df[column_name]):
                    theoretical_values = uniform.rvs(size=len(df))
                    stat, p = kstest(df[column_name], theoretical_values)
                    if p > 0.05:
                        result = f'Die Daten in der Spalte {column_name} folgen wahrscheinlich einer Uniformverteilung.'
                        print(result)
                    else:
                        result = f'Die Daten in der Spalte {column_name} folgen wahrscheinlich nicht einer Uniformverteilung.'
                        print(result)
                    results.append(result)

            with open(f"{save_distribution_path_txt}", "w") as f:
                for result in results:
                    f.write(f"{result}\n")    
        @staticmethod
        def test_binomial_distribution(csv_path, save_distribution_path_txt="data/reports_data/binomial_distribution.txt", p=0.5):
            df = pd.read_csv(csv_path)
            results =[]
            for column_name in df.columns:
                if pd.api.types.is_numeric_dtype(df[column_name]):
                    data = df[column_name].replace(-1, 0) 
                    value_counts = data.value_counts() 
                    observed_values = [value_counts.get(1, 0), value_counts.get(0, 0)]
                    n = len(data)

                    expected_values = [n*p, n*(1-p)]

                    if abs(observed_values[0] - expected_values[0]) / n < 0.05 and abs(observed_values[1] - expected_values[1]) / n < 0.05:
                        result = f'Die Daten in der Spalte {column_name} folgen wahrscheinlich einer Binomial-Verteilung.'
                        print(result)
                    else:
                        result = f'Die Daten in der Spalte {column_name} folgen wahrscheinlich nicht einer Binomial-Verteilung.'
                        print(result)
                    results.append(result)  

            with open(save_distribution_path_txt, "w") as f:
                for result in results:
                    f.write(f"{result}\n")

        @staticmethod
        def test_exponential_distribution(csv_path, save_distribution_path_txt="data/reports_data/exponential_distribution.txt"):
            df = pd.read_csv(csv_path)
            results =[]
            for column_name in df.columns:
                if pd.api.types.is_numeric_dtype(df[column_name]):  
                    # Entfernen Sie nicht-numerische Werte
                    data = df[column_name].dropna()
                    stat, p_value = kstest(data, 'expon')
                    if p_value > 0.05:
                        result = f'Die Daten in der Spalte {column_name} folgen wahrscheinlich einer Exponentialverteilung.'
                        print(result)
                    else:
                        result = f'Die Daten in der Spalte {column_name} folgen wahrscheinlich nicht einer Exponentialverteilung.'
                        print(result)
                    results.append(result)

            with open(save_distribution_path_txt, "w") as f:
                for result in results:
                    f.write(f"{result}\n")

        @staticmethod
        def test_image_brightness(source_directory, num_images=3, num_pixels=1000):
            from scipy.stats import kruskal
            from PIL import Image  

            image_files = [f for f in os.listdir(source_directory) if f.endswith('.jpg')]
            
            # Zufällige Auswahl von Bildern
            selected_images = random.sample(image_files, num_images)
            
            # Liste zur Speicherung der Helligkeitswerte
            brightness_values = []
            
            for image_file in selected_images:
                img = Image.open(os.path.join(source_directory, image_file)).convert('L')
                img_array = np.array(img)
                pixel_indices = random.sample(range(img_array.size), num_pixels)
                selected_pixels = img_array.ravel()[pixel_indices]
                brightness_values.append(selected_pixels)
            stat, p = kruskal(*brightness_values)
            return stat, p

       

class DataBalancing: 
        def balance_column(csv_path, column_name):
            df = pd.read_csv(csv_path)
            counts = df[column_name].value_counts()
            min_count = min(counts.get(-1, 0), counts.get(1, 0))
            df_balanced = pd.concat([
                df[df[column_name] == -1].sample(min_count),
                df[df[column_name] == 1].sample(min_count)
            ], axis=0)
            return df_balanced 

 
class DataVisualization:
        def plot_histogram(df, column_name, title, save_path, save_name):
            counts = df[column_name].value_counts()
            plt.bar([f'not {column_name}', f'{column_name}'], [counts[-1], counts[1]], color=['#ff69b4', '#1f77b4'])
            for i, v in enumerate([counts[-1], counts[1]]):
                plt.text(i, v, str(v), fontsize=12, ha='center', va='bottom')
            plt.title(title)
            plt.xlabel(f'{column_name} or not {column_name}')
            plt.ylabel('Count')
            plt.savefig(f"{save_path}/{save_name}.png")
            # plt.show()

        def histogram_all_columns(csv_path,save_path):
            df = pd.read_csv(csv_path)
            for column_name in df.columns:
                if np.issubdtype(df[column_name].dtype, np.number):  # Überprüfe, ob die Spalte numerisch ist
                    counts = df[column_name].value_counts()
                    counts.plot(kind='bar', title=f"Verteilung der Werte in der Spalte '{column_name}'")
                    plt.savefig(f"{save_path}/{column_name}.png")
                    # plt.show()



class Main(DataPreparation, DataTest, DataBalancing, DataVisualization): 
    """
    Die Hauptklasse, die die verschiedenen Funktionen zur Datenverarbeitung, Datenprüfung, Datenbalancierung und Datenvisualisierung enthält.
    """

    df_balanced_gender_path = "data/balanced_source_csv/gender_balanced.csv"
    df_balanced_young_path = "data/balanced_source_csv/young_balanced.csv"
    young_column = "Young" 
    save_norm_distribution_path_txt = "data/reports_data/norm_distribution.txt"
    save_binomial_distribution_path_txt = "data/reports_data/binomial_distribution.txt"    
    save_uniform_distribution_path_txt = "data/reports_data/uniform_distribution.txt"  
    save_exponential_distribution_path_txt = "data/reports_data/exponential_distribution.txt"
    DataPreparation.total_images = 10

    # Erstelle Verzeichnisse für die Datenverarbeitung
    DataPreparation.create_directories()

    # Speichere Dateinamen in einer CSV-Datei
    DataPreparation.save_filenames_to_csv(csv_path=DataPreparation.csv_path, csv_name=DataPreparation.data_ids, id_column=DataPreparation.id_column) 

    # Extrahiere alle IDs aus einer CSV-Datei
    DataPreparation.extract_all_ids(csv_path=DataPreparation.csv_path, column=DataPreparation.feature_column, id_column=DataPreparation.id_column)

    # Vergleiche Spalten in zwei CSV-Dateien
    DataPreparation.compare_columns(csv1=DataPreparation.data_ids, csv2=DataPreparation.csv_path)

    # Extrahiere IDs aus einer CSV-Datei
    DataPreparation.extract_ids(csv_path=DataPreparation.csv_path, column=DataPreparation.feature_column, id_column=DataPreparation.id_column)

    # Lösche Verzeichnisse
    DataPreparation.clear_directory(dir_path=DataPreparation.men_image_source_path_test)
    DataPreparation.clear_directory(dir_path=DataPreparation.men_image_source_path_train)
    DataPreparation.clear_directory(dir_path=DataPreparation.women_image_source_path_test)
    DataPreparation.clear_directory(dir_path=DataPreparation.women_image_source_path_train)

    # Teile die Daten zufällig in männliche und weibliche CSV-Dateien auf
    DataPreparation.split_data_random(image_folder=DataPreparation.image_folder, male_csv=DataPreparation.male_csv, female_csv=DataPreparation.female_csv, total_images=DataPreparation.total_images, id_column=DataPreparation.id_column)

    # Teste Bildendungen in einer CSV-Datei
    DataTest.test_image_extensions_in_csv(csv_path=DataPreparation.csv_path, column_name_of_image_paths=DataPreparation.id_column)

    # Überprüfe die Dateiendung einer CSV-Datei
    DataTest.check_csv_extension(csv_path=DataPreparation.csv_path)

    # Überprüfe, ob erforderliche Verzeichnisse und Daten vorhanden sind
    DataTest.check_required_directories_data_exists(directories=DataPreparation.required_directories)

    # Test the quality of a CSV file
    DataTest.test_quality_of_csv(csv_path=DataPreparation.csv_path, column_name_of_image_paths=DataPreparation.id_column)

    # Check for missing values in a CSV file
    DataTest.check_missing_values(csv_path=DataPreparation.csv_path)

    # Test for outliers in all columns of a CSV file
    DataTest.test_outliers_all_columns(csv_path=DataPreparation.csv_path)

    # Test the balance of all columns in a CSV file
    DataTest.test_balance_all_columns(csv_path=DataPreparation.csv_path)

    # Detect all outliers in a DataFrame
    DataTest.detect_all_outliers(df=pd.read_csv(DataPreparation.csv_path))

    # Detect anomalies in a CSV file
    DataTest.detect_anomaly(csv_path=DataPreparation.csv_path, id_column=DataPreparation.id_column)

    # Test the brightness of images in a directory
    stat, p = DataTest.test_image_brightness(source_directory=DataPreparation.image_folder, num_images=3, num_pixels=1000)
    print(f"Kruskal-Wallis-Test result: {stat}, p-value: {p}. A small p-value (typically less than 0.05) indicates that there is likely a significant difference in the brightness values of the selected images.")

    # Test for normal distribution in data and save the distribution to a text file
    DataTest.test_normal_distribution(data=DataPreparation.csv_path, save_distribution_path_txt=save_norm_distribution_path_txt)

    # Test for uniform distribution in data and save the distribution to a text file
    DataTest.test_uniform_distribution(data=DataPreparation.csv_path, save_distribution_path_txt=save_uniform_distribution_path_txt)

    # Test for binomial distribution in a CSV file and save the distribution to a text file
    DataTest.test_binomial_distribution(csv_path=DataPreparation.csv_path, save_distribution_path_txt=save_binomial_distribution_path_txt, p=0.5)

    # Test for exponential distribution in a CSV file and save the distribution to a text file
    DataTest.test_exponential_distribution(csv_path=DataPreparation.csv_path, save_distribution_path_txt=save_exponential_distribution_path_txt)

    # Plot histograms for all columns in a CSV file and save them as PNG files
    DataVisualization.histogram_all_columns(DataPreparation.csv_path, DataPreparation.data_vis_path)

    # Balance a column in a CSV file and save the balanced DataFrame to a new CSV file
    df_balanced_gender = DataBalancing.balance_column(csv_path=DataPreparation.csv_path, column_name=DataPreparation.feature_column)
    df_balanced_gender.to_csv(df_balanced_gender_path, index=False)

    # Plot a histogram for a column in a DataFrame and save it as a PNG file
    DataVisualization.plot_histogram(df=df_balanced_gender, column_name=DataPreparation.feature_column, title="Balanced Distribution of Genders", save_path=DataPreparation.data_vis_path, save_name="balanced_gender.png")

    # Balance another column in a CSV file and save the balanced DataFrame to a new CSV file
    df_balanced_young = DataBalancing.balance_column(csv_path=DataPreparation.csv_path, column_name=young_column)
    df_balanced_young.to_csv(df_balanced_young_path, index=False)

    # Plot a histogram for a column in a DataFrame and save it as a PNG file
    DataVisualization.plot_histogram(df=df_balanced_young, column_name=DataPreparation.feature_column, title="Balanced Distribution of Young and Old", save_path=DataPreparation.data_vis_path, save_name="balanced_young.png")

Main()