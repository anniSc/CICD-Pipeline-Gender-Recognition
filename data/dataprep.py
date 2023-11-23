import os
import subprocess
import pandas as pd
import shutil
import sys
import random
import warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from logging import warning
from PIL import Image
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import load_img
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Input
import os
import pandas as pd
import shutil



def prepare_data(n):
     # CSV lesen und für ML-Training vorbereiten
     df = pd.read_csv("data/Gender.csv")
     df_sample = df.groupby('Gender', group_keys=False).apply(lambda x: x.sample(n=n, random_state=42))
     df_sample = df_sample.sample(frac=1, random_state=42).reset_index(drop=True)
     df_sample.to_excel(f'model/excel_sheets/Gender_{n}.xlsx', index=False)
     file_names = df_sample['Images'].tolist()
     print(file_names)

     # Liste der Dateipfade erstellen
     df_sample['Images'] = df_sample['Images'].str.replace('data/img_align_celeba', 'data/selected_images')

     # source_folder = "data/img_align_celeba"
     destination_folder = 'data/selected_images'

     # Lösche die Bilder im "selected_images" Ordner, wenn er nicht leer ist
     if os.listdir(destination_folder):
          for file_name in os.listdir(destination_folder):
               file_path = os.path.join(destination_folder, file_name)
               os.remove(file_path)

     # Überprüfe, ob der Dateipfad existiert und kopiere die Datei in den Zielordner
     for image_path in file_names:
          if os.path.exists(image_path):
               print(image_path)
               shutil.copy(image_path, destination_folder)

def prepare_training_data(n=10):
     # Festlegen wo die Daten gespeichert und abgerufen werden sollen für das Training des ML-Modells
     BASE_DIR = 'data/img_align_celeba'
     image_paths = []
     image_filenames = os.listdir(BASE_DIR)

     for image in tqdm(image_filenames):
          image_path = os.path.join(BASE_DIR, image)
          image_paths.append(image_path)

     # Überprüfung der Anzahl der Bilder 
     print("Anzahl der ausgewählten Bilder für das Training beträgt:" + f'{len(image_paths)}')

     # Festlegen der zu verwendenten Labels, Features 
     list_attr_celebaCSV = 'data/list_attr_celeba.csv'
     df = pd.read_csv(list_attr_celebaCSV)
     df_Gender = df["Male"]

     # Labels für ein CNN-Modell von tensorflow.keras vorbereiten -1 = Frau, 1 = Mann zu 0 = Frau, 1 = Mann
     df_Gender = df_Gender.replace(-1, 0)
     df_Gender = df_Gender.values

     # Zusammenführen der Labels und die Dateipfade zu den Bildern damit CNN-Modell richtige Werte zu jedem Bild zuordnen kann
     df = pd.DataFrame({'Images':image_paths, 'Gender': df_Gender})

     # Speichern der Anzahl der zu trainierenden Bilder in einer Textdatei wird für train.py und validate_model.py benötigt
     with open("test/n.txt", "w") as f:
          f.write(str(n))


     # CSV lesen und für ML-Training vorbereiten
     df = pd.read_csv("data/Gender.csv")
     df_sample = df.groupby('Gender', group_keys=False).apply(lambda x: x.sample(n=n, random_state=42))
     df_sample = df_sample.sample(frac=1, random_state=42).reset_index(drop=True)
     df_sample.to_excel(f'model/excel_sheets/Gender_{n}.xlsx', index=False)
     file_names = df_sample['Images'].tolist()
     print(file_names)

     # Liste der Dateipfade erstellen
     df_sample['Images'] = df_sample['Images'].str.replace('data/img_align_celeba', 'data/selected_images')

     # source_folder = "data/img_align_celeba"
     destination_folder = 'data/selected_images'

     # Lösche die Bilder im "selected_images" Ordner, wenn er nicht leer ist
     if os.listdir(destination_folder):
          for file_name in os.listdir(destination_folder):
               file_path = os.path.join(destination_folder, file_name)
               os.remove(file_path)


     # Überprüfe, ob der Dateipfad existiert und kopiere die Datei in den Zielordner
     for image_path in file_names:
          if os.path.exists(image_path):
               print(image_path)
               shutil.copy(image_path, destination_folder)

def prepare_excel_list(selected_labels=[], n=10): 
     with open("test/n.txt", "w") as f:
          f.write(str(n))

     BASE_DIR = 'data/img_align_celeba'
     image_paths = []
     image_filenames = os.listdir(BASE_DIR)

     for image in image_filenames:
          image_path = os.path.join(BASE_DIR, image)
          image_paths.append(image_path)

     # Liste aller Dateien im Verzeichnis
     # files = os.listdir('data/attribute_list_label')
     files = os.listdir('data/attribute_list_labels')
     # Filtern Sie die Liste, um nur CSV-Dateien zu behalten
     csv_files = [file for file in files if file.endswith('.csv')]

     # Überprüfen Sie, ob genau eine CSV-Datei vorhanden ist
     if len(csv_files) == 1:
          print("Es gibt genau eine CSV-Datei im Verzeichnis.")
          csv_files = str(csv_files).replace("[", "").replace("]", "").replace("'", "")
          print(csv_files)

          # Read the CSV file
          csv_file_path = os.path.join('data/attribute_list_labels', csv_files)
          df = pd.read_csv(csv_file_path)

          # Drop columns not in selected_labels
          columns_to_drop = [col for col in df.columns if col not in selected_labels]
          df.drop(columns_to_drop, axis=1, inplace=True)
          # Assuming image_paths and selected_labels are lists
          df_image_paths = pd.DataFrame(image_paths, columns=['Images'])

          # Merge the two dataframes
          df_merge = pd.merge(df_image_paths, df, left_index=True, right_index=True)

          df_sample = df_merge.groupby('Male', group_keys=False).apply(lambda x: x.sample(n=n, random_state=42))
          df_sample = df_sample.sample(frac=1, random_state=42).reset_index(drop=True)
          df_sample.to_excel(f'model/excel_sheets/Gender_{n}.xlsx', index=False)
          
          return df_sample, n
          # Save the modified CSV file
          # df.to_csv(csv_file_path, index=False)
     elif len(csv_files) > 1:
          print("Es gibt mehr als eine CSV-Datei im Verzeichnis.")
     else:
          print("Es gibt keine CSV-Datei im Verzeichnis.")
         

selected_labels = ["Male", "Big_Nose", "No_Beard"]
df, n = prepare_excel_list(selected_labels)

df["Male"].replace({-1: 0}, inplace=True)
df["Big_Nose"].replace({-1: 0}, inplace=True)
df["No_Beard"].replace({-1: 0}, inplace=True)
df["Images"] = df["Images"].str.replace('\\', '/')
print(df)

df.to_excel(f'model/excel_sheets/Gender_{n}.xlsx', index=False)
