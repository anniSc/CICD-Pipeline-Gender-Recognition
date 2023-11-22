import os
import subprocess
import pandas as pd
import shutil

# Speichern der Anzahl der zu trainierenden Bilder in einer Textdatei
n = 20000

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



