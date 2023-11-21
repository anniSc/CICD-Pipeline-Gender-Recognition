import os
import subprocess
import pandas as pd
import shutil

# Change to the directory that contains your Git repository
# os.chdir(r'C:\Users\busse\Bachelorarbeit\CICD-Pipeline-Gender-Recognition')


# Speichern der Anzahl der zu trainierenden Bilder in einer Textdatei
n = 2
# with open("test/n.txt", "w") as f:
#      f.write(str(n))

# with open("test/test.py", "w") as f:
#      f.write(str(n))



# CSV lesen und für ML-Training vorbereiten
# df = pd.read_csv("data/Gender.csv")
# df = pd.read_csv(r"C:\Users\busse\Bachelorarbeit\CICD-Pipeline-Gender-Recognition\data\local_image_path_Gender.csv")
df = pd.read_csv("data/Gender.csv")
df_sample = df.groupby('Gender', group_keys=False).apply(lambda x: x.sample(n=n, random_state=42))
df_sample = df_sample.sample(frac=1, random_state=42).reset_index(drop=True)
# df_sample.to_excel(f'C:/Users/busse/Bachelorarbeit/CICD-Pipeline-Gender-Recognition/model/excel_sheets/Gender_{n}.xlsx', index=False)
df_sample.to_excel(f'model/excel_sheets/Gender_{n}.xlsx', index=False)
file_names = df_sample['Images'].tolist()

# Liste der Dateipfade erstellen
df_sample['Images'] = df_sample['Images'].str.replace('data/img_align_celeba', 'data/selected_images')

# df_sample.to_excel(f'model/excel_sheets/Gender_{n}.xlsx', index=False)
# source_folder = "C:/Users/busse/Bachelorarbeit/CICD-Pipeline-Gender-Recognition/data/img_align_celeba"
# destination_folder = 'C:/Users/busse/Bachelorarbeit/CICD-Pipeline-Gender-Recognition/data/selected_images/'
source_folder = "data/img_align_celeba"
destination_folder = 'data/selected_images'
# Erstelle eine Liste der Dateipfade der Bilder in "selected_images" basierend auf den Dateinamen in "df_sample"
selected_image_paths = [os.path.join(source_folder, file) for file in file_names]

# Überprüfe, ob der Dateipfad existiert und kopiere die Datei in den Zielordner
for image_path in selected_image_paths:
     if os.path.exists(image_path):
          shutil.copy(image_path, destination_folder)


