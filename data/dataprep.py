import os
import subprocess
import pandas as pd


# Change to the directory that contains your Git repository
os.chdir(r'C:\Users\busse\Bachelorarbeit\CICD-Pipeline-Gender-Recognition')

# Speichern der Anzahl der zu trainierenden Bilder in einer Textdatei
n = 1000
with open("test/n.txt", "w") as f:
     f.write(str(n))

# # CSV lesen und für ML-Training vorbereiten
# df = pd.read_csv("data/Gender.csv")
# df_sample = df.groupby('Gender', group_keys=False).apply(lambda x: x.sample(n=n, random_state=42))
# df_sample = df_sample.sample(frac=1, random_state=42).reset_index(drop=True)
# # df_sample.to_excel(f'model/excel_sheets/Gender_{n}.xlsx', index=False)

# # Liste der Dateipfade erstellen
# file_names = df['Images'].tolist()

# # Check if branch exists and switch to a new branch name if necessary
# branch_name = 'test-branch5'
# # branches = subprocess.check_output(['git', 'branch']).decode().split('\n')
# # if f'* {branch_name}' in branches:
# #     branch_name = f'test-branch-new'

# # Create a new branch
# subprocess.run(['git', 'checkout', '-b', branch_name])

# # Add each file path to the staging area
# for file_name in file_names:
#     full_path = os.path.join('img_align_celeba', file_name)
#     if os.path.exists(full_path):
#         subprocess.run(['git', 'add', full_path])

# # Commit and push to the new branch
# subprocess.run(['git', 'commit', '-m', 'Add files from Excel'])
# subprocess.run(['git', 'push', 'origin', branch_name])