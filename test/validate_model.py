# Import the necessary modules
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd
from PIL import Image



# df_test = pd.read_excel(f'C:/Users/busse/Bachelorarbeit/CICD-Pipeline-Gender-Recognition/model/excel_sheets/Gender_{n}.xlsx')
# model = load_model(f'C:/Users/busse/Bachelorarbeit/CICD-Pipeline-Gender-Recognition/model/saved_trained_Models/trained_{n}_model.h5')
# X_test = X_test.replace('data/img_align_celeba/', 'C:/Users/busse/Bachelorarbeit/CICD-Pipeline-Gender-Recognition/data/selected_images/', regex=True)

with open("test/n.txt", "r") as f:
    n = int(f.read())


df_test = pd.read_excel(f'model/excel_sheets/Gender_{n}.xlsx')
X_test = df_test['Images']


image_list = X_test.tolist()
model = load_model(f'model/saved_trained_Models/trained_{n}_model.h5')

counter = 0
for image in image_list:

    img = load_img(image, target_size=(178, 218), color_mode="grayscale")

    img_array = img_to_array(img)

    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    if predictions[0] < 0.5:
        print(f'Frau:{predictions}')
    else:
        print(f'Mann:{predictions}')
  



