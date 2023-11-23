# Import the necessary modules
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt


# df_test = pd.read_excel(f'C:/Users/busse/Bachelorarbeit/CICD-Pipeline-Gender-Recognition/model/excel_sheets/Gender_{n}.xlsx')
# model = load_model(f'C:/Users/busse/Bachelorarbeit/CICD-Pipeline-Gender-Recognition/model/saved_trained_Models/trained_{n}_model.h5')
# X_test = X_test.replace('data/img_align_celeba/', 'C:/Users/busse/Bachelorarbeit/CICD-Pipeline-Gender-Recognition/data/selected_images/', regex=True)


  


with open("test/val_acc_values.txt", "r") as f:
        val_acc_values=f.read()

with open("test/acc.txt", "r") as f:
        acc=f.read()   

with open("test/loss.txt", "r") as f:
        loss=f.read()

with open("test/val_loss.txt", "r") as f:
        val_loss=f.read()


val_acc_values = val_acc_values.replace('[', '')
val_acc_values = val_acc_values.replace(']', '')
val_acc_values = [float(val) for val in val_acc_values.split(',')]
acc = acc.replace('[', '')
acc = acc.replace(']', '')
acc = [float(val) for val in acc.split(',')]
loss = loss.replace('[', '')
loss = loss.replace(']', '')
loss = [float(val) for val in loss.split(',')]
val_loss = val_loss.replace('[', '')
val_loss = val_loss.replace(']', '')
val_loss = [float(val) for val in val_loss.split(',')]


# val_acc_values = sorted(val_acc_values, reverse=False)

# Erstellen Sie das Diagramm
epochs = range(1, len(val_acc_values) + 1)

plt.plot(epochs, val_acc_values, label='Validation Accuracy')
plt.plot(epochs, acc, label='Accuracy')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.plot(epochs, loss, label='Loss')

plt.title('Training and validation metrics')
plt.xlabel('Epochs')
plt.ylabel('Metrics')
plt.legend()

plt.show()


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
