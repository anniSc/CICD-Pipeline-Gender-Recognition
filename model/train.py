# Importieren der benötigten Bibliotheken
import os
import sys
import random
import warnings
import numpy as np
import pandas as pd
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
os.environ['KMP_DUPLICATE_LIB_OK']='True'
warnings.filterwarnings('ignore')



BASE_DIR = 'data/img_align_celeba'
image_paths = []
image_filenames = os.listdir(BASE_DIR)

for image in tqdm(image_filenames):
  image_path = os.path.join(BASE_DIR, image)
  image_paths.append(image_path)

len(image_paths)
image_paths



list_attr_celebaCSV = 'C:/Users/busse/Bachelorarbeit/CICD-Pipeline-Gender-Recognition/data/list_attr_celeba.csv'
df = pd.read_csv(list_attr_celebaCSV)
df_Male = df["Male"]
df_Male = df_Male.replace(-1, 0)
df_gender = df_Male.values
df = pd.DataFrame({'Images':image_paths, 'Gender': df_gender})
first_row = df.iloc[0]
image_df = df.drop("Gender", axis=1)




def find_file(filename, directory):
    for root, dirs, files in os.walk(directory):
        if filename in files:
            return os.path.join(root, filename)
    return None


class StopExecution(Exception):
    def _render_traceback_(self):
        pass

n = 2000

directory = 'C:/Users/busse/CICDPipeline/BA-CICD-Pipeline/model'
filename = f'trained_{n}_model.h5'  
file_path = find_file(filename, directory)

print(n)
df_sample = df.groupby('Gender', group_keys=False).apply(lambda x: x.sample(n=n, random_state=42))
sns.displot(df_sample["Gender"])
df_sample

df_sample = df_sample.sample(frac=1, random_state=42).reset_index(drop=True)
df_sample.to_excel(f'Gender_{n}.xlsx', index=False)




def extract_image_features(images):
    features = list()

    for image in tqdm(images):
        img = load_img(image, grayscale=True)
        
        img = img.resize((178, 218), Image.LANCZOS)
        img = np.array(img)
        features.append(img)

    features = np.array(features)
    features = features.reshape(len(features), 178, 218, 1)
    return features

X = extract_image_features(df_sample['Images'])

X = X / 255.0

X.shape

y_gender = np.array(df_sample['Gender'])




class MyCallback(Callback):
    def __init__(self, threshold):
        super(MyCallback, self).__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None): 
        val_acc = logs.get('val_accuracy')
        if val_acc is not None and val_acc >= self.threshold:
            self.model.stop_training = True
            # Save the model
            self.model.save(f'{n}.h5') 




# Assuming X is your feature set and y_gender is your target variable
X_train, X_test, y_train, y_test = train_test_split(X, y_gender, test_size=0.2, random_state=42)


input_shape = (178, 218, 1)
inputs = Input((input_shape))
conv_1 = Conv2D(32, kernel_size=(4, 4), activation='relu')(inputs)
max_1 = MaxPooling2D(pool_size=(2, 2))(conv_1)
conv_2 = Conv2D(64, kernel_size=(3, 3), activation='relu')(max_1)
max_2 = MaxPooling2D(pool_size=(2, 2))(conv_2)
conv_3 = Conv2D(128, kernel_size=(3, 3), activation='relu')(max_2)
max_3 = MaxPooling2D(pool_size=(2, 2))(conv_3)
conv_4 = Conv2D(256, kernel_size=(3, 3), activation='relu')(max_3)
max_4 = MaxPooling2D(pool_size=(2, 2))(conv_4)

flatten = Flatten()(max_4)

# fully connected layers
dense_1 = Dense(256, activation='relu')(flatten)
# dense_2 = Dense(256, activation='relu')(flatten)

dropout_1 = Dropout(0.3)(dense_1)
# dropout_2 = Dropout(0.3)(dense_2)

output_1 = Dense(1, activation='sigmoid', name='gender_out')(dropout_1)

# output = Dense(1, activation='sigmoid')(flatten)
model = Model(inputs=[inputs], outputs=[output_1])
# model = Model(inputs=[inputs], outputs=[output])


# Define the early stopping criteria
early_stop = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=10, verbose=1, mode='auto')

model.compile(loss=['binary_crossentropy', 'mae'],optimizer='adam', metrics=['accuracy'])



print(len(y_train))
print(len(X_train))

checkpoint_path = "training/cp-{epoch:04d}.ckpt"
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True,
    period=10)

model.fit(x=X_train, y=y_train,batch_size=32, epochs=30, validation_data=(X_test,y_test),callbacks=[cp_callback])