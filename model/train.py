# Importieren der benötigten Bibliotheken
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Input

def extract_image_features(images):
    features = list()

    for image in images:
        img = load_img(image, grayscale=True)
        
        img = img.resize((178, 218), Image.LANCZOS)
        img = np.array(img)
        features.append(img)

    features = np.array(features)
    features = features.reshape(len(features), 178, 218, 1)
    return features
print("Model Trained")

n = 200
directory = '/model'
filename = f'trained_{n}_model.h5'  
df = pd.read_csv("model/Gender.csv")
df_sample = df.groupby('Gender', group_keys=False).apply(lambda x: x.sample(n=n, random_state=42))
df_sample = df_sample.sample(frac=1, random_state=42).reset_index(drop=True)
df_sample.to_excel(f'model/excel_sheets/Gender_{n}.xlsx', index=False)
X = extract_image_features(df_sample['Images'])
X = X / 255.0
y_gender = np.array(df_sample['Gender'])
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
dense_1 = Dense(256, activation='relu')(flatten)
dropout_1 = Dropout(0.3)(dense_1)
output_1 = Dense(1, activation='sigmoid', name='gender_out')(dropout_1)
model = Model(inputs=[inputs], outputs=[output_1])
early_stop = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=10, verbose=1, mode='auto')
model.compile(loss=['binary_crossentropy', 'mae'],optimizer='adam', metrics=['accuracy'])
model.fit(x=X_train, y=y_train,batch_size=32, epochs=30, validation_data=(X_test,y_test))
model.save(f"model/saved_trained_Models/trained_{n}_model.h5")
