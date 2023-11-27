# Importieren der benötigten Bibliotheken
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Input, Concatenate

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

def train_model():
    with open("test/n.txt", "r") as f:
         n = int(f.read())
    with open("model/selected_features.txt", "r") as f:
        selected_features = f.read().splitlines()

    

    df_sample = pd.read_excel(f'model/excel_sheets/Gender_{n}.xlsx')
    X = extract_image_features(df_sample['Images'])
    X = X / 255.0
    
    additional_features = df_sample[selected_features].values

    # print(additional_features)
    # Teilen Sie die zusätzlichen Features in Trainings- und Testsets auf
    additional_features_train, additional_features_test = train_test_split(additional_features, test_size=0.2, random_state=42)

    # y_gender = np.array(additional_features)
    y_gender = np.array(df_sample["Male"])
    X_train, X_test, y_train, y_test = train_test_split(X, y_gender, test_size=0.2, random_state=42)
    input_shape = (178, 218, 1)
    input1 = Input((input_shape))
    conv_1 = Conv2D(32, kernel_size=(4, 4), activation='relu')(input1)
    max_1 = MaxPooling2D(pool_size=(2, 2))(conv_1)
    conv_2 = Conv2D(64, kernel_size=(3, 3), activation='relu')(max_1)
    max_2 = MaxPooling2D(pool_size=(2, 2))(conv_2)
    conv_3 = Conv2D(128, kernel_size=(3, 3), activation='relu')(max_2)
    max_3 = MaxPooling2D(pool_size=(2, 2))(conv_3)
    conv_4 = Conv2D(256, kernel_size=(3, 3), activation='relu')(max_3)
    max_4 = MaxPooling2D(pool_size=(2, 2))(conv_4)
    flatten = Flatten()(max_4)


    # input2 = Input(shape=(len(selected_features),))
    dense_1 = Dense(256, activation='relu')(flatten)

    # merged = Concatenate()([flatten,dense_1])
    # dropout_1 = Dropout(0.3)(merged)
    # output_1 = Dense(1, activation='sigmoid', name='gender_out')(dropout_1)


    dropout_1 = Dropout(0.3)(dense_1)
    output_1 = Dense(1, activation='sigmoid', name='gender_out')(dropout_1)
    model = Model(inputs=[input1], outputs=[output_1])

    early_stop = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=10, verbose=1, mode='auto')
    model.compile(loss=['binary_crossentropy', 'mae'],optimizer='adam', metrics=['accuracy'])
    history = model.fit(x=X_train, y=y_train,batch_size=32, epochs=50, validation_data=(X_test,y_test))
    
    # Kompilieren und Trainieren des Modells
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # history = model.fit(x=[X_train, additional_features_train], y=y_train, batch_size=32, epochs=50, validation_data=([X_test, additional_features_test], y_test))
        
    
    
    val_acc_values = history.history['val_accuracy']
    acc = history.history['accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']



    with open("test/val_acc_values.txt", "w") as f:
        f.write(str(val_acc_values))
    with open ("test/acc.txt", "w") as f:
        f.write(str(acc))
    with open("test/loss.txt", "w") as f:   
        f.write(str(loss))
    with open("test/val_loss.txt", "w") as f:
        f.write(str(val_loss))  
    model.save(f"model/saved_trained_Models/trained_{n}_model.h5")




train_model()