from tensorflow.keras.models import load_model


with open("test/n.txt", "r") as f:
    n = int(f.read())

model = load_model(f'model/saved_trained_Models/trained_{n}_model.h5')