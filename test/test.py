from tensorflow.keras.models import load_model


with open("test/n.txt", "r") as f:
    n = int(f.read())

print(f'trained_{n}_model.h5')
model = load_model(f'trained_{n}_model.h5')