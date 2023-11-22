# import unittest
# from train import preprocess_data, predict, train_model

# class MLTests(unittest.TestCase):
#     def test_data_preprocessing(self):
#         # Test case for data preprocessing
#         input_data = [...]  # Input data for preprocessing
#         expected_output = [...]  # Expected output after preprocessing
#         self.assertEqual(preprocess_data(input_data), expected_output)

#     def test_model_training(self):
#         # Test case for model training
#         input_data = [...]  # Input data for training
#         expected_output = [...]  # Expected trained model
#         self.assertEqual(train_model(input_data), expected_output)

#     def test_prediction(self):
#         # Test case for prediction
#         input_data = [...]  # Input data for prediction
#         expected_output = [...]  # Expected prediction result
#         self.assertEqual(predict(input_data), expected_output)

# if __name__ == '__main__':
#     unittest.main()


from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd
from PIL import Image
with open("test/n.txt", "r") as f:
    n = int(f.read())
# print(n)
# Import the necessary modules
# df_test = pd.read_excel(f'C:/Users/busse/Bachelorarbeit/CICD-Pipeline-Gender-Recognition/model/excel_sheets/Gender_{n}.xlsx')
df_test = pd.read_excel(f'model/excel_sheets/Gender_{n}.xlsx')
X_test = df_test['Images']
# X_test = X_test.replace('data/img_align_celeba/', 'C:/Users/busse/Bachelorarbeit/CICD-Pipeline-Gender-Recognition/data/selected_images/', regex=True)
# print(X_test)

image_list = X_test.tolist()
model = load_model(f'model/saved_trained_Models/trained_{n}_model.h5')
# model = load_model(f'C:/Users/busse/Bachelorarbeit/CICD-Pipeline-Gender-Recognition/model/saved_trained_Models/trained_{n}_model.h5')
# print(image_list)
counter = 0
for image in image_list:
    # if counter == 10: 
    #     break
    img = load_img(image, target_size=(178, 218), color_mode="grayscale")
    # Image.open(image).show()
    # Convert the image to a numpy array
    img_array = img_to_array(img)

    # Expand dimensions to match the input shape that your model expects
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    if predictions[0] < 0.5:
        print(f'Frau:{predictions}')
    else:
        print(f'Mann:{predictions}')
    # counter += 1
# #     Load the model from a .h5 file
#     model = load_model(f'C:/Users/busse/Bachelorarbeit/CICD-Pipeline-Gender-Recognition/model/saved_trained_Models/trained_{n}_model.h5')

# #     Use the loaded model for predictions
#     predictions = model.predict(img)

# Perform further operations with the loaded model
# ...


