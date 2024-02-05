import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from fairlearn.datasets import fetch_adult
from fairlearn.metrics import MetricFrame
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from torch.autograd import Variable
from torchcam.methods import GradCAM, GradCAMpp, LayerCAM, ScoreCAM, SmoothGradCAMpp, XGradCAM
from torchvision import datasets, models, transforms
from torchvision.io import read_image
from torchvision.transforms import ToTensor
from tqdm import tqdm
from model_train import DataLoaderModelTrain, SimpleCNN
import torchvision.transforms.functional as TF

class ModelTester():
    
    def test_model_robustness(model, test_dataloader, device):
        model.eval()  # Set the model to evaluation mode
        true_labels = []
        predicted_labels = []

        with torch.no_grad():  # Do not calculate gradients to speed up computation
            for inputs, labels in test_dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                output = model(inputs)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                predictions = torch.argmax(probabilities, dim=1)

                true_labels.extend(labels.cpu().numpy())
                predicted_labels.extend(predictions.cpu().numpy())

        # Calculate metrics
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels, average='weighted')
        recall = recall_score(true_labels, predicted_labels, average='weighted')
        f1 = f1_score(true_labels, predicted_labels, average='weighted')

        print(f'Genauigkeit: {accuracy}')
        print(f'Präzision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1-Score: {f1}')
        return accuracy, precision, recall, f1

    @staticmethod
    def get_model_path(directory):
        for filename in os.listdir(directory):
            if filename.endswith(".pth"):
                print(filename)
                return os.path.join(directory, filename)
                
        return None
    
    @staticmethod
    def add_noise(images, noise_factor=0.5):
        noise = torch.randn_like(images) * noise_factor
        noisy_images = images + noise
        noisy_images = torch.clamp(noisy_images, 0., 1.)
        return noisy_images

    def add_noise_and_test(model, test_dataloader, device, noise_factor=0.5):
        noisy_images = []
        labels_list = []
        
        for inputs, labels in test_dataloader:
            inputs_noisy = ModelTester.add_noise(inputs, noise_factor)
            noisy_images.append(inputs_noisy)
            labels_list.append(labels)

        # Stack images and labels separately
        noisy_images = torch.cat(noisy_images)
        labels_list = torch.cat(labels_list)

        noisy_dataloader = torch.utils.data.DataLoader(list(zip(noisy_images, labels_list)), batch_size=test_dataloader.batch_size)

        ModelTester.test_model_robustness(model, noisy_dataloader, device)



    def evaluate_model(model, test_dataloader):
        model.eval()
        predictions = []
        true_labels = []
        with torch.no_grad():
            for inputs, labels in test_dataloader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                predictions.extend(predicted)
                true_labels.extend(labels)


        # Konvertiere die Listen in numpy-Arrays
        predictions = np.array(predictions)
        true_labels = np.array(true_labels)

        # Berechne die Metriken
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, average='weighted')
        recall = recall_score(true_labels, predictions, average='weighted')
        f1 = f1_score(true_labels, predictions, average='weighted')

        print(f'Genauigkeit: {accuracy}, Präzision: {precision}, Recall: {recall}, F1-Score: {f1}')

        with open('test/metrics/metrics.txt', 'w') as outfile:
            outfile.write(f'Modellmetriken: Genauigkeit: {accuracy}, Präzision: {precision}, Recall: {recall}, F1-Score: {f1}')

        return accuracy, precision, recall, f1

    def test_predicts(model, test_dataloader, device):
        # Für jede Eingabe im Testdatensatz:
        for inputs, _ in test_dataloader:
            # Die Eingaben werden auf das Gerät verschoben (GPU oder CPU).
            inputs = inputs.to(device) 
            # Das Modell macht eine Vorhersage auf den Eingaben.
            output = model(inputs)
            # Die Ausgabe des Modells wird in Wahrscheinlichkeiten umgewandelt, indem die Softmax-Funktion angewendet wird.
            probabilities = torch.nn.functional.softmax(output, dim=1)
            # Die Klasse mit der höchsten Wahrscheinlichkeit wird als Vorhersage ausgewählt.
            predictions = torch.argmax(probabilities, dim=1)
            # Die Vorhersagen werden in eine Liste umgewandelt und ausgegeben.
            predictions_list = predictions.cpu().numpy().tolist()
            print(predictions_list)
    
    def test_noise_robustness(model, test_dataloader, device, start_noise=0.0, end_noise=1.0, step=0.1,savefig_path="test/test-plots-rauschen"):
        noise_factor = start_noise
        i = 0
        while noise_factor <= end_noise:
            noisy_images = []
            labels_list = []

            for inputs, labels in test_dataloader:
                inputs_noisy = ModelTester.add_noise(inputs, noise_factor)
                noisy_images.append(inputs_noisy)
                labels_list.append(labels)

            noisy_images = torch.cat(noisy_images)
            labels_list = torch.cat(labels_list)

            noisy_dataloader = torch.utils.data.DataLoader(list(zip(noisy_images, labels_list)), batch_size=test_dataloader.batch_size)

            print(f'::warning::Test mit noise factor: {noise_factor}')
            accuracy, precision,recall, f1 = ModelTester.test_model_robustness(model, noisy_dataloader, device)
            

            # Show a sample noisy image
            plt.figure(figsize=(12, 6))
            plt.title(f'Verauschtes Bild mit noise factor: {noise_factor}.', fontsize=10)
            plt.imshow(noisy_images[0].permute(1, 2, 0))
            plt.text(1.2, 0.6, f'Genauigkeit: {accuracy}', horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)
            plt.text(1.2, 0.5, f'Precision: {precision}', horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)
            plt.text(1.2, 0.4, f'Recall: {recall}', horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)
            plt.text(1.2, 0.3, f'F1 Score: {f1}', horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)
            plt.savefig(f"{savefig_path}/{i}.png")
            # plt.show()

            if accuracy < 0.7:
                print(f'::warning::Genauigkeit unter 70% mit noise factor: {noise_factor}. Test wird gestoppt.')
                break
            i += 1
            noise_factor += step


    def add_distortion(image, distortion_factor=0.5):
        startpoints = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        endpoints = startpoints + torch.tensor([[0.0, distortion_factor], [0.0, -distortion_factor], [0.0, 0.0], [0.0, 0.0]])
        distorted_image = TF.perspective(image, startpoints, endpoints)
        return distorted_image
        
    def test_distortion_robustness(model, test_dataloader, device, start_distortion=0.0, end_distortion=1.0, step=0.0001, savefig_path="test/test-plots-verzerrung"):
        distortion_factor = start_distortion
        i = 0
        while distortion_factor <= end_distortion:
            distorted_images = []
            labels_list = []

            for inputs, labels in test_dataloader:
                inputs_distorted = ModelTester.add_distortion(inputs, distortion_factor)
                distorted_images.append(inputs_distorted)
                labels_list.append(labels)

            distorted_images = torch.cat(distorted_images)
            labels_list = torch.cat(labels_list)

            distorted_dataloader = torch.utils.data.DataLoader(list(zip(distorted_images, labels_list)), batch_size=test_dataloader.batch_size)

            print(f'Test mit Verzerrungs-Faktor: {distortion_factor}')
            accuracy, precision, recall, f1 = ModelTester.test_model_robustness(model, distorted_dataloader, device)
            
            # Show a sample distorted image
            plt.figure(figsize=(12, 6))
            plt.title(f'Verzerrtes Bild mit Verzerrungs-Faktor: {distortion_factor}.', fontsize=10)
            plt.imshow(distorted_images[0].permute(1, 2, 0))
            plt.text(1.2, 0.6, f'Genauigkeit: {accuracy}', horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)
            plt.text(1.2, 0.5, f'Präzsion: {precision}', horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)
            plt.text(1.2, 0.4, f'Recall: {recall}', horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)
            plt.text(1.2, 0.3, f'F1-Score: {f1}', horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)
            plt.savefig(f"{savefig_path}/{i}.png")
            print(f"test/test-plots-verzerrung/{i}.png")
            plt.show()

            if accuracy < 0.7:
                print(f'::warning::Genauigkeit unter 70% mit Verzerrungs-Faktor: {distortion_factor}. Test wird gestoppt.')
                break
            i += 1
            distortion_factor += step

    def rotate_and_convert(image, angle, test_images = "data/train-test-data/test"):
        from PIL import Image
        for filename in os.listdir(test_images):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image_path = os.path.join(test_images, filename)
                image = Image.rotate(image, angle)


    def test_model_robustness_rotation(model, test_dataloader, device):
        model.eval()  # Set the model to evaluation mode
        true_labels = []
        predicted_labels = []

        with torch.no_grad():  # Do not calculate gradients to speed up computation
            for inputs, labels in test_dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                output = model(inputs)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                predictions = torch.argmax(probabilities, dim=1)

                true_labels.extend(labels.cpu().numpy())
                predicted_labels.extend(predictions.cpu().numpy())

        # Calculate metrics
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels, average='weighted', zero_division=1)
        recall = recall_score(true_labels, predicted_labels, average='weighted')
        f1 = f1_score(true_labels, predicted_labels, average='weighted')

        return accuracy, precision, recall, f1
    
    def convert_image_to_tensor(image_path):
            image = Image.open(image_path)
            transform = transforms.Compose([
                transforms.ToTensor()
            ])
            return transform(image)
    

    def rotate_tensor_image(image_tensor, angle):
        rotated_image = TF.rotate(image_tensor, angle)
        return rotated_image    
    
    def test_rotation_robustness(model, test_dataloader, device, start_angle=0.0, end_angle=270.0, step=90.0, savefig_path="test/test-plots-rotation"):
        rotation_angle = start_angle
        i = 0
        while rotation_angle <= end_angle:
            rotated_images = []
            labels_list = []

            for inputs, labels in test_dataloader:
           
                for input in inputs:
                    input_rotated = ModelTester.rotate_tensor_image(input,rotation_angle)
                    rotated_images.append(input_rotated)
                labels_list.append(labels)

            rotated_images = torch.stack(rotated_images)
            labels_list = torch.cat(labels_list)

            rotated_dataloader = torch.utils.data.DataLoader(list(zip(rotated_images, labels_list)), batch_size=test_dataloader.batch_size)

            print(f'Test mit rotation angle: {rotation_angle}')
            accuracy, precision, recall, f1 = ModelTester.test_model_robustness_rotation(model, rotated_dataloader, device)
            
            # Show a sample rotated image
            plt.figure(figsize=(12, 6))
            plt.title(f'Rotiertes Bild mit Rotation um: {rotation_angle}°.', fontsize=10)
            plt.imshow(rotated_images[0].permute(1, 2, 0))
            plt.text(1.2, 0.6, f'Genauigkeit: {accuracy}', horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)
            plt.text(1.2, 0.5, f'Präzision: {precision}', horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)
            plt.text(1.2, 0.4, f'Recall: {recall}', horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)
            plt.text(1.2, 0.3, f'F1-Score: {f1}', horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)
            plt.savefig(fr"{savefig_path}/{i}.png")
            print(f"test/test-plots-rotation/{i}.png")
            # plt.show()
            i += 1
            rotation_angle += step


class Main_Model_Test(ModelTester):
    def rotate_and_convert(angle, test_images = "data/train-test-data/test/", save_dir="rotated_images/"):
        from PIL import Image
        import os
        i = 0
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for filename in os.listdir(test_images):
            if angle == 360: 
                break
            else:
                image_path = os.path.join(test_images, filename)
                image = Image.open(image_path)
                rotated_image = image.rotate(angle)
                save_path = os.path.join(save_dir, f"rotated_{angle}_{filename}")
                rotated_image.save(save_path)
                
            angle += 10

    def run_tests():
        IMAGE_SIZE = (178, 218)
        BATCH_SIZE_FILE = 'test/epochs/batch_size.txt'
        TRAIN_DIR = 'data/train-test-data/train'
        TEST_DIR = 'data/train-test-data/test'
        MODEL_PATH = 'test/model_to_be_tested'
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_tester = ModelTester()    
        # Define the path to the model
        model_path =  model_tester.get_model_path(MODEL_PATH) # Replace with your actual model path

        # Check if the model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No model found at {model_path}")

        # Initialize the model
        model = SimpleCNN()  # Replace SimpleCNN with your actual model class

        # Load the state dict
        state_dict = torch.load(model_path)

        # Apply the state dict to the model
        model.load_state_dict(state_dict)

        # Move the model to the correct device
        model = model.to(device)

        if os.path.exists(BATCH_SIZE_FILE):
            with open(BATCH_SIZE_FILE, 'r') as f:
                batch_size = int(f.read())
        else:
            batch_size = 64


        transform =  transform = transforms.Compose([
            transforms.Resize((178, 218)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        _,test_dataloader = DataLoaderModelTrain.load_data(train_dir=TRAIN_DIR, test_dir=TEST_DIR, transform=transform, batch_size=batch_size)


        # ModelTester.evaluate_model(model, test_dataloader)
        # ModelTester.test_model_robustness(model, test_dataloader, device)
        # ModelTester.add_noise_and_test(model, test_dataloader, device)
        # ModelTester.test_noise_robustness(model, test_dataloader, device, end_noise=2.0, step = 0.01)
        # ModelTester.test_distortion_robustness(model, test_dataloader, device, end_distortion=2.0, step = 0.001)
        ModelTester.test_rotation_robustness(model, test_dataloader, device, end_angle=270.0, step = 40.0)  
Main_Model_Test.run_tests()