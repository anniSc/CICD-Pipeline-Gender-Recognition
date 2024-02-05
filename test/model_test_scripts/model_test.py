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
import torchvision
import random
from fairlearn.metrics import (
    MetricFrame,
    count,
    false_negative_rate,
    false_positive_rate,
    selection_rate,
)

from torchvision.transforms.functional import normalize, resize


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

            print(f'::warning::Test mit Verrauschungs-Faktor: {noise_factor}')
            accuracy, precision,recall, f1 = ModelTester.test_model_robustness(model, noisy_dataloader, device)
            

            # Show a sample noisy image
            plt.figure(figsize=(12, 6))
            plt.title(f'Verauschtes Bild mit Verrauschungs-Faktor: {noise_factor}.', fontsize=10)
            plt.imshow(noisy_images[0].permute(1, 2, 0))
            plt.text(1.1, 0.6, f'Genauigkeit: {accuracy}', horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)
            plt.text(1.1, 0.5, f'Precision: {precision}', horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)
            plt.text(1.1, 0.4, f'Recall: {recall}', horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)
            plt.text(1.1, 0.3, f'F1 Score: {f1}', horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)
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
                print(f'::warning:: Genauigkeit unter 70% mit Verzerrungs-Faktor: {distortion_factor}. Test wird gestoppt.')
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

            print(f'Test mit: {rotation_angle}° (Grad)')
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

class TestFairness():
    train_men = "data/train-test-data/train/men"
    train_women = "data/train-test-data/train/women"
    sensitive_features = ["men","women"]
    train = "data/train-test-data/train"
    test = "data/train-test-data/test"
    merged_csv = "test/csv/gender_labelled.csv" 

    transform = transforms.Compose([
            transforms.Resize((178, 218)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    transform = transforms.Compose([transforms.ToTensor()])
    
  
    def get_sensitive_features(merged_csv, label_men=1, label_women=-1):
        df = pd.read_csv(merged_csv)
        sensitive_features = df['Male'].tolist()
        sensitive_features = pd.Series(sensitive_features).replace({label_women: 'Frau', label_men: 'Mann'}).tolist()
        return sensitive_features
    

    def get_fairness_metrics(merged_csv,train_dataloader, model, transform, sensitive_features):
        y_test = []
        y_pred = []
        
        for inputs, labels in train_dataloader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_test.extend(labels.numpy())
            y_pred.extend(preds.numpy())
        
        metrics = MetricFrame(metrics=accuracy_score, y_true=y_test, y_pred=y_pred, sensitive_features=sensitive_features)
        return metrics, y_test, y_pred


    def create_gender_labelled_csv(men_folder, women_folder, output_csv, label_men = 1, label_women=-1):
        men_files = os.listdir(men_folder)
        women_files = os.listdir(women_folder)
    
        men_df = pd.DataFrame({
            'filename': men_files,
            'Male': [label_men]*len(men_files)  # 1 für Männer
        })
        women_df = pd.DataFrame({
            'filename': women_files,
            'Male': [label_women]*len(women_files)  # -1 für Frauen
        })

        combined_df = pd.concat([men_df, women_df])
        
        combined_df.to_csv(output_csv, index=False)
        return os.path.abspath(output_csv)

    def plot_bar_fairnesscheck(groups, accuracies,metrics):
        groups = metrics.by_group.index.tolist()
        accuracies = metrics.by_group.values.tolist()
        plt.bar(groups, accuracies)
        plt.title('Genauigkeit von Gruppen')
        plt.xlabel('Gruppe')
        plt.ylabel('Genauigkeit')
        # plt.show()
        plt.savefig("test/metrics/plot_bar.jpg", dpi=100)

    def analyze_metrics(sensitive_features, y_test, y_pred):
        from fairlearn.metrics import (
            MetricFrame,
            count,
            false_negative_rate,
            false_positive_rate,
            selection_rate,
        )
    
        metrics = {
            "accuracy": accuracy_score,
            "precision": precision_score,
            "false positive rate": false_positive_rate,
            "false negative rate": false_negative_rate,
            "selection rate": selection_rate,
            "count": count,
        }
        print(y_test)
        metric_frame = MetricFrame(
            metrics=metrics, y_true=y_test, y_pred=y_pred, sensitive_features=sensitive_features
        )

        ax = metric_frame.by_group.plot.bar(
            subplots=True,
            layout=[3, 3],
            legend=False,
            figsize=[12, 8],
            title="Metriken!",
        )

        for row in ax:
            for subplot in row:
                subplot.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        
        plt.savefig("test/metrics/metrics.jpg", dpi=100)
        # plt.show()
        return metric_frame
        # Die Metriken, die auf dem Bild erstellt werden, sind:
        # - accuracy (Genauigkeit): Gibt den Anteil der korrekt vorhergesagten Werte an.
        # - precision (Präzision): Gibt den Anteil der korrekt vorhergesagten positiven Werte an.
        # - false positive rate (Falsch-Positiv-Rate): Gibt den Anteil der falsch vorhergesagten positiven Werte an.
        # - false negative rate (Falsch-Negativ-Rate): Gibt den Anteil der falsch vorhergesagten negativen Werte an.
        # - selection rate (Auswahlrate): Gibt den Anteil der ausgewählten Werte an.
        # - count (Anzahl): Gibt die Anzahl der Werte an.
        


    def run_fairness_tests(train_dataloader,model,transform):
        merged_csv = TestFairness.create_gender_labelled_csv(TestFairness.train_men,TestFairness.train_women,TestFairness.merged_csv)

        metrics, y_test, y_pred =  TestFairness.get_fairness_metrics(merged_csv, train_dataloader, model, transform, TestFairness.get_sensitive_features(merged_csv))
        groups = metrics.by_group.index.tolist()
        accuracies = metrics.by_group.values.tolist()
        TestFairness.plot_bar_fairnesscheck(groups, accuracies,metrics)
        metric_frame = TestFairness.analyze_metrics(sensitive_features=TestFairness.get_sensitive_features(merged_csv=merged_csv), y_test=y_test, y_pred=y_pred)


        import matplotlib.pyplot as plt

        fig1 = metric_frame.by_group.plot(
            kind="bar",
            ylim=[0, 2],
            subplots=True,
            layout=[3, 3],
            legend=False,
            figsize=[12, 8],
            title="Alle Metriken mit zugewiesenem y-Achsenbereich anzeigen",
        )

        fig2 = metric_frame.by_group[["count"]].plot(
            kind="pie",
            subplots=True,
            layout=[1, 1],
            legend=True,
            figsize=[12, 8],
            labels=["Frau","Mann"],
            autopct="%.2f",
            title="Metriken als Kuchendiagramm",
        )


        fig1[0][0].figure.savefig("test/metricsFairlearn/Fig1metricsFairLearn.jpg", dpi=100)
        fig2[0][0].figure.savefig("test/metricsFairlearn/Fig2metricsFairLearn.jpg", dpi=100)



import os
import random
import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision.transforms.functional import normalize, resize

class ModelVisualizer:
    def __init__(self, model_path, target_layer):
        self.model = self.load_model(model_path)
        self.cam_extractor = GradCAMpp(self.model, target_layer)

    def load_model(self, model_path):
        model = SimpleCNN()
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model

    def get_image_paths(self, dir_path):
        return [os.path.join(dir_path, img) for img in os.listdir(dir_path)]

    def select_images(self, men_dir, women_dir):
        men_images = self.get_image_paths(men_dir)
        women_images = self.get_image_paths(women_dir)
        return random.sample(men_images, 1) + random.sample(women_dir, 1)

    def process_image(self, img_path):
        img = torchvision.io.read_image(img_path)
        return normalize(resize(img, (178, 218)) / 255., [0.485, 0.456, 0.406], [0.220, 0.224, 0.225])

    def visualize_model(self, men_dir = "data/train-test-data/train/men", women_dir = "data/train-test-data/train/women"):
        men_images = [os.path.join(men_dir, img) for img in os.listdir(men_dir)]
        women_images = [os.path.join(women_dir, img) for img in os.listdir(women_dir)]
        selected_images = random.sample(men_images, 1) + random.sample(women_images, 1)
        for i, img_path in enumerate(selected_images):
            img = torchvision.io.read_image(img_path)
            input_tensor = torchvision.transforms.functional.normalize(
                torchvision.transforms.functional.resize(img, (178, 218)) / 255.,
                [0.485, 0.456, 0.406],
                [0.220, 0.224, 0.225]
            )
            out = self.model(input_tensor.unsqueeze(0))
            activation_map = self.cam_extractor(1, out)
            activation_map = activation_map[0].squeeze(0).numpy()

            plt.imshow(activation_map, cmap='jet')
            plt.savefig(f'test/activation_map/activation_map_{i}.png')
            plt.show()



class Main_Model_Test(ModelTester, TestFairness,ModelVisualizer):
    def rotate_and_convert(angle, test_images = "data/train-test-data/test/", save_dir="rotated_images/"):
        from PIL import Image
        import os
        i = 0
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for filename in os.listdir(test_images):
            if angle == 360: 
                break
         
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
        model_path =  model_tester.get_model_path(MODEL_PATH) # Replace with your actual model path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No model found at {model_path}")
        model = SimpleCNN() 
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
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
        train_dataloader,test_dataloader = DataLoaderModelTrain.load_data(train_dir=TRAIN_DIR, test_dir=TEST_DIR, transform=transform, batch_size=batch_size)
        # ModelTester.evaluate_model(model, test_dataloader)
        # ModelTester.test_model_robustness(model, test_dataloader, device)
        # ModelTester.add_noise_and_test(model, test_dataloader, device)
        ModelTester.test_noise_robustness(model, test_dataloader, device, end_noise=2.0, step = 0.01)
        # ModelTester.test_distortion_robustness(model, test_dataloader, device, end_distortion=2.0, step = 0.001)
        # ModelTester.test_rotation_robustness(model, test_dataloader, device, end_angle=270.0, step = 90.0)
        # TestFairness.run_fairness_tests(train_dataloader, model, transform)
        # visualizer = ModelVisualizer(model_path, "conv2")
        # visualizer.visualize_model()

        



Main_Model_Test.run_tests()