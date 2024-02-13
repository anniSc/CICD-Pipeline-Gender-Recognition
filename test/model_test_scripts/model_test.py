import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from fairlearn.metrics import MetricFrame
from model_train import DataLoaderModelTrain, SimpleCNN
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from torchcam.methods import GradCAMpp
from torchvision import transforms
from torchvision.transforms.functional import normalize, resize
import sys 
sys.path.insert(0, 'model/model_scripts/model_train.py')
from model_train import SimpleCNN, SimpleCNN2, SimpleCNN3
import re


class ModelTester:
    """
    Eine Klasse, die Methoden zum Testen und Evaluieren eines Modells bereitstellt.
    """

    def test_model_robustness(model, test_dataloader, device):
        """
        Testet die Robustheit des Modells, indem es auf den Testdaten Vorhersagen trifft und Metriken berechnet.

        Args:
            model (torch.nn.Module): Das trainierte Modell.
            test_dataloader (torch.utils.data.DataLoader): Der DataLoader für die Testdaten.
            device (torch.device): Das Gerät (CPU oder GPU), auf dem die Berechnungen durchgeführt werden sollen.

        Returns:
            tuple: Ein Tupel mit den berechneten Metriken (Genauigkeit, Präzision, Recall, F1-Score).
        """
        model.eval()  # Setze das Modell in den Evaluationsmodus
        true_labels = []
        predicted_labels = []

        with torch.no_grad():  # Berechne keine Gradienten, um die Berechnung zu beschleunigen
            for inputs, labels in test_dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                output = model(inputs)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                predictions = torch.argmax(probabilities, dim=1)

                true_labels.extend(labels.cpu().numpy())
                predicted_labels.extend(predictions.cpu().numpy())

        # Berechne Metriken
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(
            true_labels,
            predicted_labels,
            average="weighted")
        recall = recall_score(
            true_labels,
            predicted_labels,
            average="weighted")
        f1 = f1_score(true_labels, predicted_labels, average="weighted")

        print(f"Genauigkeit: {accuracy}")
        print(f"Präzision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1-Score: {f1}")
        return accuracy, precision, recall, f1

    @staticmethod
    def get_model_path(directory):
        """
        Gibt den Pfad zur Modelldatei mit der Erweiterung ".pth" zurück, die sich im angegebenen Verzeichnis befindet.
        
        Args:
            directory (str): Das Verzeichnis, in dem nach der Modelldatei gesucht werden soll.
        
        Returns:
            str: Der vollständige Pfad zur Modelldatei, wenn eine gefunden wurde. Andernfalls None.
        """
        for filename in os.listdir(directory):
            if filename.endswith(".pth"):
                print(filename)
                return os.path.join(directory, filename)

        return None

    @staticmethod
    def add_noise(images, noise_factor=0.5):
        """
        Fügt Rauschen zu den Bildern hinzu.

        Args:
            images (torch.Tensor): Ein Tensor mit den Bildern.
            noise_factor (float, optional): Der Faktor für das Rauschen. Standardmäßig 0.5.

        Returns:
            torch.Tensor: Ein Tensor mit den rauschigen Bildern.
        """
        noise = torch.randn_like(images) * noise_factor
        noisy_images = images + noise
        noisy_images = torch.clamp(noisy_images, 0.0, 1.0)
        return noisy_images

    def add_noise_and_test_robustness(
            model, test_dataloader, device, noise_factor=0.5):
        """
        Fügt Rauschen zu den Eingabebildern hinzu und testet die Robustheit des Modells.

        Args:
            model (torch.nn.Module): Das zu testende Modell.
            test_dataloader (torch.utils.data.DataLoader): Der DataLoader für den Testdatensatz.
            device (torch.device): Das Gerät, auf dem das Modell ausgeführt wird.
            noise_factor (float, optional): Der Faktor für das Rauschen. Standardwert ist 0.5.

        Returns:
            None
        """
        noisy_images = []
        labels_list = []

        for inputs, labels in test_dataloader:
            inputs_noisy = ModelTester.add_noise(inputs, noise_factor)
            noisy_images.append(inputs_noisy)
            labels_list.append(labels)

        # Stapelt Bilder und Labels separat
        noisy_images = torch.cat(noisy_images)
        labels_list = torch.cat(labels_list)

        noisy_dataloader = torch.utils.data.DataLoader(
            list(zip(noisy_images, labels_list)), batch_size=test_dataloader.batch_size
        )

        ModelTester.test_model_robustness(model, noisy_dataloader, device)

    def evaluate_model(model, test_dataloader):
        """
        Bewertet das Modell anhand des Testdatensatzes und gibt die Metriken zurück.

        Args:
            model (torch.nn.Module): Das trainierte Modell.
            test_dataloader (torch.utils.data.DataLoader): Der DataLoader für den Testdatensatz.

        Returns:
            float: Die Genauigkeit des Modells.
            float: Die Präzision des Modells.
            float: Der Recall des Modells.
            float: Der F1-Score des Modells.
        """
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
        precision = precision_score(
            true_labels, predictions, average="weighted")
        recall = recall_score(true_labels, predictions, average="weighted")
        f1 = f1_score(true_labels, predictions, average="weighted")

        print(
            f"Genauigkeit: {accuracy}, Präzision: {precision}, Recall: {recall}, F1-Score: {f1}"
        )

        with open("test/metrics/metrics.txt", "w") as outfile:
            outfile.write(
                f"Modellmetriken: Genauigkeit: {accuracy}, Präzision: {precision}, Recall: {recall}, F1-Score: {f1}"
            )

        return accuracy, precision, recall, f1

    def test_predicts(model, test_dataloader, device):
        """
        Führt Vorhersagen für das gegebene Modell auf dem Testdatensatz durch.

        Args:
            model (torch.nn.Module): Das Modell, das für die Vorhersagen verwendet werden soll.
            test_dataloader (torch.utils.data.DataLoader): Der DataLoader, der den Testdatensatz enthält.
            device (torch.device): Das Gerät (GPU oder CPU), auf dem die Vorhersagen durchgeführt werden sollen.

        Returns:
            None
        """
        # Für jede Eingabe im Testdatensatz:
        for inputs, _ in test_dataloader:
            # Die Eingaben werden auf das Gerät verschoben (GPU oder CPU).
            inputs = inputs.to(device)
            # Das Modell macht eine Vorhersage auf den Eingaben.
            output = model(inputs)
            # Die Ausgabe des Modells wird in Wahrscheinlichkeiten umgewandelt,
            # indem die Softmax-Funktion angewendet wird.
            probabilities = torch.nn.functional.softmax(output, dim=1)
            # Die Klasse mit der höchsten Wahrscheinlichkeit wird als
            # Vorhersage ausgewählt.
            predictions = torch.argmax(probabilities, dim=1)
            # Die Vorhersagen werden in eine Liste umgewandelt und ausgegeben.
            predictions_list = predictions.cpu().numpy().tolist()
            print(predictions_list)

    def test_noise_robustness(
            model,
            test_dataloader,
            device,
            start_noise=0.0,
            end_noise=1.0,
            step=0.1,
            savefig_path="test/test-plots-rauschen",
        ):
        """
        Führt einen Test der Rauschrobustheit des Modells durch.

        Args:
            model (torch.nn.Module): Das zu testende Modell.
            test_dataloader (torch.utils.data.DataLoader): Der DataLoader für den Testdatensatz.
            device (torch.device): Das Gerät, auf dem der Test ausgeführt werden soll.
            start_noise (float, optional): Der Startwert für den Rauschfaktor. Standardmäßig 0.0.
            end_noise (float, optional): Der Endwert für den Rauschfaktor. Standardmäßig 1.0.
            step (float, optional): Der Schrittweite für den Rauschfaktor. Standardmäßig 0.1.
            savefig_path (str, optional): Der Pfad zum Speichern der generierten Plots. Standardmäßig "test/test-plots-rauschen".

        Returns:
            None
        """
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

            noisy_dataloader = torch.utils.data.DataLoader(
                list(zip(noisy_images, labels_list)),
                batch_size=test_dataloader.batch_size,
            )

            ModelTester.test_model_robustness(model, noisy_dataloader, device)

            print(f"::warning::Test mit Verrauschungs-Faktor: {noise_factor}")
            accuracy, precision, recall, f1 = ModelTester.test_model_robustness(
                model, noisy_dataloader, device
            )

            # Show a sample noisy image
            plt.figure(figsize=(12, 6))
            plt.title(
                f"Verauschtes Bild mit Verrauschungs-Faktor: {noise_factor}.",
                fontsize=10,
            )
            plt.imshow(noisy_images[0].permute(1, 2, 0))
            plt.text(
                1.1,
                0.6,
                f"Genauigkeit: {accuracy}",
                horizontalalignment="left",
                verticalalignment="center",
                transform=plt.gca().transAxes,
            )
            plt.text(
                1.1,
                0.5,
                f"Precision: {precision}",
                horizontalalignment="left",
                verticalalignment="center",
                transform=plt.gca().transAxes,
            )
            plt.text(
                1.1,
                0.4,
                f"Recall: {recall}",
                horizontalalignment="left",
                verticalalignment="center",
                transform=plt.gca().transAxes,
            )
            plt.text(
                1.1,
                0.3,
                f"F1 Score: {f1}",
                horizontalalignment="left",
                verticalalignment="center",
                transform=plt.gca().transAxes,
            )
            plt.savefig(f"{savefig_path}/{i}.png")
            # plt.show()

            if accuracy < 0.7:
                print(
                    f"::warning::Genauigkeit unter 70% mit noise factor: {noise_factor}. Test wird gestoppt."
                )
                plt.close()
                break
            i += 1
            noise_factor += step
            plt.close()

    def add_distortion(image, distortion_factor=0.5):
            """
            Fügt eine Verzerrung zum Bild hinzu.

            Args:
                image (torch.Tensor): Das Eingangsbild.
                distortion_factor (float, optional): Der Verzerrungsfaktor. Standardwert ist 0.5.

            Returns:
                torch.Tensor: Das verzerrte Bild.
            """
            startpoints = torch.tensor(
                [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
            endpoints = startpoints + torch.tensor(
                [
                    [0.0, distortion_factor],
                    [0.0, -distortion_factor],
                    [0.0, 0.0],
                    [0.0, 0.0],
                ]
            )
            distorted_image = TF.perspective(image, startpoints, endpoints)
            return distorted_image

    def test_distortion_robustness(
            model,
            test_dataloader,
            device,
            start_distortion=0.0,
            end_distortion=1.0,
            step=0.0001,
            savefig_path="test/test-plots-verzerrung",
        ):
            """
            Testet die Robustheit des Modells gegenüber Verzerrungen.

            Args:
                model (torch.nn.Module): Das zu testende Modell.
                test_dataloader (torch.utils.data.DataLoader): Der DataLoader für den Testdatensatz.
                device (torch.device): Das Gerät, auf dem das Modell ausgeführt wird.
                start_distortion (float, optional): Der Startwert für den Verzerrungsfaktor. Standardmäßig 0.0.
                end_distortion (float, optional): Der Endwert für den Verzerrungsfaktor. Standardmäßig 1.0.
                step (float, optional): Der Schrittweite für den Verzerrungsfaktor. Standardmäßig 0.0001.
                savefig_path (str, optional): Der Pfad zum Speichern der generierten Plots. Standardmäßig "test/test-plots-verzerrung".

            Returns:
                None
            """
            distortion_factor = start_distortion
            i = 0
            while distortion_factor <= end_distortion:
                distorted_images = []
                labels_list = []

                for inputs, labels in test_dataloader:
                    inputs_distorted = ModelTester.add_distortion(
                        inputs, distortion_factor)
                    distorted_images.append(inputs_distorted)
                    labels_list.append(labels)

                distorted_images = torch.cat(distorted_images)
                labels_list = torch.cat(labels_list)

                distorted_dataloader = torch.utils.data.DataLoader(
                    list(zip(distorted_images, labels_list)),
                    batch_size=test_dataloader.batch_size,
                )

                print(f"Test mit Verzerrungs-Faktor: {distortion_factor}")
                accuracy, precision, recall, f1 = ModelTester.test_model_robustness(
                    model, distorted_dataloader, device
                )

                # Show a sample distorted image
                plt.figure(figsize=(12, 6))
                plt.title(
                    f"Verzerrtes Bild mit Verzerrungs-Faktor: {distortion_factor}.",
                    fontsize=10,
                )
                plt.imshow(distorted_images[0].permute(1, 2, 0))
                plt.text(
                    1.2,
                    0.6,
                    f"Genauigkeit: {accuracy}",
                    horizontalalignment="left",
                    verticalalignment="center",
                    transform=plt.gca().transAxes,
                )
                plt.text(
                    1.2,
                    0.5,
                    f"Präzsion: {precision}",
                    horizontalalignment="left",
                    verticalalignment="center",
                    transform=plt.gca().transAxes,
                )
                plt.text(
                    1.2,
                    0.4,
                    f"Recall: {recall}",
                    horizontalalignment="left",
                    verticalalignment="center",
                    transform=plt.gca().transAxes,
                )
                plt.text(
                    1.2,
                    0.3,
                    f"F1-Score: {f1}",
                    horizontalalignment="left",
                    verticalalignment="center",
                    transform=plt.gca().transAxes,
                )
                plt.savefig(f"{savefig_path}/{i}.png")
                # print(f"test/test-plots-verzerrung/{i}.png")
                # plt.show()
                plt.clf()
                if accuracy < 0.7:
                    print(
                        f"::warning:: Genauigkeit unter 70% mit Verzerrungs-Faktor: {distortion_factor}. Test wird gestoppt."
                    )
                    plt.clf()
                    break
                i += 1
                distortion_factor += step
                plt.clf()

    def rotate_and_convert(
            image, angle, test_images="data/train-test-data/test"):
        """
        Dreht das Bild um den angegebenen Winkel und konvertiert es in ein anderes Format.

        :param image: Das Bild, das gedreht und konvertiert werden soll.
        :type image: PIL.Image
        :param angle: Der Winkel, um den das Bild gedreht werden soll.
        :type angle: float
        :param test_images: Der Pfad zum Ordner mit den Testbildern.
        :type test_images: str
        """
        from PIL import Image

        for filename in os.listdir(test_images):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image_path = os.path.join(test_images, filename)
                image = Image.rotate(image, angle)

    def test_model_robustness_rotation(model, test_dataloader, device):
            """
            Testet die Robustheit des Modells gegenüber Rotationen.

            Args:
                model (torch.nn.Module): Das zu testende Modell.
                test_dataloader (torch.utils.data.DataLoader): Der DataLoader für den Testdatensatz.
                device (torch.device): Das Gerät, auf dem die Berechnungen durchgeführt werden sollen.

            Returns:
                tuple: Ein Tupel mit den Metriken Accuracy, Precision, Recall und F1-Score.
            """
            model.eval()  # Setze das Modell in den Evaluationsmodus
            true_labels = []
            predicted_labels = []

            with torch.no_grad():  # Berechnung der Gradienten und deaktivieren, um die Berechnung zu beschleunigen
                for inputs, labels in test_dataloader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    output = model(inputs)
                    probabilities = torch.nn.functional.softmax(output, dim=1)
                    predictions = torch.argmax(probabilities, dim=1)

                    true_labels.extend(labels.cpu().numpy())
                    predicted_labels.extend(predictions.cpu().numpy())

            # Berechnung der Metriken
            accuracy = accuracy_score(true_labels, predicted_labels)
            precision = precision_score(
                true_labels, predicted_labels, average="weighted", zero_division=1
            )
            recall = recall_score(
                true_labels,
                predicted_labels,
                average="weighted")
            f1 = f1_score(true_labels, predicted_labels, average="weighted")

            return accuracy, precision, recall, f1



    def rotate_tensor_image(image_tensor, angle):
            """
            Rotiert ein Bildtensor um den angegebenen Winkel.

            Args:
                image_tensor (Tensor): Der Bildtensor, der rotiert werden soll.
                angle (float): Der Rotationswinkel in Grad.

            Returns:
                Tensor: Der rotierte Bildtensor.
            """
            rotated_image = TF.rotate(image_tensor, angle)
            return rotated_image

    def test_rotation_robustness(
        model,
        test_dataloader,
        device,
        start_angle=0.0,
        end_angle=270.0,
        step=90.0,
        savefig_path="test/test-plots-rotation",
    ):
        """
        Testet die Robustheit des Modells gegenüber Rotationen.

        Args:
            model (nn.Module): Das zu testende Modell.
            test_dataloader (DataLoader): Der DataLoader für den Testdatensatz.
            device (str): Das Gerät, auf dem die Berechnungen durchgeführt werden sollen.
            start_angle (float, optional): Der Startwinkel für die Rotation. Standardmäßig 0.0.
            end_angle (float, optional): Der Endwinkel für die Rotation. Standardmäßig 270.0.
            step (float, optional): Der Schritt für die Rotation. Standardmäßig 90.0.
            savefig_path (str, optional): Der Pfad zum Speichern der generierten Plots. Standardmäßig "test/test-plots-rotation".

        Returns:
            None
        """
        rotation_angle = start_angle
        i = 0
        while rotation_angle <= end_angle:
            rotated_images = []
            labels_list = []

            for inputs, labels in test_dataloader:
                for input in inputs:
                    input_rotated = ModelTester.rotate_tensor_image(
                        input, rotation_angle
                    )
                    rotated_images.append(input_rotated)
                labels_list.append(labels)

            rotated_images = torch.stack(rotated_images)
            labels_list = torch.cat(labels_list)

            rotated_dataloader = torch.utils.data.DataLoader(
                list(zip(rotated_images, labels_list)),
                batch_size=test_dataloader.batch_size,
            )

            print(f"Test mit: {rotation_angle}° (Grad)")
            (
                accuracy,
                precision,
                recall,
                f1,
            ) = ModelTester.test_model_robustness_rotation(
                model, rotated_dataloader, device
            )

            # Zeige ein Beispielbild nach der Rotation
            # Dataloader für das rotiertes Bild erstellen
            rotated_dataloader = torch.utils.data.DataLoader(
                list(zip(rotated_images, labels_list)),
                batch_size=test_dataloader.batch_size,
            )

            # Model auf das rotierte Bild testen
            print(f"Test mit: {rotation_angle}° (Grad)")
            (
                accuracy,
                precision,
                recall,
                f1,
            ) = ModelTester.test_model_robustness_rotation(
                model, rotated_dataloader, device
            )

            
            plt.figure(figsize=(12, 6))
            plt.title(
                f"Rotiertes Bild mit Rotation um: {rotation_angle}°.", fontsize=10
            )
            plt.imshow(rotated_images[0].permute(1, 2, 0))
            plt.text(
                1.2,
                0.6,
                f"Genauigkeit: {accuracy}",
                horizontalalignment="left",
                verticalalignment="center",
                transform=plt.gca().transAxes,
            )
            plt.text(
                1.2,
                0.5,
                f"Präzision: {precision}",
                horizontalalignment="left",
                verticalalignment="center",
                transform=plt.gca().transAxes,
            )
            plt.text(
                1.2,
                0.4,
                f"Recall: {recall}",
                horizontalalignment="left",
                verticalalignment="center",
                transform=plt.gca().transAxes,
            )
            plt.text(
                1.2,
                0.3,
                f"F1-Score: {f1}",
                horizontalalignment="left",
                verticalalignment="center",
                transform=plt.gca().transAxes,
            )
            plt.savefig(rf"{savefig_path}/{i}.png")
            print(f"test/test-plots-rotation/{i}.png")
            plt.clf()

            # Check accuracy threshold and break if below 70%
            if accuracy < 0.7:
                print(
                    f"::warning:: Genauigkeit unter 70% mit Rotations-Faktor: {rotation_angle}. Test wird gestoppt."
                )
                plt.clf()
                break

            i += 1
            rotation_angle += step
            plt.clf()


class TestFairness:
    """
    Eine Klasse, die Methoden zur Durchführung von Fairness-Tests für ein Modell bereitstellt.

    Attribute:
        train_men (str): Der Pfad zum Ordner mit den Trainingsdaten für Männer.
        train_women (str): Der Pfad zum Ordner mit den Trainingsdaten für Frauen.
        sensitive_features (list): Eine Liste der sensiblen Merkmale.
        train (str): Der Pfad zum Trainingsdatensatz.
        test (str): Der Pfad zum Testdatensatz.
        merged_csv (str): Der Pfad zur kombinierten CSV-Datei.

    Methoden:
        get_sensitive_features(merged_csv, label_men=1, label_women=-1): Gibt die sensiblen Merkmale zurück.
        get_fairness_metrics(merged_csv, train_dataloader, model, transform, sensitive_features): Berechnet die Fairness-Metriken.
        create_gender_labelled_csv(men_folder, women_folder, output_csv, label_men=1, label_women=-1): Erstellt eine CSV-Datei mit den gelabelten Daten.
        plot_bar_fairnesscheck(groups, accuracies, metrics): Erstellt ein Balkendiagramm der Fairness-Metriken.
        analyze_metrics(sensitive_features, y_test, y_pred): Analysiert die Fairness-Metriken.
        clear_file(file_path): Löscht den Inhalt einer Datei.
        run_fairness_tests(train_dataloader, model, transform): Führt die Fairness-Tests aus.
    """
    train_men = "data/train-test-data/train/men"
    train_women = "data/train-test-data/train/women"
    sensitive_features = ["men", "women"]
    train = "data/train-test-data/train"
    test = "data/train-test-data/test"
    merged_csv = "test/csv/gender_labelled.csv"

    transform = transforms.Compose(
        [
            transforms.Resize((178, 218)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    transform = transforms.Compose([transforms.ToTensor()])

    def get_sensitive_features(merged_csv, label_men=1, label_women=-1):
        """
        Gibt die sensiblen Merkmale zurück.

        :param merged_csv: Der Pfad zur kombinierten CSV-Datei.
        :type merged_csv: str
        :param label_men: Der Label-Wert für Männer, defaults to 1
        :type label_men: int, optional
        :param label_women: Der Label-Wert für Frauen, defaults to -1
        :type label_women: int, optional
        :return: Eine Liste der sensiblen Merkmale
        :rtype: list
        """
        df = pd.read_csv(merged_csv)
        sensitive_features = df["Male"].tolist()
        sensitive_features = (
            pd.Series(sensitive_features)
            .replace({label_women: "Frau", label_men: "Mann"})
            .tolist()
        )
        return sensitive_features

    def get_fairness_metrics(
        merged_csv, train_dataloader, model, transform, sensitive_features
    ):
        """
        Berechnet die Fairness-Metriken.

        :param merged_csv: Der Pfad zur kombinierten CSV-Datei.
        :type merged_csv: str
        :param train_dataloader: Der Trainings-Dataloader.
        :type train_dataloader: DataLoader
        :param model: Das Modell.
        :type model: nn.Module
        :param transform: Die Transformationen für die Eingabedaten.
        :type transform: torchvision.transforms.Compose
        :param sensitive_features: Die sensiblen Merkmale.
        :type sensitive_features: list
        :return: Die berechneten Metriken, die wahren Labels und die vorhergesagten Labels.
        :rtype: tuple
        """
        y_test = []
        y_pred = []

        for inputs, labels in train_dataloader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_test.extend(labels.numpy())
            y_pred.extend(preds.numpy())

        metrics = MetricFrame(
            metrics=accuracy_score,
            y_true=y_test,
            y_pred=y_pred,
            sensitive_features=sensitive_features,
        )
        print(metrics)
        return metrics, y_test, y_pred

    def create_gender_labelled_csv(
        men_folder, women_folder, output_csv, label_men=1, label_women=-1
    ):
        """
        Erstellt eine CSV-Datei mit den gelabelten Daten.

        :param men_folder: Der Pfad zum Ordner mit den Trainingsdaten für Männer.
        :type men_folder: str
        :param women_folder: Der Pfad zum Ordner mit den Trainingsdaten für Frauen.
        :type women_folder: str
        :param output_csv: Der Pfad zur Ausgabedatei.
        :type output_csv: str
        :param label_men: Der Label-Wert für Männer, defaults to 1
        :type label_men: int, optional
        :param label_women: Der Label-Wert für Frauen, defaults to -1
        :type label_women: int, optional
        :return: Der absolute Pfad zur Ausgabedatei.
        :rtype: str
        """
        men_files = os.listdir(men_folder)
        women_files = os.listdir(women_folder)

        men_df = pd.DataFrame(
            {
                "filename": men_files,
                "Male": [label_men] * len(men_files),  # 1 für Männer
            }
        )
        women_df = pd.DataFrame(
            {
                "filename": women_files,
                "Male": [label_women] * len(women_files),  # -1 für Frauen
            }
        )

        combined_df = pd.concat([men_df, women_df])

        combined_df.to_csv(output_csv, index=False)
        return os.path.abspath(output_csv)

    def plot_bar_fairnesscheck(groups, accuracies, metrics):
        """
        Erstellt ein Balkendiagramm der Fairness-Metriken.

        :param groups: Die Gruppen.
        :type groups: list
        :param accuracies: Die Genauigkeiten.
        :type accuracies: list
        :param metrics: Die Metriken.
        :type metrics: MetricFrame
        """
        groups = metrics.by_group.index.tolist()
        accuracies = metrics.by_group.values.tolist()
        plt.bar(groups, accuracies)
        plt.title("Genauigkeit von Gruppen")
        plt.xlabel("Gruppe")
        plt.ylabel("Genauigkeit")
        plt.savefig("test/metrics/plot_bar.jpg", dpi=100)
        plt.clf()

    def analyze_metrics(sensitive_features, y_test, y_pred):
        """
        Analysiert die Fairness-Metriken.

        :param sensitive_features: Die sensiblen Merkmale.
        :type sensitive_features: list
        :param y_test: Die wahren Labels.
        :type y_test: list
        :param y_pred: Die vorhergesagten Labels.
        :type y_pred: list
        :return: Das Metriken-Framework.
        :rtype: MetricFrame
        """
        from fairlearn.metrics import (MetricFrame, count, false_negative_rate,
                                       false_positive_rate, selection_rate)

        metrics = {
            "accuracy": accuracy_score,
            "precision": precision_score,
            "false positive rate": false_positive_rate,
            "false negative rate": false_negative_rate,
            "selection rate": selection_rate,
            "count": count,
        }

        metric_frame = MetricFrame(
            metrics=metrics,
            y_true=y_test,
            y_pred=y_pred,
            sensitive_features=sensitive_features,
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
        plt.clf()

        return metric_frame

    def clear_file(file_path):
        """
        Löscht den Inhalt einer Datei.

        :param file_path: Der Pfad zur Datei, deren Inhalt gelöscht werden soll.
        :type file_path: str
        """
        with open(file_path, "w") as outfile:
            outfile.write("")

    def run_fairness_tests(train_dataloader, model, transform):
        """
        Führt die Fairness-Tests aus.

        :param train_dataloader: Der Trainings-Dataloader.
        :type train_dataloader: DataLoader
        :param model: Das Modell.
        :type model: nn.Module
        :param transform: Die Transformationen für die Eingabedaten.
        :type transform: torchvision.transforms.Compose
        """
        merged_csv = TestFairness.create_gender_labelled_csv(
            TestFairness.train_men, TestFairness.train_women, TestFairness.merged_csv
        )
        metrics, y_test, y_pred = TestFairness.get_fairness_metrics(
            merged_csv,
            train_dataloader,
            model,
            transform,
            TestFairness.get_sensitive_features(merged_csv),
        )
        accuracy_per_group = metrics.by_group

        TestFairness.clear_file("test/metrics/fairness_metrics.txt")
        for group, accuracy in accuracy_per_group.items():
            print(f"Genauigkeit für die Gruppe: {group}: {accuracy}")
            with open("test/metrics/fairness_metrics.txt", "a") as outfile:
                outfile.write(
                    f"Genauigkeit für die Gruppe: {group}: {accuracy} \n")

        groups = metrics.by_group.index.tolist()
        accuracies = metrics.by_group.values.tolist()
        TestFairness.plot_bar_fairnesscheck(groups, accuracies, metrics)
        metric_frame = TestFairness.analyze_metrics(
            sensitive_features=TestFairness.get_sensitive_features(
                merged_csv=merged_csv
            ),
            y_test=y_test,
            y_pred=y_pred,
        )

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
            labels=["Frau", "Mann"],
            autopct="%.2f",
            title="Metriken als Kuchendiagramm",
        )

        fig1[0][0].figure.savefig(
            "test/metricsFairlearn/Fig1metricsFairLearn.jpg", dpi=100
        )
        fig2[0][0].figure.savefig(
            "test/metricsFairlearn/Fig2metricsFairLearn.jpg", dpi=100
        )
        fig1[0][0].figure.clf()
        fig2[0][0].figure.clf()


class ModelExplainability:
    def __init__(self, model_path, target_layer, model):
        """
        Initialisiert das ModelExplainability-Objekt.

        :param model_path: Der Pfad zum Modell.
        :type model_path: str
        :param target_layer: Die Ziel-Ebene des Modells.
        :type target_layer: str
        :param model: Das Modellobjekt.
        :type model: Model
        """
        self.model = model
        self.cam_extractor_grad = GradCAMpp(self.model, target_layer)

    def get_image_paths(self, dir_path):
        """
        Gibt eine Liste der Bildpfade im angegebenen Verzeichnis zurück.

        :param dir_path: Der Pfad zum Verzeichnis.
        :type dir_path: str
        :return: Eine Liste der Bildpfade.
        :rtype: list[str]
        """
        return [os.path.join(dir_path, img) for img in os.listdir(dir_path)]

    def select_images(self, men_dir, women_dir):
        """
        Wählt zufällig ein Bild aus dem Männerverzeichnis und ein Bild aus dem Frauenverzeichnis aus.

        :param men_dir: Der Pfad zum Männerverzeichnis.
        :type men_dir: str
        :param women_dir: Der Pfad zum Frauenverzeichnis.
        :type women_dir: str
        :return: Eine Liste der ausgewählten Bildpfade.
        :rtype: list[str]
        """
        men_images = self.get_image_paths(men_dir)
        women_images = self.get_image_paths(women_dir)
        return random.sample(men_images, 1) + random.sample(women_images, 1)

    def process_image(self, img_path):
        """
        Verarbeitet das Bild anhand des angegebenen Bildpfads.

        :param img_path: Der Pfad zum Bild.
        :type img_path: str
        :return: Das verarbeitete Bild.
        :rtype: torch.Tensor
        """
        img = torchvision.io.read_image(img_path)
        return normalize(
            resize(img, (178, 218)) / 255.0,
            [0.485, 0.456, 0.406],
            [0.220, 0.224, 0.225],
        )

    def visualize_model_grad(
        self,
        men_dir="data/train-test-data/train/men",
        women_dir="data/train-test-data/train/women",
    ):
        """
        Visualisiert das Modell anhand ausgewählter Bilder aus den Männer- und Frauenverzeichnissen.

        :param men_dir: Der Pfad zum Männerverzeichnis.
        :type men_dir: str
        :param women_dir: Der Pfad zum Frauenverzeichnis.
        :type women_dir: str
        """
        men_images = [os.path.join(men_dir, img)
                      for img in os.listdir(men_dir)]
        women_images = [os.path.join(women_dir, img)
                        for img in os.listdir(women_dir)]
        selected_images = random.sample(
            men_images, 1) + random.sample(women_images, 1)
        
        for i, img_path in enumerate(selected_images):
            img = torchvision.io.read_image(img_path)
            input_tensor = torchvision.transforms.functional.normalize(
                torchvision.transforms.functional.resize(
                    img, (178, 218)) / 255.0,
                [0.485, 0.456, 0.406],
                [0.220, 0.224, 0.225],
            )
            out = self.model(input_tensor.unsqueeze(0))
            activation_map = self.cam_extractor_grad(1, out)
            activation_map = activation_map[0].squeeze(0).numpy()
            plt.close()
            plt.imshow(activation_map, cmap="jet")
            plt.savefig(f"test/activation_map/activation_map_{i}.png")
            plt.clf()


class Main_Model_Test(ModelTester, TestFairness, ModelExplainability):
    
    def extract_model_name(full_path):
        """
        Extrahiert den Modellnamen aus dem angegebenen vollständigen Pfad.

        Args:
            full_path (str): Der vollständige Pfad, aus dem der Modellname extrahiert werden soll.

        Returns:
            str: Der extrahierte Modellname.
        """
        filename = os.path.basename(full_path)
        return filename.split("_")[0]
    
    def run_tests():
        """
        Führt verschiedene Tests auf dem Modell aus.

        Diese Funktion lädt das Modell, führt verschiedene Tests auf dem Modell durch
        und visualisiert die Ergebnisse. Es werden Tests zur Modellbewertung, Robustheit,
        Fairness und Erklärbarkeit durchgeführt.

        Args:
            None

        Returns:
            None
        """
        IMAGE_SIZE = (178, 218)
        BATCH_SIZE_FILE = "test/epochs/batch_size.txt"
        TRAIN_DIR = "data/train-test-data/train"
        TEST_DIR = "data/train-test-data/test"
        MODEL_PATH = "test/model_to_be_tested"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_tester = ModelTester()
        model_path = model_tester.get_model_path(
            MODEL_PATH
        )  # Ersetzen Sie dies durch Ihren tatsächlichen Modellpfad
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Kein Modell gefunden unter {model_path}")
        
        print(model_path)
        model_name = Main_Model_Test.extract_model_name(model_path)
        print(model_name)

        model_classes = {
            "SimpleCNN": SimpleCNN,
            "SimpleCNN2": SimpleCNN2,
            "SimpleCNN3": SimpleCNN3
        } 
        model_name = Main_Model_Test.extract_model_name(model_path)
        if model_name in model_classes:
            model = model_classes[model_name]()
        else:
            print(f"Unbekannter Modellname: {model_name}")
            model = None

        if model is not None:
            state_dict = torch.load(model_path)
            model.load_state_dict(state_dict)
        
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
        model = model.to(device)
        if os.path.exists(BATCH_SIZE_FILE):
            with open(BATCH_SIZE_FILE, "r") as f:
                batch_size = int(f.read())
        else:
            batch_size = 32
        transform = transform = transforms.Compose(
            [
                transforms.Resize((178, 218)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        train_dataloader, test_dataloader = DataLoaderModelTrain.load_data(
            train_dir=TRAIN_DIR,
            test_dir=TEST_DIR,
            transform=transform,
            batch_size=batch_size,
        )
        ModelTester.evaluate_model(model, test_dataloader)
        ModelTester.test_model_robustness(model, test_dataloader, device)
        ModelTester.test_noise_robustness(
            model, test_dataloader, device, end_noise=1.0, step=0.1
        )
        ModelTester.test_distortion_robustness(
            model, test_dataloader, device, end_distortion=0.05, step=0.0001
        )
        ModelTester.test_rotation_robustness(
            model, test_dataloader, device, end_angle=270.0, step=10.0
        )
        TestFairness.run_fairness_tests(train_dataloader, model, transform)
        
        ModelExplainability(model_path, "conv2", model=model).visualize_model_grad()


Main_Model_Test.run_tests()
