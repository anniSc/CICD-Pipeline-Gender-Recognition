import glob
import os
from datetime import datetime

import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch import nn
from torchvision import datasets, transforms
from tqdm import tqdm
import psutil 
import matplotlib.pyplot as plt
import time

class SimpleCNN(nn.Module):
    def __init__(self):
        self.name = "SimpleCNN"
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(33456, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class SimpleCNN2(nn.Module):    
    """Ein einfaches CNN-Modell zur Klassifikation von Bildern.
    
    Dieses Modell besteht aus mehreren Convolutional- und Fully Connected-Schichten.
    Es wird verwendet, um Bilder in zwei Kategorien zu klassifizieren.
    
    Attributes:
        name (str): Der Name des Modells.
        conv1 (nn.Conv2d): Die erste Convolutional-Schicht.
        pool (nn.MaxPool2d): Die Max Pooling-Schicht.
        conv2 (nn.Conv2d): Die zweite Convolutional-Schicht.
        fc1 (nn.Linear): Die erste Fully Connected-Schicht.
        fc2 (nn.Linear): Die zweite Fully Connected-Schicht.
        fc3 (nn.Linear): Die dritte Fully Connected-Schicht.
        dropout (nn.Dropout): Die Dropout-Schicht.
    """
        
    def __init__(self):
        self.name = "SimpleCNN2"
        super(SimpleCNN2, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(33456, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


class SimpleCNN3(nn.Module):    
    def __init__(self):
        """
        Einfache CNN-Architektur mit 3 Convolutional-Schichten.

        Args:
            None

        Returns:
            None
        """
        self.name = "SimpleCNN3"
        super(SimpleCNN3, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(66912, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)
        self.dropout = nn.Dropout(0.5)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)

    def forward(self, x):
        """
        Führt die Vorwärtsberechnung des Modells durch.

        Args:
            x (torch.Tensor): Eingabetensor mit den Dimensionen (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Ausgabetensor mit den Dimensionen (batch_size, num_classes).
        """
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

class Trainer:
    """
    Eine Klasse, die den Trainingsprozess für ein PyTorch-Modell verwaltet.

    Args:
        model (torch.nn.Module): Das zu trainierende PyTorch-Modell.
        train_dataloader (torch.utils.data.DataLoader): Der Datenlader für den Trainingsdatensatz.
        test_dataloader (torch.utils.data.DataLoader): Der Datenlader für den Test-/Validierungsdatensatz.
        epochs (int): Die Anzahl der Epochen, um das Modell zu trainieren.
        batch_size (int): Die Batch-Größe für das Training.

    Attribute:
        model (torch.nn.Module): Das zu trainierende PyTorch-Modell.
        train_dataloader (torch.utils.data.DataLoader): Der Datenlader für den Trainingsdatensatz.
        test_dataloader (torch.utils.data.DataLoader): Der Datenlader für den Test-/Validierungsdatensatz.
        epochs (int): Die Anzahl der Epochen, um das Modell zu trainieren.
        batch_size (int): Die Batch-Größe für das Training.
        criterion (torch.nn.Module): Die Verlustfunktion für das Training.
        optimizer (torch.optim.Optimizer): Der Optimierer zum Aktualisieren der Modellparameter.
        patience (int): Die Anzahl der Epochen, um auf eine Verbesserung der Validierungsgenauigkeit zu warten, bevor das Training abgebrochen wird.
        best_accuracy (float): Die beste Validierungsgenauigkeit, die während des Trainings erreicht wurde.
        early_stopping_counter (int): Der Zähler zur Verfolgung der Anzahl der Epochen ohne Verbesserung der Validierungsgenauigkeit.
    """

    def __init__(self, model, train_dataloader,
                 test_dataloader, epochs, batch_size):
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.epochs = epochs
        self.batch_size = batch_size
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        self.patience = 10
        self.best_accuracy = 0.0
        self.early_stopping_counter = 0

    def train(self):
        now = datetime.now()
        formatted_now = now.strftime("%d-%m-%Y" + "_%H-%M-%S")
        cpu_percentages = []
        memory_percentages = []
        time_stamps = []
    
        """
        Trainiert das Modell mit den angegebenen Trainings- und Test-/Validierungsdatensätzen.

        Gibt den Trainingsverlust und die Validierungsgenauigkeit während des Trainingsprozesses aus.
        Speichert das Modell mit der höchsten erreichten Validierungsgenauigkeit.
        Führt ein vorzeitiges Beenden durch, wenn die Validierungsgenauigkeit für eine bestimmte Anzahl von Epochen nicht verbessert wird.
        """


        start_time = time.time()
        for epoch in range(self.epochs):
            running_loss = 0.0
            for i, data in enumerate(tqdm(self.train_dataloader), 0):
                inputs, labels = data
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            correct = 0
            total = 0
            if i % 10 == 9:
                print(
                    "[%d, %5d] loss: %.3f" %
                    (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

            with torch.no_grad():
                for val_data in self.test_dataloader:
                    val_images, val_labels = val_data
                    val_outputs = self.model(val_images)
                    _, predicted = torch.max(val_outputs.data, 1)
                    total += val_labels.size(0)
                    correct += (predicted == val_labels).sum().item()
            accuracy = correct / total


            cpu_percentages.append(psutil.cpu_percent())
            memory_percentages.append(psutil.virtual_memory().percent)
            time_stamps.append(time.time()+start_time)


            if accuracy > 0.9:
                torch.save(
                    self.model.state_dict(),
                    f"model/PyTorch_Trained_Models/{model_name}_model_epoch_{epoch}_accuracy_{accuracy:.2f}_{formatted_now}.pth",
                )

            if accuracy > 0.95:
                torch.save(
                    self.model.state_dict(),
                    f"model/PyTorch_Trained_Models/{model_name}_model_epoch_{epoch}_accuracy_{accuracy:.2f}_{formatted_now}.pth",
                )
                break

            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.early_stopping_counter = 0
                print(f"Genauigkeit: {accuracy:.2f}")
            else:
                self.early_stopping_counter += 1

            if self.early_stopping_counter >= self.patience:
                print("Early stopping")
                break

        torch.save(
            self.model.state_dict(),
            f"model/PyTorch_Trained_Models/{model_name}_model_epoch_{epoch}_accuracy_{accuracy:.2f}_{formatted_now}.pth",
        )
        print(
            f"Training beende. Genauigkeit: {accuracy:.2f}" +
            f"Epoch: {epoch}")
        print(
            "Gespeicherter Pfad: ",
            f"model/PyTorch_Trained_Models/{model_name}_model_epoch_{epoch}_accuracy_{accuracy:.2f}_{formatted_now}.pth",
        )
        self.plot_cpu_memory_usage(cpu_percentages, memory_percentages, time_stamps)
        
    def plot_cpu_memory_usage(self, cpu_percentages, memory_percentages, time_stamps):
        """
        Zeichnet die CPU- und Speichernutzung über die Zeit.

        Parameter:
        cpu_percentages (Liste): Eine Liste von CPU-Nutzungsprozenten. Jedes Element in der Liste repräsentiert die CPU-Nutzung zu einem bestimmten Zeitpunkt.
        memory_percentages (Liste): Eine Liste von Speichernutzungsprozenten. Jedes Element in der Liste repräsentiert die Speichernutzung zu einem bestimmten Zeitpunkt.
        time_stamps (Liste): Eine Liste von Zeitstempeln, die den CPU- und Speichernutzungsprozenten entsprechen. Jedes Element in der Liste repräsentiert den Zeitpunkt, zu dem die entsprechenden CPU- und Speichernutzungsprozentsätze aufgezeichnet wurden.

        Gibt zurück:
        None
        """
   
        plt.xlabel("Zeit (s)")
        plt.ylabel("Nutzung (%)")
        plt.legend(["CPU", "Speicher"])
        plt.title("CPU- und Speichernutzung beim Trainieren des Modells")
        plt.plot(time_stamps, cpu_percentages, label="CPU Nutzung", color="red", linewidth=3)
        plt.plot(time_stamps, memory_percentages, label="Speichernutzung", color="blue", linewidth=3)
        plt.savefig("model/cpu_memory_usage.png")
        # plt.show()

class DataLoaderModelTrain:
    def __init__(self, batch_size, transform):
        self.batch_size = batch_size
        self.transform = transform

    @staticmethod
    def load_data(test_dir, train_dir, transform, batch_size):
        train_dataset = datasets.ImageFolder(
            root=train_dir, transform=transform)
        test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=True
        )
        return train_dataloader, test_dataloader


class Main(DataLoaderModelTrain):
    def __init__(self):
        self.batch_size = 64
        self.epochs = 50
        self.test_dir = "data/train-test-data/test"
        self.transform = transforms.Compose(
            [
                transforms.Resize((178, 218)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self.train_dir = "data/train-test-data/train"
        self.train_dataloader, self.test_dataloader = DataLoaderModelTrain.load_data(
            train_dir=self.train_dir,
            test_dir=self.test_dir,
            transform=self.transform,
            batch_size=self.batch_size,
        )

        self.model = SimpleCNN()
        self.trainer = Trainer(
            self.model,
            self.train_dataloader,
            self.test_dataloader,
            self.epochs,
            self.batch_size,
        )
        self.model_save_path = f"model/PyTorch_Trained_Models/"
        self.model_test_path = f"test/model_to_be_tested/model_to_be_tested.pth"

    @staticmethod
    def clean_up_pth(directory):
        """
        Löscht .pth-Dateien im angegebenen Verzeichnis.

        Args:
            directory (str): Der Verzeichnispfad.

        Returns:
            None

        Wird benötigt, um die Speicherauslastung zu reduzieren.
        Verhindert, dass die Ladefunktion des zu testenden Modells nicht mehr als ein Modell läd.
        """
        if not os.path.exists(directory):
            files = glob.glob(os.path.join(directory, "*.txt"))
            for file in files:
                if file == "default.txt":
                    return file
        else:
            files = glob.glob(os.path.join(directory, "*.pth"))
            for file in files:
                os.remove(file)
        return "Modell gelöscht!" 

    def train_and_save(self, model_name,model_test_path="test/model_to_be_tested/"):
        """
        Trainiere das Modell und speichere das trainierte Modell.

        Args:
            model_test_path (str, optional): Der Verzeichnispfad zum Speichern des zu testenden Modells.

        Returns:
            None
        """
  
        now = datetime.now()
        formatted_now = now.strftime("%d-%m-%Y")

        torch.save(
            self.model.state_dict(),
            f"{self.model_save_path}model_{self.batch_size}"
            + "-"
            + f"{self.epochs}"
            + ".pth",
        )

        if self.model_test_path is not None:
            self.clean_up_pth(model_test_path)
            torch.save(
                self.model.state_dict(), f"{model_test_path}{model_name}_{formatted_now}" + ".pth"
            )
        else:
            torch.save(
                self.model.state_dict(), f"{model_test_path}{model_name}_{formatted_now}" + ".pth"
            )
    





if __name__ == "__main__":
    model = SimpleCNN3()
    model_name = model.name
    batch_size = 32

    epochs = 10
    test_dir = "data/train-test-data/test"
    model_save_path = f"model/PyTorch_Trained_Models/"
    model_test_path = f"test/model_to_be_tested/model_to_be_tested.pth"
    train_dir = "data/train-test-data/train"
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)   
    criterion = nn.CrossEntropyLoss() 
    patience = 10
    best_accuracy = 0.96
    early_stopping_counter = 0
    
    transform = transforms.Compose(
        [
            transforms.Resize((178,218)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    train_dataloader, test_dataloader = DataLoaderModelTrain.load_data(
        train_dir=train_dir,
        test_dir=test_dir,
        transform=transform,
        batch_size=batch_size,
    )
    trainer = Trainer(
        model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        epochs=epochs,
        batch_size=batch_size,
        optimizer=optimizer,
        patience=patience,
        best_accuracy=best_accuracy,
        early_stopping_counter=early_stopping_counter,
        criterion=criterion
    )
    m = Main(
        batch_size=batch_size,
        epochs=epochs,
        test_dir=test_dir,
        transform=transform,
        train_dir=train_dir,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        model=model,
        model_save_path=model_save_path,
        model_test_path=model_test_path,
    )
    script_path = os.path.abspath(__file__)

    with open(script_path, 'r') as file:
        script_code = file.read()   
    with open(f"deploy/model_train.py", 'w') as file:
        file.write(script_code)

    with open(script_path, 'r') as file:
        script_code = file.read()   
    with open(f"test/model_test_scripts/model_train.py", 'w') as file:
        file.write(script_code)


    trainer.train()
    m.train_and_save(model_name=model_name)
