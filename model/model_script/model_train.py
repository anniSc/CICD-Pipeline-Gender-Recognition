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
import math
# def math(Hin, Win, padding, dilation, kernel_size, stride):
#     Hout = math.floor((Hin + 2*padding - dilation*(kernel_size-1) - 1) / stride) + 1
#     Wout = math.floor((Win + 2*padding - dilation*(kernel_size-1) - 1) / stride) + 1
#     print(Hout, Wout) 
#     return Hout, Wout
class SimpleCNN(nn.Module):
    """
    Einfaches CNN-Modell zur Klassifizierung von Bildern.

    Args:
        None

    Attributes:
        conv1 (nn.Conv2d): Erste Faltungsoperationsschicht.
        pool (nn.MaxPool2d): Max-Pooling-Schicht.
        conv2 (nn.Conv2d): Zweite Faltungsoperationsschicht.
        fc1 (nn.Linear): Erste vollständig verbundene Schicht.
        fc2 (nn.Linear): Zweite vollständig verbundene Schicht.
        fc3 (nn.Linear): Dritte vollständig verbundene Schicht.

    Methods:
        forward(x): Führt die Vorwärtsberechnung des Modells durch.

    Returns:
        x (torch.Tensor): Ausgabe des Modells.
    """
    def __init__(self):
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

    def __init__(self, model, train_dataloader, test_dataloader, epochs,optimizer,criterion, batch_size, patience=10,best_accuracy=0.0,early_stopping_counter=0):
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.epochs = epochs
        self.batch_size = batch_size
        self.criterion = criterion
        self.optimizer = optimizer
        self.patience = patience
        self.best_accuracy = best_accuracy
        self.early_stopping_counter = early_stopping_counter

    def train(self):
      
        """
    Trainiert das Modell mit den angegebenen Trainings- und Test-/Validierungsdatensätzen.

    Diese Methode führt die folgenden Schritte aus:
    1. Initialisiert Listen für die CPU- und Speichernutzung sowie Zeitstempel.
    2. Führt das Training für eine festgelegte Anzahl von Epochen durch.
    3. In jeder Epoche werden die Daten durch das Modell geführt, der Verlust berechnet und die Gewichte des Modells aktualisiert.
    4. Nach jeder Epoche wird die Genauigkeit des Modells auf den Test-/Validierungsdaten berechnet.
    5. Die CPU- und Speichernutzung sowie der Zeitstempel werden nach jeder Epoche aufgezeichnet.
    6. Wenn die Genauigkeit 90% oder 95% übersteigt, wird der aktuelle Zustand des Modells gespeichert.
    7. Wenn die Genauigkeit nicht innerhalb einer festgelegten Anzahl von Epochen verbessert wird, wird das Training frühzeitig beendet ("early stopping").
    8. Am Ende des Trainings wird der Zustand des Modells gespeichert und die CPU- und Speichernutzung über die Zeit geplottet.

    Parameter:
    Keine
    Gibt zurück:
    None
    """
        


        now = datetime.now()
        formatted_now = now.strftime("%d-%m-%Y" + "_%H-%M-%S")
        cpu_percentages = []
        memory_percentages = []
        time_stamps = []
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
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 100))
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
            time_stamps.append(time.time() + start_time)

            if accuracy > 0.9:
                torch.save(
                    self.model.state_dict(),
                    f"model/PyTorch_Trained_Models/model_epoch_{epoch}_accuracy_{accuracy:.2f}_{formatted_now}.pth",
                )

            if accuracy > 0.95:
                torch.save(
                    self.model.state_dict(),
                    f"model/PyTorch_Trained_Models/model_epoch_{epoch}_accuracy_{accuracy:.2f}_{formatted_now}.pth",
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
            f"model/PyTorch_Trained_Models/model_epoch_{epoch}_accuracy_{accuracy:.2f}_{formatted_now}.pth",
        )
        print(f"Training beende. Genauigkeit: {accuracy:.2f}" + f"Epoch: {epoch}")
        print(
            "Gespeicherter Pfad: ",
            f"model/PyTorch_Trained_Models/model_epoch_{epoch}_accuracy_{accuracy:.2f}_{formatted_now}.pth",
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
        plt.plot(time_stamps, cpu_percentages, label="CPU Nutzung")
        plt.plot(time_stamps, memory_percentages, label="Speichernutzung")
        plt.xlabel("Zeit (s)")
        plt.ylabel("Nutzung (%)")
        plt.title("CPU- und Speichernutzung über die Zeit")
        plt.legend()
        plt.savefig("model/cpu_memory_usage.png")
        plt.show()

class DataLoaderModelTrain:
    def __init__(self, batch_size, transform):
        """
        Initialisiert die DataLoaderModelTrain-Klasse.

        Args:
            batch_size (int): Die Batch-Größe für den DataLoader.
            transform (torchvision.transforms): Die Transformationen, die auf die Daten angewendet werden sollen.
        """
        self.batch_size = batch_size
        self.transform = transform

    @staticmethod
    def load_data(test_dir, train_dir, transform, batch_size):
        """
        Lädt die Trainings- und Testdaten und erstellt DataLoader-Objekte.

        Args:
            test_dir (str): Der Pfad zum Verzeichnis mit den Testdaten.
            train_dir (str): Der Pfad zum Verzeichnis mit den Trainingsdaten.
            transform (torchvision.transforms): Die Transformationen, die auf die Daten angewendet werden sollen.
            batch_size (int): Die Batch-Größe für den DataLoader.

        Returns:
            tuple: Ein Tupel bestehend aus den Trainings- und Test-DataLoader-Objekten.
        """
        train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
        test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=True
        )
        return train_dataloader, test_dataloader


class Main(DataLoaderModelTrain):
    """
    Hauptklasse zum Trainieren und Speichern eines PyTorch-Modells.

    Attribute:
        batch_size (int): Die Batch-Größe für das Training des Modells.
        epochs (int): Die Anzahl der Epochen für das Training des Modells.
        test_dir (str): Der Verzeichnispfad für die Testdaten.
        transform (torchvision.transforms.Compose): Die Daten-Transformationspipeline.
        train_dir (str): Der Verzeichnispfad für die Trainingsdaten.
        train_dataloader (torch.utils.data.DataLoader): Der Datenlader für die Trainingsdaten.
        test_dataloader (torch.utils.data.DataLoader): Der Datenlader für die Testdaten.
        model (SimpleCNN): Das PyTorch-Modell.
        trainer (Trainer): Das Trainer-Objekt zum Trainieren des Modells.
        model_save_path (str): Der Verzeichnispfad zum Speichern des trainierten Modells.
        model_test_path (str): Der Verzeichnispfad zum Speichern des zu testenden Modells.
    """

    def __init__(
        self,
        batch_size=64,
        epochs=50,
        test_dir="data/train-test-data/test",
        transform=None,
        train_dir="data/train-test-data/train",
        train_dataloader=None,
        test_dataloader=None,
        model=SimpleCNN(),
        model_save_path="model/PyTorch_Trained_Models/",
        model_test_path="test/model_to_be_tested/",
    ):
        """
        Initialisiert die ModelTrain-Klasse.

        Parameter:
        - batch_size (int): Die Batch-Größe für das Training und Testen.
        - epochs (int): Die Anzahl der Epochen für das Training.
        - test_dir (str): Der Verzeichnispfad für die Testdaten.
        - transform (torchvision.transforms.Compose): Die Daten-Transformationspipeline.
        - train_dir (str): Der Verzeichnispfad für die Trainingsdaten.
        - train_dataloader (torch.utils.data.DataLoader): Der Datenlader für die Trainingsdaten.
        - test_dataloader (torch.utils.data.DataLoader): Der Datenlader für die Testdaten.
        - model (SimpleCNN): Das Modell für die Geschlechtererkennung.
        - trainer (Trainer): Der Trainer zum Trainieren des Modells.
        - model_save_path (str): Der Verzeichnispfad zum Speichern des trainierten Modells.
        - model_test_path (str): Der Verzeichnispfad für das zu testende Modell.
        """
        self.batch_size = batch_size
        self.epochs = epochs
        self.test_dir = test_dir
        self.model_save_path = model_save_path
        self.model_test_path = model_test_path
        self.train_dir = train_dir
        self.transform = transform
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader


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
        files = glob.glob(os.path.join(directory, "*.pth"))
        for file in files:
            os.remove(file)

    def train_and_save(self, model_test_path="test/model_to_be_tested/"):
        """
        Trainiere das Modell und speichere das trainierte Modell.

        Args:
            model_test_path (str, optional): Der Verzeichnispfad zum Speichern des zu testenden Modells.

        Returns:
            None
        """
  
        now = datetime.now()
        formatted_now = now.strftime("%d-%m-%Y")

        if self.model_test_path is not None:
            self.clean_up_pth(model_test_path)
            torch.save(
                self.model.state_dict(), f"{model_test_path}{formatted_now}" + ".pth"
            )
        else:
            torch.save(
                self.model.state_dict(), f"{model_test_path}{formatted_now}" + ".pth"
            )


class ImageClassificationBase(nn.Module):
    
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        acc = ImageClassificationBase.accuracy(out, labels)  # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))

    def accuracy(outputs, labels):
        _, preds = torch.max(outputs, dim=1)
        return torch.tensor(torch.sum(preds == labels).item() / len(preds))




class class_finder(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 64 x 16 x 16
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 128 x 8 x 8
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 256 x 4 x 4
            nn.Flatten(),
            nn.Linear(256*4*4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        return self.network(x)




if __name__ == "__main__":
    model = SimpleCNN()
    # model = class_finder()
    # model = class_finder()
    batch_size = 64
    import torch.optim as optim

    epochs = 50
    test_dir = "data/train-test-data/test"
    model_save_path = f"model/PyTorch_Trained_Models/"
    model_test_path = f"test/model_to_be_tested/model_to_be_tested.pth"
    train_dir = "data/train-test-data/train"
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)   
    criterion = nn.CrossEntropyLoss() 
    patience = 10
    best_accuracy = 0.96
    early_stopping_counter = 5
    # transform = transforms.Compose(
    #     [
    #         transforms.Resize((180,220)),
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #     ]
    # )
    transform = transforms.Compose([ transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) ])
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
    trainer.train()
    m.train_and_save()
