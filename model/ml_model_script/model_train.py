import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import torchvision.transforms as transforms
import os
import glob
from datetime import datetime

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(33456 , 120)
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

    def __init__(self, model, train_dataloader, test_dataloader, epochs, batch_size):
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
        """
        Trainiert das Modell mit den angegebenen Trainings- und Test-/Validierungsdatensätzen.

        Gibt den Trainingsverlust und die Validierungsgenauigkeit während des Trainingsprozesses aus.
        Speichert das Modell mit der höchsten erreichten Validierungsgenauigkeit.
        Führt ein vorzeitiges Beenden durch, wenn die Validierungsgenauigkeit für eine bestimmte Anzahl von Epochen nicht verbessert wird.
        """
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
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

            with torch.no_grad():
                for val_data in self.test_dataloader:
                    val_images, val_labels = val_data
                    val_outputs = self.model(val_images)
                    _, predicted = torch.max(val_outputs.data, 1)
                    total += val_labels.size(0)
                    correct += (predicted == val_labels).sum().item()
            accuracy = correct / total

            if accuracy > 0.9:
                torch.save(self.model.state_dict(), f'model/PyTorch_Trained_Models/model_epoch_{epoch+1}_accuracy_{accuracy:.2f}.pth')

            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.early_stopping_counter = 0
                print(f"Genauigkeit: {accuracy:.2f}")
            else:
                self.early_stopping_counter += 1

            if self.early_stopping_counter >= self.patience:
                print('Early stopping')
                break

        print('Finished Training')

class DataLoaderModelTrain:
    def __init__(self, batch_size, transform):
        self.batch_size = batch_size
        self.transform = transform
    
    @staticmethod
    def load_data(test_dir, train_dir, transform, batch_size):
        train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
        test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        return train_dataloader, test_dataloader


class Main():
    def __init__(self):
        self.batch_size = 64
        self.epochs = 1
        self.test_dir = 'data/train-test-data/test'
        self.transform = transforms.Compose([
            transforms.Resize((178, 218)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.train_dir = 'data/train-test-data/train'
        self.train_dataloader, self.test_dataloader = DataLoaderModelTrain.load_data(train_dir=self.train_dir,test_dir=self.test_dir,transform=self.transform,batch_size=self.batch_size)

        self.model = SimpleCNN()
        self.trainer = Trainer(self.model, self.train_dataloader, self.test_dataloader, self.epochs, self.batch_size)
        self.model_save_path = f'model/PyTorch_Trained_Models/'
        self.model_test_path = f'test/model_to_be_tested/model_to_be_tested.pth'


    @staticmethod
    def clean_up_pth(directory):
        files = glob.glob(os.path.join(directory, '*.pth'))
        for file in files:
            os.remove(file)

    def train_and_save(self, model_test_path="test/model_to_be_tested/"):
        self.trainer.train()
        now = datetime.now()
        formatted_now = now.strftime("%d-%m-%Y")
        
        torch.save(self.model.state_dict(), f'{self.model_save_path}model_{self.batch_size}' + '-' + f'{self.epochs}' + '.pth')
        if self.model_test_path is not None:
            self.clean_up_pth(model_test_path)            
            torch.save(self.model.state_dict(), f"{model_test_path}{formatted_now}" + ".pth")
        else:
            torch.save(self.model.state_dict(), f"{model_test_path}{formatted_now}" + ".pth")



if __name__ == "__main__":
    m = Main()
    m.train_and_save()