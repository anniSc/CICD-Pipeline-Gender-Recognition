# %%
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms, models
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import DataLoader
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import os
import pandas as pd
from torchvision.io import read_image
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
from torch import cuda
import matplotlib.pyplot as plt
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm



epochs = 5
batch_size = 64

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.ImageFolder(root='C:/Users/busse/Bachelorarbeit/CICD-Pipeline-Gender-Recognition/data/train',transform=transform)
test_dataset = datasets.ImageFolder(root= 'C:/Users/busse/Bachelorarbeit/CICD-Pipeline-Gender-Recognition/data/val',transform=transform)
df_men_train = pd.read_csv("C:/Users/busse/Bachelorarbeit/CICD-Pipeline-Gender-Recognition/model/excel_sheets/image_paths_men.csv")
df_women_train = pd.read_csv("C:/Users/busse/Bachelorarbeit/CICD-Pipeline-Gender-Recognition/model/excel_sheets/image_paths_women.csv")
df_women_test = pd.read_csv("C:/Users/busse/Bachelorarbeit/CICD-Pipeline-Gender-Recognition/model/excel_sheets/image_paths_val_women.csv")
df_men_test = pd.read_csv("C:/Users/busse/Bachelorarbeit/CICD-Pipeline-Gender-Recognition/model/excel_sheets/image_paths_val_men.csv")
merged_df_train = pd.concat([df_men_train, df_women_train], ignore_index=True)
merged_df_test = pd.concat([df_men_test, df_women_test], ignore_index=True)
merged_df_test.to_csv("merged_df_test.csv")
merged_df_train.to_csv("merged_df_train.csv")


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 1])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 2]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


train_data = CustomImageDataset(annotations_file='C:/Users/busse/Bachelorarbeit/CICD-Pipeline-Gender-Recognition/model/merged_df_train.csv',img_dir='C:/Users/busse/Bachelorarbeit/CICD-Pipeline-Gender-Recognition/data/train')
test_data =  CustomImageDataset(annotations_file='C:/Users/busse/Bachelorarbeit/CICD-Pipeline-Gender-Recognition/model/merged_df_test.csv',img_dir='C:/Users/busse/Bachelorarbeit/CICD-Pipeline-Gender-Recognition/data/val')

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
 
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
img = train_features[0].permute(1, 2, 0)
plt.imshow(img, cmap="gray")

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print("Using {} device".format(device))



class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)  # 3 input channels, 6 output channels, 5x5 kernel
        self.pool = nn.MaxPool2d(4, 4)  # 4x4 max pooling
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(2,2)  # 6 input channels, 16 output channels, 5x5 kernel
        self.fc1 = nn.Linear(33456 , 120)  # 53*53 from the dimension of the image after convolutions
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)  # 2 output classes
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten the output of the conv layers
        x = F.relu(self.fc1(x))  # Apply ReLU to the output of the first fully connected layer
        x = F.relu(self.fc2(x))  # Apply ReLU to the output of the second fully connected layer
        x = self.fc3(x)  # No activation function is applied to the output of the last layer
        return x


model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Assume `val_dataloader` is your DataLoader for the validation set
for epoch in range(epochs):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(tqdm(train_dataloader), 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()  
         

        # Trainingsverlauf ausgeben
        running_loss += loss.item()

        #Alle 10 Batches wird der Loss ausgegeben
        if i % 10 == 9:    
            # Berechnung der Accuracy
            correct = 0
            total = 0
            with torch.no_grad():
                for val_data in test_dataloader:
                    val_images, val_labels = val_data
                    val_outputs = model(val_images)
                    _, predicted = torch.max(val_outputs.data, 1)
                    total += val_labels.size(0)
                    correct += (predicted == val_labels).sum().item()
            val_accuracy = correct / total

            print(f'Epoch: {epoch + 1}, Batch: {i + 1}, Loss: {running_loss / 10}, Validation Accuracy: {val_accuracy * 100}%')
            running_loss = 0.0

print('Finished Training')

# %%
torch.save(model.state_dict(), f'model{batch_size}.pth')

# %%
import torch
from torch.utils.data import DataLoader

# Load the model
model = SimpleCNN()
model.load_state_dict(torch.load(f'model{batch_size}.pth'))

# Assume `test_dataset` is your ImageFolder dataset
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Use the model to make predictions
for inputs, _ in test_loader:
    inputs = inputs.to(device)
    output = model(inputs)
    probabilities = torch.nn.functional.softmax(output, dim=1)
    predictions = torch.argmax(probabilities, dim=1)
    predictions_list = predictions.cpu().numpy().tolist()
    print(predictions_list)

# %%
import torch
from torch.utils.data import DataLoader


# Load the model
model = SimpleCNN()
model.load_state_dict(torch.load(f'model{batch_size}.pth'))

# Assume `test_dataset` is your ImageFolder dataset
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Use the model to make predictions and calculate accuracy
correct = 0
total = 0

for inputs, labels in test_loader:
    inputs = inputs.to(device)
    labels = labels.to(device)
    
    # Forward pass
    outputs = model(inputs)
    _, predicted = torch.max(outputs.data, 1)
    
    # Update total and correct counts
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

# Calculate accuracy
accuracy = correct / total
print('Accuracy of the model on the test images: {}%'.format(100 * accuracy))


