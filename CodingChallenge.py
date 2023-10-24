# imports
# TODO: add your imports here

import torch
import torch.nn as nn
import pathlib
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split




class RSiMCCDataset(Dataset):
    def __init__(self):
        super().__init__()
        # get images
        image_files = [x.resolve() for x in pathlib.Path(".").glob('data/*/*')]
        # Image.open() has a bug, this is a workaround
        self.images=[read_image(str(p)) for p in image_files] 
        # get labels from image path
        labels = [x.parts[-2] for x in image_files]
        self.classes = sorted(list(set(labels)))
        self.labels = [self.label_to_tensor(lbl) for lbl in labels]

        assert len(self.labels) == len(self.images), f"Found {len(self.labels)} labels and {len(self.images)} images"

    def label_to_tensor(self, lbl):
        """
        Converts the string label to a one-hot tensor where every entry is zero except the label which is one.
        
        """
        assert lbl in self.classes, f"Class {lbl} not a valid class (valid classes: {self.classes})"
        t = torch.zeros(len(self.classes))
        t[self.classes.index(lbl)] = 1
        return t

    def tensor_to_label(self, t):
        """
        Returns the classname in string format
        """
        assert len(t.shape) == 1, f"Can only convert 1-dimensional tensors (shape of tensor: {t.shape})"
        assert len(t) == len(self.classes), f"Lenght of tensor ({len(t)}) does not match number of classes ({len(classes)})"
        return self.classes[t.argmax()]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx].float()/255
        return img, self.labels[idx]

class DeepResCNN(nn.Module):
    def __init__(self, nb_classes):
        super(DeepResCNN, self).__init__()
        self.nb_classes = nb_classes
        
        # inputs (N,3,64,64)
        # Classical processing of images with pattern recognition
        # Follows ResNET architecture: Conv2d -> BN -> ReLU (-> MaxPool)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=8)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        # no pooling in 1st step

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=2)
        self.bn2   = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5)
        self.bn3   = nn.BatchNorm2d(32)
        self.relu3 = nn.ReLU()
        # no pooling in 3rd step

        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        self.bn4   = nn.BatchNorm2d(32)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=5)
        self.bn5   = nn.BatchNorm2d(16)
        self.relu5 = nn.ReLU()
        # no pooling in 5th step

        self.conv6 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, padding=2)
        self.bn6   = nn.BatchNorm2d(16)
        self.relu6 = nn.ReLU()
        self.pool6 = nn.MaxPool2d(kernel_size=2, stride=2)
        

        self.drop7 = nn.Dropout(p=0.05)
        self.lin7  = nn.Linear(16*4*4, nb_classes)
        
    def forward(self, x):        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = x + self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = x + self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.pool4(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)

        x = x + self.conv6(x)
        x = self.bn6(x)
        x = self.relu6(x)
        x = self.pool6(x)

        # outputs (None, 8, 4, 4)
        x = self.drop7(x)
        x = self.lin7(x.view(-1, 16*4*4))
        return x


def train_model(model, train_loader, val_loader, loss_fn, optimizer, device, num_epochs):
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    for epoch in range(num_epochs):
        # Training
        running_loss = 0.0
        correct = 0
        total = 0
        for batch, (X, y) in enumerate(train_loader):
            # Move tensors to the configured device
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()

            # Compute prediction and loss
            pred = model(X)

            # Compute loss and its gradients
            loss = loss_fn(pred, y)
            loss.backward()
            

            # Backpropagation step
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(pred.data, 1)
            total += y.size(0)
            correct += (predicted == y.argmax(1)).sum().item()

            # Display progress
            if batch % 5 == 0:
                loss = loss.item()
                print(f"Epoch {epoch+1}, batch {batch+1}/{len(train_loader)}, loss: {loss:.4f}")
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        print(f"Epoch {epoch+1}, train loss: {train_loss:.4f}, train accuracy: {train_acc:.2f}%")

        # Validation
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch, (X, y) in enumerate(val_loader):
                X, y = X.to(device), y.to(device)
                pred = model(X)
                loss = loss_fn(pred, y)
                running_loss += loss.item()
                _, predicted = torch.max(pred.data, 1)
                total += y.size(0)
                correct += (predicted == y.argmax(1)).sum().item()
            val_loss = running_loss / len(val_loader)
            val_acc = 100 * correct / total
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            print(f"Epoch {epoch+1}, val loss: {val_loss:.4f}, val accuracy: {val_acc:.2f}%")
    return model # discarded to keep the structure of the provided notebook: train_losses, train_accs, val_losses, val_accs



def get_my_model(nb_classes=10):
    model = DeepResCNN(nb_classes)
    return model

def get_my_ds():
    dataset = RSiMCCDataset()
    return dataset


def train(model, dataset):
    # TODO: add everything you need to train a model based on YOUR definition of the ds
    assert model.nb_classes == len(dataset.classes)

    # Define training parameters
    num_epochs = 85 # 100
    batch_size = 200 # 128
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create data loaders for training and validation sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Inspect a batch of data
    train_features, train_labels = next(iter(train_loader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape:  {train_labels.size()}")


    criterion = torch.nn.CrossEntropyLoss() # CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Call my training function
    print(f'\nTraining the model with {num_epochs} epochs for {len(train_loader)} batches  of size {batch_size} each, representing an amount of 80% of the dataset i.e. {train_size} elements.\nTraining algorithm is SGD with momentum 0.9 and learning rate 1e-3 optimizing the crossentropy loss.\nUsing device {device}.\n')
    print(f'The loss used is CrossEntropyLoss() and the model is a DeepResCNN with {model.nb_classes} classes predicted.\nThe number of parameters is {sum(p.numel() for p in model.parameters() if p.requires_grad)}.\n')

    
    train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs)

    print("Training finished.")
    return model


import gc
def free_memory():
    gc.collect()
    torch.cuda.empty_cache()




def summary():
    # TODO: print all information that you think may be relevant
    print("Relevant information has been displayed during training.\n")



def final_evaluate(model, dataset):
    device = next(model.parameters()).device
    free_memory() # free memory before the heavy lifting aka evaluating the whols dataset per batch
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=300, shuffle=False)
    total = len(dataset)
    correct = 0

    with torch.no_grad():
        for _, (X, y) in enumerate(val_loader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            predicted = torch.argmax(pred, dim=1)
            true_labels = torch.argmax(y, dim=1)
            correct += (predicted == true_labels).sum().item()
        val_acc = correct / total
    return val_acc
