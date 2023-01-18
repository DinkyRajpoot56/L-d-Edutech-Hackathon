import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import time
from tqdm.auto import tqdm
from model import build_model
from dataset import get_datasets, get_data_loaders
from utils import save_model, save_plots, LRScheduler
# Construct the argument parser.
parser = argparse.ArgumentParser()
parser.add_argument(
    '-e', '--epochs', type=int, default=20,
    help='number of epochs to train our network for'
)
parser.add_argument(
    '-b', '--batch-size', dest='batch_size', type=int, default=32,
    help='batch size for data loaders'
)
parser.add_argument(
    '-lr', '--learning-rate', type=float,
    dest='learning_rate', default=0.001,
    help='learning rate for training the model'
)
args = vars(parser.parse_args())
We load all the custom modules that we need
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import time
from tqdm.auto import tqdm
from model import build_model
from dataset import get_datasets, get_data_loaders
from utils import save_model, save_plots, LRScheduler
# Construct the argument parser.
parser = argparse.ArgumentParser()
parser.add_argument(
    '-e', '--epochs', type=int, default=20,
    help='number of epochs to train our network for'
)
parser.add_argument(
    '-b', '--batch-size', dest='batch_size', type=int, default=32,
    help='batch size for data loaders'
)
parser.add_argument(
    '-lr', '--learning-rate', type=float,
    dest='learning_rate', default=0.001,
    help='learning rate for training the model'
)
args = vars(parser.parse_args())
We load all the custom modules that we need. For the argument parser we have the following flags:

--epochs: To control the number of epochs to train for.
--batch-size: To provide the batch size directly while executing the script.
--learning-rate: We can also control the initial learning rate using this flag.
The Training and Validation Functions
We will use pretty standard PyTorch image classification functions for training and validation. The following block contains both functions.

train.py
# Training function.
def train(model, trainloader, optimizer, criterion):
    model.train()
    print('Training')
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        counter += 1
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # Forward pass.
        outputs = model(image)
        # Calculate the loss.
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()
        # Calculate the accuracy.
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == labels).sum().item()
        # Backpropagation
        loss.backward()
        # Update the weights.
        optimizer.step()
    
    # Loss and accuracy for the complete epoch.
    epoch_loss = train_running_loss / counter
    # epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
    epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
    return epoch_loss, epoch_acc
# Validation function.
def validate(model, testloader, criterion):
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1
            
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            # Forward pass.
            outputs = model(image)
            # Calculate the loss.
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            # Calculate the accuracy.
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()
        
    # Loss and accuracy for the complete epoch.
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / len(testloader.dataset))
    return epoch_loss, epoch_acc
For both the functions, we return the loss and accuracy value for each epoch at the end.

The Main Code Block
Finally, the rest of the code is inside the main block.

train.py
if __name__ == '__main__':
    # Load the training and validation datasets.
    dataset_train, dataset_valid = get_datasets()
    print(f"[INFO]: Number of training images: {len(dataset_train)}")
    print(f"[INFO]: Number of validation images: {len(dataset_valid)}")
    # Load the training and validation data loaders.
    train_loader, valid_loader = get_data_loaders(
        dataset_train=dataset_train, 
        dataset_valid=dataset_valid,
        batch_size=args['batch_size'],
        num_workers=4
    )
    # Learning_parameters. 
    lr = args['learning_rate']
    epochs = args['epochs']
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Computation device: {device}")
    print(f"Learning rate: {lr}")
    print(f"Epochs to train for: {epochs}\n")
    # Load the model.
    model = build_model(
        pretrained=True,
        fine_tune=True, 
        num_classes=2
    ).to(device)
    
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")
    # Optimizer.
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Loss function.
    criterion = nn.CrossEntropyLoss()
    # Initialize LR scheduler.
    lr_scheduler = LRScheduler(optimizer, patience=1, factor=0.5)
    # Lists to keep track of losses and accuracies.
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []
    # Start the training.
    for epoch in range(epochs):
        print(f"[INFO]: Epoch {epoch+1} of {epochs}")
        train_epoch_loss, train_epoch_acc = train(
            model, 
            train_loader, 
            optimizer, 
            criterion
        )
        valid_epoch_loss, valid_epoch_acc = validate(
            model, 
            valid_loader, 
            criterion
        )
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)
        print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
        print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
        lr_scheduler(valid_epoch_loss)
        print('-'*50)
        time.sleep(5)
        
    # Save the trained model weights.
    save_model(epochs, model, optimizer, criterion)
    # Save the loss and accuracy plots.
    save_plots(train_acc, valid_acc, train_loss, valid_loss)
    print('TRAINING COMPLETE')
Let’s go over some of the important parts: