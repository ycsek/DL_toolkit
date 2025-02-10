'''
Author: Jason Shi
Date: 02-11-2024 13:48:48
Last Editors: Jason
Contact Last Editors: D23090120503@cityu.edu.mo
LastEditTime: 04-11-2024 01:09:03
'''

#! This module is responsible for the training process of the model and records the loss and accuracy using wandb.
import wandb
from tqdm import tqdm


def train(model, device, train_loader, criterion, optimizer, epoch, total_epochs):
    '''
    Train the model

    @param:
    model: Neural network models
    device: CPU or GPU(cuda)
    train_loader: DataLoader for training dataset
    criterion: Loss function(use CrossEntropyLoss)
    optimizer: Optimizer
    epoch: Current epoch
    total_epochs: Total epochs

    @return:
    avg_loss: Average loss
    accuracy: Accuracy

    '''
    # Initialize the model to train mode
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # Use tqdm to show the progress bar
    progress_bar = tqdm(enumerate(train_loader), total=len(
        train_loader), desc=f"Epoch [{epoch}/{total_epochs}]")

    # Iterate over the training dataset
    for batch_idx, (inputs, targets) in progress_bar:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar.set_postfix(loss=loss.item(), acc=100.*correct/total)

    avg_loss = running_loss / total
    accuracy = 100. * correct / total

    # use wandb to log
    wandb.log({
        'Train Loss': avg_loss,
        'Train Accuracy': accuracy,
        'Epoch': epoch
    })

    # return the average loss and accuracy
    return avg_loss, accuracy
