'''
Author: Jason Shi
Date: 02-11-2024 13:14:41
Last Editors: Jason
Contact Last Editors: D23090120503@cityu.edu.mo
LastEditTime: 04-11-2024 01:09:26
'''
#! This module is responsible for the evaluation process of the model and records the relevant information using wandb.
import torch
import wandb
from tqdm import tqdm


def evaluate(model, device, test_loader, criterion, epoch, total_epochs, phase='Test'):
    '''
    @param:
    model: Neural network models
    device: CPU or GPU(cuda)
    test_loader: DataLoader for testing dataset
    criterion: Loss function(use CrossEntropyLoss)
    epoch(int): Current epoch
    total_epochs(int): Total epochs
    phase(str): Test or Validation

    @return:
    avg_loss(float): Average loss
    accuracy(float): Accuracy

    '''
    # Initialize model
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    
    with torch.no_grad():
        progress_bar = tqdm(enumerate(test_loader), total=len(
            test_loader), desc=f"{phase} [{epoch}/{total_epochs}]")
        for batch_idx, (inputs, targets) in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar.set_postfix(loss=loss.item(), acc=100.*correct/total)

    avg_loss = running_loss / total
    accuracy = 100. * correct / total

    wandb.log({
        f'{phase} Loss': avg_loss,
        f'{phase} Accuracy': accuracy,
        'Epoch': epoch
    })

    print(f'{phase} Loss: {avg_loss:.4f}, {phase} Accuracy: {accuracy:.2f}%')

    return avg_loss, accuracy
