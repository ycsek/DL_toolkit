'''
Author: Jason Shi
Date: 02-11-2024 13:01:37
Last Editors: Jason
Contact Last Editors: D23090120503@cityu.edu.mo
LastEditTime: 24-01-2025 13:52:28
'''

#! main.py which is responsible for parsing the arguments and calling the training and evaluation functions.


from utils.utils_datasets import get_dataset
from utils.utils_networks import get_network
from utils.utils_train import train
from utils.utils_evaluate import evaluate
import wandb
from torchvision.utils import save_image
import torch.optim as optim
import torch.nn as nn
import torch
import time
import argparse
import os
import sys
sys.path.append('../')


# get time, return the current time, this part is used to record the start and end time of the training process


def get_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def save_images(images, epoch, batch_idx):
    save_image(images, f'images/epoch{epoch}_batch{batch_idx}.png')


def main():
    # print the status of the cudnn cuda and the start time of the training
    print("CUDNN STATUS: {}".format(torch.backends.cudnn.enabled))
    print("CUDA STATUS: {}".format(torch.cuda.is_available()))
    start_time = time.time()
    print("STARTING TRAINING:{}".format(get_time()))

    # parse the arguments, including dataset, network, epochs, batch_size, learning_rate, num_workers, and device
    parser = argparse.ArgumentParser(
        description='Deep Learning Training Script')
    parser.add_argument('--dataset', type=str,
                        default='MNIST', help='datasets')
    parser.add_argument('--network', type=str, default='MLP', help='networks')
    parser.add_argument('--epochs', type=int, default=10, help='epochs')
    parser.add_argument('--batch_size', type=int,
                        default=64, help='batch_size')
    parser.add_argument('--learning_rate', type=float,
                        default=0.001, help='learning rate')
    parser.add_argument('--num_workers', type=int,
                        default=4, help='num_workers')
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--data_path', type=str,
                        default='./data', help='path to the dataset')
    args = parser.parse_args()

    # init wandb
    wandb.init(project="deep-learning-project", config={
        "dataset": args.dataset,
        "network": args.network,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "num_workers": args.num_workers
    })

    config = wandb.config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # get the dataset
    train_loader, test_loader, channel, im_size, num_classes = get_dataset(
        args.dataset, args.data_path)

    # get the network model
    model = get_network(config.network, channel=channel,
                        input_size=im_size, num_classes=num_classes).to(device)

    # define the loss function and optimizer
    criterion = nn.CrossEntropyLoss().to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # training and testing
    best_acc = 0.0
    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc = train(
            model, device, train_loader, criterion, optimizer, epoch, config.epochs)
        test_loss, test_acc = evaluate(
            model, device, test_loader, criterion, epoch, config.epochs, phase='Test')

        # save the best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'best_model.pth')

    # log device info
    wandb.log({'GPU': torch.cuda.get_device_name(0)})

    # print the training and testing results
    print(f"Epoch: {epoch}/{config.epochs}, Train Loss: {train_loss:.4f}, Train Acc: {
          train_acc:.2f}%, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    print(f"Done, The best test acc: {best_acc:.2f}%")
    end_time = time.time()
    print("END:{}".format(get_time()))
    print("TRAINING TIME: {:.2f} seconds".format(end_time - start_time))
    wandb.finish()


if __name__ == '__main__':
    main()
