# A simple deep learning model training framework

## Introduction :

This project is used to train different deep learning models (MLP, ConvNet, LeNet, AlexNet, VGG) on various datasets (MNIST, CIFAR-10, CIFAR-100, SVHN), and use Weights & Biases (wandb) to record the training and evaluation results.



### Configuring wandb:

1. Sign up for a Weights & Biases account (https://wandb.ai/site/)

2. Run in terminal

   ```
   wandb login
   ```



### Installation 

```
conda env create -f environment.yaml;
conda activate wtorch
```



### utils Description:

Utils includes `datasets.py, networks.py, train.py, and evaluate.py`. They will be called as a toolkit in `main.py`.



### Run:

Use the following code and run it in the terminal. The parameters can be modified by yourself.

```
python main.py --dataset MNIST --network MLP --epochs 500 --batch_size 128 --learning_rate 0.001
```

The results can be viewed in wandb.



**If you have any questions, please contact me or Prof. Chen**

***Your feedback will be highly appreciated !*** 



<p align='right'> Best,<br>
    Jason<br>
    6/11/2024
</p>

