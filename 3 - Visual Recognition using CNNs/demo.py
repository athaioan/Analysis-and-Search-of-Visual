import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from torch.utils.data import DataLoader
import torch
from types import SimpleNamespace
from model import Network
import time

### Setting arguments
args = SimpleNamespace(epochs=300,
                       batch_size=64,
                       # lr=10**(-3),
                       lr=0.1,

                       data_train_folder="C:/Users/johny/Desktop/Visual Analysis/Projects/3 - Visual Recognition using CNNs/data/train_set/",
                       data_test_folder="C:/Users/johny/Desktop/Visual Analysis/Projects/3 - Visual Recognition using CNNs/data/test_set/",
                       session_name="Default5C/",
                       # n_filters=[24, 48, 96],
                       n_filters=[64, 128, 256],

                       shuffle=True,

                       )

## loading the datasets
data_train = os.listdir(args.data_train_folder)

data_test = os.listdir(args.data_test_folder)

train_data, train_labels = construct_data(data_train, args.data_train_folder)

train_loader = Cifar_Dataset(train_data, train_labels)
train_loader = DataLoader(train_loader, batch_size=args.batch_size, shuffle=args.shuffle)


test_data, test_labels = construct_data(data_test, args.data_test_folder)
test_loader = Cifar_Dataset(test_data, test_labels)
test_loader = DataLoader(test_loader, batch_size=args.batch_size, shuffle=args.shuffle)

## loading model
## Initializing the model
model = Network(10, args.n_filters).cuda()
model.epochs = args.epochs
model.session_name = args.session_name

if not os.path.exists(model.session_name):
    os.makedirs(model.session_name)


criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)


t = time.process_time()

## Training
for current_epoch in range(model.epochs):

    model.epoch = current_epoch

    print("Training epoch...")
    model.train_epoch(train_loader, optimizer, criterion)

    torch.save(model.state_dict(), model.session_name + "stage_1.pth")

    model.visualize_graph()


elapsed_time = time.process_time() - t

model.load_pretrained(model.session_name + "stage_1.pth")

## Evaluate model
train_conf_matrix = model.val_epoch(train_loader, criterion, verbose=True)
test_conf_matrix = model.val_epoch(test_loader, criterion, verbose=True)


print("Train Accuracy:", np.sum(np.diag(train_conf_matrix))/np.sum(train_conf_matrix))
print("Test Accuracy:", np.sum(np.diag(test_conf_matrix))/np.sum(test_conf_matrix))
print("Training Time:", elapsed_time)