from torch import nn
import torch
import matplotlib.pyplot as plt
import numpy as np

class Network(nn.Module):
    def __init__(self, n_classes, n_filters):
        super(Network, self).__init__()

        self.train_history = {"loss": [],
                              "accuracy": []}
        self.val_history = {"loss": [],
                            "accuracy": []}

        self.min_val = np.inf
        self.n_classes = n_classes

        ## relu
        self.relu = nn.ReLU()
        # self.relu = nn.LeakyReLU()
        self.dropout_30 = nn.Dropout(p=0.3)

        self.pool = nn.MaxPool2d(2, stride=2)

        ## layer 1
        self.conv1 = nn.Conv2d(3, n_filters[0], 5, stride=1, padding=0)
        self.batch_norm_1 = nn.BatchNorm2d(64)
        ## layer 2
        self.conv2 = nn.Conv2d(n_filters[0], n_filters[1], 3, stride=1, padding=0)
        self.batch_norm_2 = nn.BatchNorm2d(128)
        ## layer 3
        self.conv3 = nn.Conv2d(n_filters[1], n_filters[2], 3, stride=1, padding=0)

        ## FC1
        self.fc1 = nn.Linear(2*2*n_filters[2], 512, bias=False)
        # self.fc1 = nn.Linear(n_filters[2], 512, bias=False)
        self.batch_norm_3 = nn.BatchNorm1d(512) #1d due to having FC layer


        ## FC12
        # self.fc12 = nn.Linear(512, 128, bias=False)

        self.fc2 = nn.Linear(512, self.n_classes, bias=False)


        return


    def forward(self, x):

        ## layer 1
        x = self.conv1(x)
        x = self.relu(x)
        x = self.batch_norm_1(x)
        x = self.pool(x)

        ## layer 2
        x = self.conv2(x)
        x = self.relu(x)
        x = self.batch_norm_2(x)
        x = self.pool(x)

        ## layer 3
        x = self.conv3(x)
        x = self.pool(x)

        ## flattening x
        x = x.reshape(x.size(0), -1)

        ## Classification head
        x = self.fc1(x)
        x = self.relu(x)
        x = self.batch_norm_3(x)

        # x = self.fc12(x)
        # x = self.relu(x)
        x = self.dropout_30(x)
        x = self.fc2(x)

        return x


    def load_pretrained(self, pth_file):

        weights_dict = torch.load(pth_file)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in weights_dict.items() if
                           k in model_dict and weights_dict[k].shape == model_dict[k].shape}
        #
        # no_pretrained_dict = {k: v for k, v in model_dict.items() if
        #                    not (k in weights_dict) or weights_dict[k].shape != model_dict[k].shape}

        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

        return


    def train_epoch(self, dataloader, optimizer, criterion, verbose=True):

        train_loss = 0
        train_accuracy = 0
        self.train()

        for index, data in enumerate(dataloader):


            img = data[0]
            label = data[1]

            x = self(img)

            loss = criterion(x, label)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ### adding batch loss into the overall loss
            train_loss += loss

            preds = torch.argmax(x,1)
            ### adding batch loss into the overall loss
            batch_accuracy = sum(preds == label) / x.shape[0]
            train_accuracy += batch_accuracy

            if verbose:
                ### Printing epoch results
                print('Train Epoch: {}/{}\n'
                      'Step: {}/{}\n'
                      'Batch ~ Loss: {:.4f}\n'
                      'Batch ~ Accuracy: {:.4f}\n'.format(self.epoch+1, self.epochs,
                                                      index + 1, len(dataloader),
                                                      loss.data.cpu().numpy(),
                                                      batch_accuracy.data.cpu().numpy()))


        self.train_history["loss"].append(train_loss / (index+1))
        self.train_history["accuracy"].append(train_accuracy / (index+1))

        return



    def val_epoch(self, dataloader, criterion, verbose=True):

           val_loss = 0
           val_accuracy = 0
           self.eval()

           conf_matrix = np.zeros((self.n_classes, self.n_classes))
           with torch.no_grad():

               for index, data in enumerate(dataloader):

                   img = data[0]
                   label = data[1]

                   x = self(img)
                   loss = criterion(x, label)

                   ### adding batch loss into the overall loss
                   val_loss += loss

                   preds = torch.argmax(x, 1)
                   ### adding batch loss into the overall loss
                   batch_accuracy = sum(preds == label) / x.shape[0]
                   val_accuracy += batch_accuracy

                   preds = preds.data.cpu().numpy()
                   labels = label.data.cpu().numpy()

                   for p_index in range(preds.shape[0]):
                       conf_matrix[labels[p_index], preds[p_index]] += 1


                   if verbose:
                       ### Printing epoch results
                       print('Step: {}/{}\n'
                             'Batch ~ Loss: {:.4f}\n'
                             'Batch ~ Accuracy: {:.4f}\n'.format(index + 1, len(dataloader),
                                                                 loss.data.cpu().numpy(),
                                                                 batch_accuracy.data.cpu().numpy()))


           return conf_matrix

    def visualize_graph(self):

        ## Plotting loss
        plt.figure()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss Graph")

        plt.plot(np.arange(len(self.train_history["loss"])), self.train_history["loss"], label="train")

        plt.legend()
        plt.savefig(self.session_name+"loss.png")
        plt.close()


        ## Plotting accyracy
        plt.figure()
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Accuracy Graph")

        plt.plot(np.arange(len(self.train_history["accuracy"])), self.train_history["accuracy"], label="train")

        plt.legend()
        plt.savefig(self.session_name+"accuracy.png")
        plt.close()
