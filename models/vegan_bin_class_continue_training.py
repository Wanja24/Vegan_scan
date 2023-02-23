# import libraries
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from sklearn.metrics import confusion_matrix

warnings.simplefilter(action='ignore', category=FutureWarning)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split

# read data
# todo: change directory
df = pd.read_csv("../food_matrix.csv")
df.head()


# features
X = df.iloc[:, 0:-1]
# labels
y = df.iloc[:, -1]
# split data into training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)

# hyperparameters
# todo: adapt hyperparameters
EPOCHS = 3
BATCH_SIZE = 64
LEARNING_RATE = 0.1


# train data
class TrainData(Dataset):

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


train_data = TrainData(torch.tensor(X_train.values).to(torch.float32),
                       torch.tensor(y_train.values).to(torch.float32))


# test data
class TestData(Dataset):

    def __init__(self, X_data):
        self.X_data = X_data

    def __getitem__(self, index):
        return self.X_data[index]

    def __len__(self):
        return len(self.X_data)


test_data = TestData(torch.tensor(X_test.values).to(torch.float32))

# create DataLoaders
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=1)

# Binary Classification network architecture
class BinaryClassification(nn.Module):
    def __init__(self, inputs_size):
        super(BinaryClassification, self).__init__()
        self.layer_1 = nn.Linear(inputs_size, 64)
        self.layer_2 = nn.Linear(64, 64)
        self.layer_out = nn.Linear(64, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.batchnorm2 = nn.BatchNorm1d(64)

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)

        return x

# accuracy calculation
def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc


if __name__ == "__main__" :

    # select device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # select model, loss function and optimizer
    num_features = len(X_train.columns)
    model = BinaryClassification(num_features)
    model.to(device)
    print(model)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # load a previously trained model to continue training
    checkpoint = torch.load('../saved_models/vegan_model_2.1.tar')  # todo: change directory/file name
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']

    losses = []

    # train model for several epochs on training data
    # thereby track the batch loss, calculate epoch loss and epoch accuracy
    model.train()
    for e in range(epoch+1, epoch+EPOCHS+1):
        epoch_loss = 0
        epoch_acc = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            y_pred = model(X_batch)

            loss = criterion(y_pred, y_batch.unsqueeze(1))
            acc = binary_acc(y_pred, y_batch.unsqueeze(1))
            # acc = (((y_pred.sigmoid() > 0.5) == y_batch).float().mean()).item()

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc

            losses.append(loss.item())

        print(f'Epoch {e + 0:03}: | Loss: {epoch_loss / len(train_loader):.5f} | Acc: {epoch_acc / len(train_loader):.3f}')

        # save checkpoint of the model after each epoch
        torch.save({
            'epoch': e,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch_loss': epoch_loss,
            'epoch_acc': epoch_acc
        }, f'../saved_models/vegan_model_cont_1.{e}.tar')  # todo: change directory/file name

    # plot the loss
    plt.plot(losses)
    plt.title('loss vs batches')
    plt.xlabel('batches')
    plt.ylabel('loss')
    plt.show()

    # save model (alternatives)
    # torch.save(model, 'saved_models/vegan_model_1.pt')
    # torch.save(model.state_dict(), 'saved_models/vegan_model_1.pt')

"""
# put this in a new file when you saved the model
    y_pred_list = []
    model.eval()
    with torch.no_grad():
        for X_batch in test_loader:
            X_batch = X_batch.to(device)
            y_test_pred = model(X_batch)
            y_test_pred = torch.sigmoid(y_test_pred)
            y_pred_tag = torch.round(y_test_pred)
            y_pred_list.append(y_pred_tag.cpu().numpy())

    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    confusion_matrix(y_test, y_pred_list)
"""