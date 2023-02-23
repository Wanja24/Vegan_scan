# source for code: https://towardsdatascience.com/pytorch-tabular-binary-classification-a0368da5bb89
import pandas as pd
import warnings

from sklearn.metrics import confusion_matrix

warnings.simplefilter(action='ignore', category=FutureWarning)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split


df = pd.read_csv("../data/iris.csv")
df.head()

# just for iris data: fake it to binary
df['variety'] = df['variety'].astype('category')
encode_map = {
    'Setosa': 0,
    'Versicolor': 1,
    'Virginica': 1
}
df['variety'].replace(encode_map, inplace=True)


X = df.iloc[:, 0:-1]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)

EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.1


## train data
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


## test data
class TestData(Dataset):

    def __init__(self, X_data):
        self.X_data = X_data

    def __getitem__(self, index):
        return self.X_data[index]

    def __len__(self):
        return len(self.X_data)


test_data = TestData(torch.tensor(X_test.values).to(torch.float32))

train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=1)


class BinaryClassification(nn.Module):
    def __init__(self, inputs_size):
        super(BinaryClassification, self).__init__()
        # Number of input features is 12.
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


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc


if __name__ == "__main__" :

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    num_features = len(X_train.columns)
    model = BinaryClassification(num_features)
    model.to(device)
    print(model)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()
    for e in range(1, EPOCHS + 1):
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

        print(f'Epoch {e + 0:03}: | Loss: {epoch_loss / len(train_loader):.5f} | Acc: {epoch_acc / len(train_loader):.3f}')

    # todo: save model
    # torch.save(model, 'saved_models/iris_model_1.pt')
    # torch.save(model.state_dict(), 'saved_models/iris_model_1.pt')

    # todo: put this in a new file when you saved the model
    # todo: load model
"""
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