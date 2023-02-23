import torch
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from vegan_bin_class import X_train, LEARNING_RATE, BinaryClassification, test_loader, train_loader, y_test

# load model
# alternative: model.load_state_dict(torch.load("saved_models/vegan_model_2.pt"))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
num_features = len(X_train.columns)
model = BinaryClassification(num_features)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
checkpoint = torch.load('../saved_models/vegan_model_3.1.tar')  # todo: change directory/file name
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
epoch_loss = checkpoint['epoch_loss']
epoch_acc = checkpoint['epoch_acc']

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
confusion = confusion_matrix(y_test, y_pred_list)
test_acc = (y_pred_list == y_test).mean()*100  # todo: correct accuracy calculation?
print(f'train accuracy: {(epoch_acc/len(train_loader))}')
print(f'test accuracy: {test_acc}')
print('Confusion matrix for test data:')
print(confusion)

