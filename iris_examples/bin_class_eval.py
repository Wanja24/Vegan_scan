# source for code: https://towardsdatascience.com/pytorch-tabular-binary-classification-a0368da5bb89
from iris_examples.bin_class import *

# todo: put this in a new file when you saved the model
# todo: load model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
num_features = len(X_train.columns)
model = BinaryClassification(num_features)
model.load_state_dict(torch.load("../saved_models/iris_model_1.pt"))

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
print(confusion)
