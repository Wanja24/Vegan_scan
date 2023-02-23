#Import libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import pandas as pd
import pickle

#Load dataset
df = pd.read_csv("data/iris.csv")
df.head()

# just for iris data: fake it to binary
df['variety'] = df['variety'].astype('category')
encode_map = {
    'Setosa': 0,
    'Versicolor': 1,
    'Virginica': 1
}
df['variety'].replace(encode_map, inplace=True)

df.head()

# Split dataset into training set and test set
X = df.iloc[:, 0:-1]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69) # 80% training and 20% test

# todo: adapt hyperparameter
ESTIMATORS = 10

# Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=ESTIMATORS)

# Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

# Predict test data
y_pred=clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# Confusion matrix
confusion = confusion_matrix(y_test, y_pred)
print("Confusion matrix:")
print(confusion)

# save model
pickle.dump(clf, open("saved_models/iris_rf.pkl", "wb"))

# load model
#loaded_model = pickle.load(open("saved_models/iris_rf.pkl", "rb"))

