#Import libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import pandas as pd
import pickle

#Load dataset
# todo: change directory
df = pd.read_csv("food_matrix.csv")
df.head()


# Split dataset into training set and test set
X = df.iloc[:, 0:-1]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69) # 80% training and 20% test

# todo: maybe adapt hyperparameter
ESTIMATORS = 100

# Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=ESTIMATORS)

# Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

# Predict test data
y_pred=clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",(y_test == y_pred).mean())

# Confusion matrix
confusion = confusion_matrix(y_test, y_pred)
print("Confusion matrix:")
print(confusion)

# save model
pickle.dump(clf, open("saved_models/vegan_rf.pkl", "wb"))  # todo: change directory/file name

# load model
#loaded_model = pickle.load(open("saved_models/iris_rf.pkl", "rb"))

