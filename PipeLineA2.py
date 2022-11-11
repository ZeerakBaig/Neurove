from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot
import pandas as pd
from sklearn.metrics import plot_confusion_matrix
from sklearn import metrics
# create dataset
dataFrame_A = pd.read_csv('C:/Users/User/Desktop/My Courses/Honours/Biometrics/Project/Facial_Paralysis_Project'
                          '/NewData.csv')
print(dataFrame_A)

# shuffling the data in  the dataframe using sample
shuffled_dataframe = dataFrame_A.sample(frac=1)
print(shuffled_dataframe)

X = shuffled_dataframe.drop(
    ['Face_Edge', 'Width_Right_Eye', 'Width_Left_Eye', 'Right_Eye_Right_Nose', 'Left_Eye_Left_Nose',
     'Left_Eye_M_Mouth', 'Right_Eye_M_Mouth', 'Left_Nose_M_Mouth', 'Right_Nose_M_Mouth',
     'Left_M_M_Mouth',
     'Right_M_M_Mouth', 'Health_Status', 'Left_Eye_Left_Mouth', 'Right_Eye_Right_Mouth'], axis=1)
y = shuffled_dataframe['Health_Status']
# split into train test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# define lists to collect scores
train_scores, test_scores = list(), list()
# define the tree depths to evaluate
values = [i for i in range(1, 51)]
# evaluate a decision tree for each depth
for i in values:
    # configure the model
    model = KNeighborsClassifier(n_neighbors=i)
    # fit model on the training dataset
    model.fit(X_train, y_train)
    # evaluate on the train dataset
    train_yhat = model.predict(X_train)
    train_acc = accuracy_score(y_train, train_yhat)
    train_scores.append(train_acc)
    # evaluate on the test dataset
    test_yhat = model.predict(X_test)
    test_acc = accuracy_score(y_test, test_yhat)
    test_scores.append(test_acc)
    # summarize progress
    #print('>%d, train: %.3f, test: %.3f' % (i, train_acc, test_acc))
# plot of train and test scores vs number of neighbors
pyplot.plot(values, train_scores, '-o', label='Train')
pyplot.plot(values, test_scores, '-o', label='Test')
pyplot.legend()
#pyplot.show()
print(confusion_matrix(y_test, test_yhat))
print(classification_report(y_test, test_yhat))
print("Accuracy: ", metrics.accuracy_score(y_test, test_yhat)*100)
