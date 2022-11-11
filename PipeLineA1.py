import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import plot_confusion_matrix
from xgboost import XGBClassifier
from sklearn import metrics

dataFrame_A = pd.read_csv('C:/Users/User/Desktop/My Courses/Honours/Biometrics/Project/Facial_Paralysis_Project'
                          '/NewData.csv')
print(dataFrame_A)

# shuffling the data in  the dataframe using sample
shuffled_dataframe = dataFrame_A.sample(frac=1)
print(shuffled_dataframe)

# splitting the data into training and test
X = shuffled_dataframe.drop(
    ['Face_Edge', 'Width_Right_Eye', 'Width_Left_Eye', 'Right_Eye_Right_Nose', 'Left_Eye_Left_Nose',
     'Left_Eye_M_Mouth', 'Right_Eye_M_Mouth', 'Left_Nose_M_Mouth', 'Right_Nose_M_Mouth',
     'Left_M_M_Mouth',
     'Right_M_M_Mouth', 'Health_Status', 'Left_Eye_Left_Mouth', 'Right_Eye_Right_Mouth'], axis=1)
y = shuffled_dataframe['Health_Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=7)
model = XGBClassifier()
model.fit(X_train, y_train)
y_predictions = model.predict(X_test)
predictions = [round(value) for value in y_predictions]
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
print(confusion_matrix(y_test, y_predictions))
print(classification_report(y_test, y_predictions))
print("Accuracy: ", metrics.accuracy_score(y_test, predictions)*100)
