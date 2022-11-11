import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import plot_confusion_matrix
import seaborn as sns
import numpy as np
from Support_Vector_Machine import SVM

dataFrame_A = pd.read_csv('C:/Users/User/Desktop/My Courses/Honours/Biometrics/Project/Facial_Paralysis_Project'
                          '/NewData.csv')

print(dataFrame_A)

# shuffling the data in  the dataframe using sample
shuffled_dataframe = dataFrame_A.sample(frac=1)
print(shuffled_dataframe)
# shuffled_dataframe.str.strip()
# splitting the data into training and test
X = shuffled_dataframe.drop(
    ['Face_Edge', 'Width_Right_Eye', 'Width_Left_Eye', 'Right_Eye_Right_Nose', 'Left_Eye_Left_Nose',
     'Left_Eye_M_Mouth', 'Right_Eye_M_Mouth', 'Left_Nose_M_Mouth', 'Right_Nose_M_Mouth',
     'Left_M_M_Mouth',
     'Right_M_M_Mouth', 'Health_Status', 'Left_Eye_Left_Mouth', 'Right_Eye_Right_Mouth'], axis=1)
y = shuffled_dataframe['Health_Status']

s_scalar = StandardScaler()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, stratify=y, random_state=32)
svClassifier = SVC(kernel='linear', C=10)
svClassifier.fit(X_train, y_train)

y_prediction = svClassifier.predict(X_test)

# generating the confusion matrix
print(confusion_matrix(y_test, y_prediction))
print(classification_report(y_test, y_prediction))
print("Accuracy: ", metrics.accuracy_score(y_test, y_prediction)*100)
# displaying the confusion matrix
# plot_confusion_matrix(svClassifier, X_test, y_test)
#plt.show()

