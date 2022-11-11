from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

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


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

randomForest = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None, criterion='gini',
                                      max_depth=None, max_features='sqrt', max_leaf_nodes=None, max_samples=None,
                                      min_impurity_decrease=0.0, min_samples_leaf=1, min_samples_split=2,
                                      min_weight_fraction_leaf=0.0, n_estimators=10000, n_jobs=None, oob_score=False,
                                      random_state=None, verbose=0, warm_start=False)
# fit function is used to train the model with training set
randomForest.fit(X_train, y_train)
prediction = randomForest.predict(X_test)
print(randomForest.get_params())

print(confusion_matrix(y_test, prediction))
print(classification_report(y_test, prediction))
print("Accuracy: ", metrics.accuracy_score(y_test, prediction) * 100)
# scores = cross_val_score(randomForest, X, y, cv=5)
# print(scores)
