"""Importing Libraries"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


"""Data Collection and Processing"""

sonar_data=pd.read_csv("/content/Copy of sonar data.csv",header=None)
sonar_data.head()
sonar_data.shape
sonar_data.describe()
sonar_data[60].value_counts()
sonar_data.groupby(60).mean()


#splitting data and labels

X=sonar_data.drop(columns=60,axis=1)
Y=sonar_data[60]
print(X)
print(Y)


"""Splitting Training and Test data"""

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1,stratify=Y,random_state=1)
print(X.shape,X_train.shape,X_test.shape,Y.shape,Y_train.shape,Y_test.shape)
print(Y_test)


"""Training the Model"""

model=LogisticRegression()
model.fit(X_train,Y_train)


"""Model Evaluation"""

X_train_prediction=model.predict(X_train)
training_acc=accuracy_score(X_train_prediction,Y_train)
print('Accuracy on training data :',training_acc)

X_test_prediction=model.predict(X_test)
test_acc=accuracy_score(X_test_prediction,Y_test)
print("Accuracy on test data :",test_acc)


"""Making a Predictive System"""

input_data=(0.0453,0.0523,0.0843,0.0689,0.1183,0.2583,0.2156,0.3481,0.3337,0.2872,0.4918,0.6552,0.6919,0.7797,0.7464,0.9444,1.0000,0.8874,0.8024,0.7818,0.5212,0.4052,0.3957,0.3914,0.3250,0.3200,0.3271,0.2767,0.4423,0.2028,0.3788,0.2947,0.1984,0.2341,0.1306,0.4182,0.3835,0.1057,0.1840,0.1970,0.1674,0.0583,0.1401,0.1628,0.0621,0.0203,0.0530,0.0742,0.0409,0.0061,0.0125,0.0084,0.0089,0.0048,0.0094,0.0191,0.0140,0.0049,0.0052,0.0044)
input_data_as_numpy_array=np.asarray(input_data)
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
prediction=model.predict(input_data_reshaped)
print(prediction)
if prediction[0]=='R':
  print("The object is a rock")
else:
  print("The object is a mine")


