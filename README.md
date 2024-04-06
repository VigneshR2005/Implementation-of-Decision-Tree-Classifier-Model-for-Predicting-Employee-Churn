# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. import the required libraries.
2. Upload and read the dataset.
3. Check for any null values using the isnull() function.
4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5. Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: R Vignesh
RegisterNumber: 212222230172
*/
```
```python
import pandas as pd
data = pd.read_csv('Employee.csv')
data.head()
data.isnull().sum()
data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["salary"] = le.fit_transform(data["salary"])
data.head()

x = data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y = data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2, random_state = 100)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion = "entropy")
dt.fit(x_train, y_train)
y_pred = dt.predict(x_test)

from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
## DATA HEAD:
![ML21](https://github.com/Senthamil1412/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119120228/76857cf4-cbce-48e2-9faf-08e2d96e005d)

## NULL VALUES:
![ml22](https://github.com/Senthamil1412/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119120228/eb902294-9d3f-4d06-804a-19f39ca4f704)


## ASSIGNMENT OF X VALUES:
![ML23](https://github.com/Senthamil1412/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119120228/ab22e7c5-8091-4d8b-8e12-69f0e083b2a6)


## ASSIGNMENT OF Y VALUES:
![ML24](https://github.com/Senthamil1412/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119120228/3fda7d90-de30-4cf1-997d-34356eaacc21)


## Converting string literals to numerical values using label encoder :
![ML25](https://github.com/Senthamil1412/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119120228/52a00f61-a082-4564-b04c-a3ad0b2dca1c)




## Accuracy :
![ML26](https://github.com/Senthamil1412/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119120228/6ebf81f4-85d3-4689-94ab-2261ff81b7cd)




## Prediction :
![ML27](https://github.com/Senthamil1412/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119120228/663ad3ed-38f2-4395-8d30-ef301b43eb12)







## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
