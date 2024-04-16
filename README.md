# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Upload and read the dataset.
3. Check for any null values using the isnull() function.
4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5. Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: ARSHITHA MS
RegisterNumber: 212223240015 
*/
import pandas as pd

# Load the dataset
data = pd.read_csv("Employee.csv")

# Display the first few rows of the dataset
data.head()

# Get information about the dataset
data.info()

# Check for missing values
data.isnull().sum()

# Count the number of employees who left and stayed
data["left"].value_counts()

# Encode the 'salary' column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["salary"] = le.fit_transform(data["salary"])

# Display the first few rows of the modified dataset
data.head()

# Select features (independent variables)
x = data[["satisfaction_level", "last_evaluation", "number_project", "average_montly_hours", "time_spend_company", "Work_accident", "promotion_last_5years", "salary"]]
x.head()  # No departments and no left

# Select the target variable (dependent variable)
y = data["left"]

# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)

# Train the Decision Tree classifier
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train, y_train)

# Predict the target variable for the test set
y_pred = dt.predict(x_test)

# Calculate the accuracy of the model
from sklearn import metrics
accuracy = metrics.accuracy_score(y_test, y_pred)
accuracy

# Predict whether an employee will leave based on given features
dt.predict([[0.5, 0.8, 9, 260, 6, 0, 1, 2]])
```

## Output:
### Head:
### Data.info():
### isnull() and sum():
### Data Value Counts():
### Data.head() for salary:
### x.head:
### Accuracy Value:
### Data Prediction:



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
