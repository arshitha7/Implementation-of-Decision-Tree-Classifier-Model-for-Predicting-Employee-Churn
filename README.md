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
![image](https://github.com/arshitha7/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/144979143/d6580fa8-13d0-408e-88c3-632e0796d46a)

### Data.info():
![image](https://github.com/arshitha7/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/144979143/cefca8e2-2b5a-4ccf-bf8b-332932e10ee4)

### isnull() and sum():
![image](https://github.com/arshitha7/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/144979143/b6359d32-8b08-4486-9929-2b0d041c7862)

### Data Value Counts():
![image](https://github.com/arshitha7/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/144979143/9c4f9d91-ac2e-42cd-9c19-3a5af4427d60)

### Data.head() for salary:
![image](https://github.com/arshitha7/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/144979143/706fe879-72d5-44fe-a823-b63edaeb47a1)

### x.head:
![image](https://github.com/arshitha7/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/144979143/8ca2c3ef-8efa-4d0a-afb8-647a9ff55751)

### Accuracy Value:
![image](https://github.com/arshitha7/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/144979143/8367b0a1-41c4-4f7c-ab71-42b088043a53)

### Data Prediction:
![image](https://github.com/arshitha7/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/144979143/cc6a39bf-de06-4c02-9c3b-fac826c35a06)



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
