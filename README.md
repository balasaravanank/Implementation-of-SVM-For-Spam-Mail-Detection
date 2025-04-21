## Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import libraries.

2.Read the CSV file and display data using head().

3.Split the dataset using train_test_split().

4.Calculate predictions and accuracy.

5.Print the outputs.

6.End the program.

## Program:
```
Program to implement the SVM For Spam Mail Detection..

import chardet, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn import metrics

# Detect encoding
with open('spam.csv', 'rb') as f:
    print(chardet.detect(f.read(100000)))

# Load data
data = pd.read_csv('spam.csv', encoding='windows-1252')
print(data.head())
print(data.info())
print(data.isnull().sum())

# Split data
x = data['v1'].values   # Labels
y = data['v2'].values   # Messages
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Text vectorization
cv = CountVectorizer()
x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)

# Train & predict
model = SVC()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print("Predictions:", y_pred)

# Accuracy
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
```

## Output:
## ENCODING DETECTED
![Screenshot 2025-04-21 232423](https://github.com/user-attachments/assets/102b5f40-76b4-4389-84f6-8b081d83f402)
## FIRST FEW ROWS, DATA INFO, MISSING VALUES
![Screenshot 2025-04-21 232432](https://github.com/user-attachments/assets/eb57e1bb-d8b2-499d-a0d1-28193c783037)
## PREDICTED LABELS
![Screenshot 2025-04-21 232438](https://github.com/user-attachments/assets/496d3781-2560-47ed-bc38-a7b50c42435a)
## MODEL ACCURACY
![Screenshot 2025-04-21 232441](https://github.com/user-attachments/assets/cae6295e-3af8-4c82-89d4-ef45896c2122)


## Developed by : BALA SARAVANAN K
## Reg no: 24900611
## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
