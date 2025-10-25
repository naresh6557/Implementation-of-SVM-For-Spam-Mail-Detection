# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Detect File Encoding: Use chardet to determine the dataset's encoding.
2. Load Data: Read the dataset with pandas.read_csv using the detected encoding.
3. Inspect Data: Check dataset structure with .info() and missing values with .isnull().sum().
4. Split Data: Extract text (x) and labels (y) and split into training and test sets using train_test_split.
5. Convert Text to Numerical Data: Use CountVectorizer to transform text into a sparse matrix.
6. Train SVM Model: Fit an SVC model on the training data.
7. Predict Labels: Predict test labels using the trained SVM model.
8. Evaluate Model: Calculate and display accuracy with metrics.accuracy_score.

## Program:
```
Program to implement the SVM For Spam Mail Detection..
Developed by: Naresh kumar r
RegisterNumber:  212224040213
```
```
import chardet
file='spam.csv'
with open(file, 'rb') as rawdata:
    result = chardet.detect (rawdata.read(100000))
result
```
```
import pandas as pd
data=pd.read_csv('spam.csv', encoding='Windows-1252')
```
```
data.info()
```
```
data.isnull().sum()
```
```
x=data["v1"].values
y=data["v2"].values
```
```
from sklearn.model_selection import train_test_split
x_train, x_test, y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=0)
```
```
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
```
```
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
```
```
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train, y_train)
y_pred=svc.predict(x_test)
y_pred
```
```
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```
## Output:
<img width="1226" height="198" alt="image" src="https://github.com/user-attachments/assets/466269ae-94f0-46f1-8a12-6e125def101f" />

<img width="1376" height="405" alt="image" src="https://github.com/user-attachments/assets/08928159-1890-475d-a4e1-2ebb6e630386" />

<img width="1133" height="354" alt="image" src="https://github.com/user-attachments/assets/ca9679f9-7a1b-4bfe-9283-e5ac33d6bbd3" />

<img width="1129" height="230" alt="image" src="https://github.com/user-attachments/assets/3aa6bae2-6547-40cd-bf2c-c037ebde1ce2" />


<img width="1260" height="145" alt="image" src="https://github.com/user-attachments/assets/49d45cde-e08d-4493-b82d-bab61b163ee1" />

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
