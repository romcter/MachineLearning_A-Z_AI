# Coding Exercise 4: Dataset Splitting and Feature Scaling
# 1: Import necessary Python libraries: pandas, train_test_split from sklearn.model_selection, and StandardScaler from sklearn.preprocessing.
# 2: Load the Iris dataset using Pandas read.csv. Dataset name is titanic.csv.
# 3: Use train_test_split to split the dataset into an 80-20 training-test set.
# 4: Apply random_state with 42 value in train_test_split function for reproducible results.
# 5: Print X_train, X_test, Y_train, and Y_test to understand the dataset split.
# 6: Use StandardScaler to apply feature scaling on the training and test sets.
# 7: Print scaled training and test sets to verify feature scaling.

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv('iris.csv')

X = df.drop('target', axis=1)

y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(X_train, X_test, y_train, y_test)