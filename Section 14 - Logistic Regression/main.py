import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
# np.set_printoptions(precision=2)
# print(y_test.reshape(len(y_test), 1))
# print("=============================")
# print(y_pred.reshape(len(y_pred), 1))
# print("=============================")
# print(classifier.predict(scaler.transform([[30,87000]])))
print(np.concatenate((y_test.reshape(len(y_test), 1), y_pred.reshape(len(y_pred), 1)), 1))

confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
cm_display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[False, True])
accuracy_score(y_test, y_pred)

cm_display.plot()
plt.show()

# X_grid = np.arange(min(X), max(X), 0.01)
# X_grid = X_grid.reshape((len(X_grid), 1))
# plt.scatter(X, y, color='red')
# plt.plot(X_grid, regressor.predict(X_grid), color='blue')
# plt.title('Truth or Bluff (Random Forest Regressor)')
# plt.xlabel('Position level')
# plt.ylabel('Salary')
# plt.show()
