# Coding exercise 5: Feature scaling for Machine Learning
# Import the necessary libraries for data preprocessing, including the StandardScaler and train_test_split classes.
#
# Load the "Wine Quality Red" dataset into a pandas DataFrame. You can use the pd.read_csv function for this. Make sure you set the correct delimeter for the file.
#
# Split your dataset into an 80-20 training-test set. Set random_state to 42 to ensure reproducible results.
#
# Create an instance of the StandardScaler class.
#
# Fit the StandardScaler on features from the training set, excluding the target variable 'Quality'.
#
# Use the "fit_transform" method of the StandardScaler object on the training dataset.
#
# Apply the "transform" method of the StandardScaler object on the test dataset.
#
# Print your scaled training and test datasets to verify the feature scaling process.

# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Load the Wine Quality Red dataset
dataset = pd.read_csv('winequality-red.csv', delimiter=';')

# Separate features and target
X = dataset.drop('quality', axis=1)
y = dataset['quality']

# Split the dataset into an 80-20 training-test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Create an instance of the StandardScaler class
scaler = StandardScaler()

# Fit the StandardScaler on the features from the training set and transform it
X_train = scaler.fit_transform(X_train)

# Apply the transform to the test set
X_test = scaler.transform(X_test)

# Print the scaled training and test datasets
print(X_train, X_test, y_train, y_test)