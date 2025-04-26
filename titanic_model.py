from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# Checking if train.csv is present and loading it
df = pd.read_csv('train.csv')
print(df.head())
# Load the Titanic dataset
df = pd.read_csv('train.csv')

# Display first 5 rows of the data
print("First 5 Rows:")
print(df.head())

# Show basic information about dataset
print("\nDataset Information:")
print(df.info())

# Show basic statistics (mean, std dev, etc.)
print("\nStatistical Summary:")
print(df.describe())

# Check for missing values
print("\nMissing Values in Each Column:")
print(df.isnull().sum())

# Fill missing values in 'Age' column with median age
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill missing values in 'Embarked' with the most common value (mode)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop the 'Cabin' column completely because too many missing values
df.drop('Cabin', axis=1, inplace=True)

# Drop 'Ticket', 'Name', and 'PassengerId' because they are not needed
df.drop(['Ticket', 'Name', 'PassengerId'], axis=1, inplace=True)

# Convert 'Sex' column: male → 0, female → 1
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Convert 'Embarked' column: S → 0, C → 1, Q → 2
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Final check - preview cleaned data
print("\nCleaned Dataset Preview:")
print(df.head())


# Separate the features (X) and target (y)
X = df.drop('Survived', axis=1)  # Features: all columns except 'Survived'
y = df['Survived']               # Target: Survived column

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Check the shapes to confirm
print("Training Set Shape:", X_train.shape)
print("Testing Set Shape:", X_test.shape)


#step 6

# Initialize the Logistic Regression model
model = LogisticRegression()

# Train (fit) the model on the training data
model.fit(X_train, y_train)

# Predict on the testing data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)

# Show confusion matrix and classification report
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

#random forest 
# Initialize the Random Forest model
rf_model = RandomForestClassifier(random_state=42)

# Train (fit) the Random Forest model
rf_model.fit(X_train, y_train)

# Predict on the testing set
rf_y_pred = rf_model.predict(X_test)

# Evaluate Random Forest model
rf_accuracy = accuracy_score(y_test, rf_y_pred)
print("\nRandom Forest Model Accuracy:", rf_accuracy)

# Confusion Matrix and Classification Report
print("\nRandom Forest Confusion Matrix:")
print(confusion_matrix(y_test, rf_y_pred))

print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_y_pred))



# Import
import seaborn as sns
import matplotlib.pyplot as plt

# Confusion Matrix Plot for Random Forest
conf_matrix = confusion_matrix(y_test, rf_y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')

plt.title('Random Forest Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
