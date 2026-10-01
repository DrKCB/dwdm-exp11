
import numpy as np 
import pandas as pd 
from sklearn import datasets 
from sklearn.model_selection import train_test_split 
from sklearn.naive_bayes import GaussianNB 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix 
# Load the Iris dataset 
iris = datasets.load_iris() 
X = iris.data  # Features 
y = iris.target  # Target variable (species) 
# Split the dataset into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
# Initialize the Gaussian Naïve Bayes classifier 
model = GaussianNB() 

# Fit the model on the training data 
model.fit(X_train, y_train)

# Make predictions on the test data 
y_pred = model.predict(X_test) 
 
# Evaluate the model 
accuracy = accuracy_score(y_test, y_pred) 
cm = confusion_matrix(y_test, y_pred) 
report = classification_report(y_test, y_pred) 
 
# Output the results 
print("Accuracy:", accuracy) 
print("\nConfusion Matrix:\n", cm) 
print("\nClassification Report:\n", report)