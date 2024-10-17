# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Loading the dataset from a URL
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
col_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
dataset = pd.read_csv(url, names=col_names)

# Selecting features and target variable
X = dataset.iloc[:, [0, 3]].values  # Features: Sepal Length and Petal Width
y = dataset.iloc[:, -1].values        # Target: Species

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Feature Scaling using MinMaxScaler to avoid negative values
sc = MinMaxScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the Gaussian Naive Bayes model on the Training set
gaussian_classifier = GaussianNB()
gaussian_classifier.fit(X_train, y_train)

# Predicting the Test set results with Gaussian Naive Bayes
y_pred_gaussian = gaussian_classifier.predict(X_test)

# Training the Multinomial Naive Bayes model on the Training set
multinomial_classifier = MultinomialNB()
multinomial_classifier.fit(X_train, y_train)

# Predicting the Test set results with Multinomial Naive Bayes
y_pred_multinomial = multinomial_classifier.predict(X_test)

# Evaluating Gaussian Naive Bayes model
print("Gaussian Naive Bayes Predictions: ", y_pred_gaussian)
print("~" * 20)
ac_gaussian = accuracy_score(y_test, y_pred_gaussian)
print("Gaussian Model Accuracy: ", ac_gaussian * 100, "%")
print("~" * 20)
cm_gaussian = confusion_matrix(y_test, y_pred_gaussian)
print("Gaussian Model Confusion Matrix: ")
print(cm_gaussian)
print("Classification Report for Gaussian Naive Bayes:")
print(classification_report(y_test, y_pred_gaussian, zero_division=0))  # Add zero_division here

# Evaluating Multinomial Naive Bayes model
print("Multinomial Naive Bayes Predictions: ", y_pred_multinomial)
print("~" * 20)
ac_multinomial = accuracy_score(y_test, y_pred_multinomial)
print("Multinomial Model Accuracy: ", ac_multinomial * 100, "%")
print("~" * 20)
cm_multinomial = confusion_matrix(y_test, y_pred_multinomial)
print("Multinomial Model Confusion Matrix: ")
print(cm_multinomial)
print("Classification Report for Multinomial Naive Bayes:")
print(classification_report(y_test, y_pred_multinomial, zero_division=0))  # Add zero_division here

# Visualization of Confusion Matrices
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.heatmap(cm_gaussian, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title('Gaussian Naive Bayes Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

plt.subplot(1, 2, 2)
sns.heatmap(cm_multinomial, annot=True, fmt='d', cmap='Greens', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title('Multinomial Naive Bayes Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

plt.tight_layout()
plt.show()

# Visualizing Accuracy Over Time (for both models)
plt.figure(figsize=(8, 4))
models = ['Gaussian Naive Bayes', 'Multinomial Naive Bayes']
accuracies = [ac_gaussian * 100, ac_multinomial * 100]
plt.bar(models, accuracies, color=['skyblue', 'lightgreen'])
plt.ylim(0, 100)
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy (%)')
plt.show()
