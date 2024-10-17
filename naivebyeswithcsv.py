import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report

try:
    df = pd.read_csv('numeric_scv.csv')
except FileNotFoundError:
    print("Error: The file 'numeric_scv.csv' was not found.")
    exit()

print("DataFrame preview:")
print(df.head())

X = df[['job', 'degree']]  
target=df['salary_more_then_10k']
y = target                  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

classifier = GaussianNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print("Actual values:", y_test.values)
print("Predicted values:", y_pred)

print(classification_report(y_test, y_pred))

# Visualization
x_min, x_max = X['job'].min() - 1, X['job'].max() + 1
y_min, y_max = X['degree'].min() - 1, X['degree'].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.5, cmap=plt.cm.coolwarm)
plt.scatter(X['job'], X['degree'], c=y, edgecolors='k', marker='o', cmap=plt.cm.coolwarm)
plt.xlabel('Job')
plt.ylabel('Degree')
plt.title('Naive Bayes Decision Boundary')
plt.show()
