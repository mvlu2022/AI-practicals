import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

try:
    df = pd.read_csv('numeric_scv1.csv')
except FileNotFoundError:
    print("Error: The file 'knn_data.csv' was not found.")
    exit()

print("DataFrame preview:")
print(df.head())

X = df[['feature1', 'feature2']]  
y = df['target']                  


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

classifier = KNeighborsClassifier(n_neighbors=3) 
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print("Actual values:", y_test.values)
print("Predicted values:", y_pred)

print(classification_report(y_test, y_pred))

# Visualization
x_min, x_max = X['feature1'].min() - 1, X['feature1'].max() + 1
y_min, y_max = X['feature2'].min() - 1, X['feature2'].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.5, cmap=plt.cm.coolwarm)
plt.scatter(X['feature1'], X['feature2'], c=y, edgecolors='k', marker='o', cmap=plt.cm.coolwarm)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('KNN Decision Boundary')
plt.show()
