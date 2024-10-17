import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report


df = pd.read_csv('numeric_scv.csv')
target=df['salary_more_then_10k']

X = df[['job', 'degree']] 
y = target                 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

base_estimator = DecisionTreeClassifier(max_depth=1)
classifier = AdaBoostClassifier(estimator=base_estimator, n_estimators=50, random_state=4)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print("Actual values:", y_test.values)
print("Predicted values:", y_pred)

print(classification_report(y_test, y_pred))

x_min, x_max = X['job'].min() - 1, X['degree'].max() + 1
y_min, y_max = X['job'].min() - 1, X['degree'].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.5, cmap=plt.cm.coolwarm)
plt.scatter(X['job'], X['degree'], c=y, edgecolors='k', marker='o', cmap=plt.cm.coolwarm)
plt.xlabel('Job')
plt.ylabel('Degree')
plt.title('AdaBoost Decision Boundary')
plt.show()
