import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('decisiontree.csv')

inputs = df.drop('salary_more_then_10k', axis='columns')
target = df['salary_more_then_10k']

le_company = LabelEncoder()
le_job = LabelEncoder()
le_degree = LabelEncoder()
inputs['company_n'] = le_company.fit_transform(inputs['campany'])
inputs['job_n'] = le_job.fit_transform(inputs['job'])
inputs['degree_n'] = le_degree.fit_transform(inputs['degree'])

x = inputs[['company_n', 'job_n', 'degree_n']]
y = target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)

classifier = svm.SVC(kernel='linear', gamma='auto', C=2)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

print("Actual values:", y_test.values)
print("Predicted values:", y_pred)

x_min, x_max = x['company_n'].min() - 1, x['company_n'].max() + 1
y_min, y_max = x['job_n'].min() - 1, x['job_n'].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

Z = classifier.predict(np.c_[xx.ravel(), yy.ravel(), np.zeros_like(xx.ravel())])  
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.5, cmap=plt.cm.coolwarm)
plt.scatter(x['company_n'], x['job_n'], c=y, edgecolors='k', marker='o', cmap=plt.cm.coolwarm)
plt.xlabel('Company (Encoded)')
plt.ylabel('Job (Encoded)')
plt.title('SVM Decision Boundary')
plt.show()
