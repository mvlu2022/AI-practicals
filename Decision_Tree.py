import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Load a sample dataset from iris
iris=load_iris()
x=iris.data
y=iris.target
x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.1, random_state=42)

# classification of decision tree
clf = DecisionTreeClassifier(random_state=42)
# Train the model
clf.fit(x_train,y_train)

# predictions
y_pred=clf.predict(x_test)

# print accuracy
accuracy=np.mean(y_pred==y_test)
print(f'Accuracy:{accuracy*100:.2f}%')
plt.figure(figsize=(12,8))
plot_tree(clf,filled=True, feature_names=iris.feature_names,class_names=iris.target_names)
plt.title("Decision Tree Visualization")
plt.show()