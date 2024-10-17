import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
iris=load_iris()
print(iris)
x=iris.data
y=iris.target
y_test,y_train,x_test,x_train=train_test_split(y,x,test_size=0.1,random_state=100)
clf=DecisionTreeClassifier(random_state=42)
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
print("Y prediction: ",y_pred)
accuracy=np.mean(y_pred==y_test)
print(f'Accuracy:{accuracy*100:.3f}%')
plt.figure(figsize=(12,8))
plot_tree(clf,filled=True,feature_names=iris.feature_names,class_names=iris.target_names)
plt.title("Decision Tree Visualization")
plt.show()