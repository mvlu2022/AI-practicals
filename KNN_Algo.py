import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report

X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, n_classes=2, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


k = 3
knn = KNeighborsClassifier(n_neighbors=k)

# Train the model
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

accuracy = metrics.accuracy_score(y_test, y_pred)
print("K-NN Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

confusion = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(confusion)

plt.figure(figsize=(6, 5))
metrics.ConfusionMatrixDisplay(confusion).plot(cmap='Blues')
plt.title('Confusion Matrix - K-NN')
plt.show()

plt.figure(figsize=(8, 6))
models = ['K-NN']
accuracies = [accuracy * 100]

plt.bar(models, accuracies, color=['skyblue'])
plt.title('Accuracy of K-NN Classifier')
plt.xlabel('Model')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 100)
plt.grid(axis='y')
plt.show()

