import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier, VotingClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report

# Create a synthetic dataset with 50 samples, 2 features, and 2 classes
X, y = make_classification(n_samples=50, n_features=2, n_informative=2, n_redundant=0, n_classes=2, random_state=42)

# Split the dataset into training and test sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train AdaBoost classifier with default base estimator (Decision Tree)
abc_default = AdaBoostClassifier(n_estimators=50, learning_rate=1, algorithm='SAMME')
model_default = abc_default.fit(X_train, y_train)

# Predict the response for the test dataset using the default model
y_pred_default = model_default.predict(X_test)

# Calculate and print the accuracy of the default model
accuracy_default = metrics.accuracy_score(y_test, y_pred_default)
print("Default AdaBoost Accuracy:", accuracy_default)

# Create and train AdaBoost classifier with Support Vector Classifier as base estimator
svc = SVC(probability=True, kernel='linear')
abc_svc = AdaBoostClassifier(n_estimators=50, estimator=svc, learning_rate=1, algorithm='SAMME')
model_svc = abc_svc.fit(X_train, y_train)

# Predict the response for the test dataset using the SVC model
y_pred_svc = model_svc.predict(X_test)

# Calculate and print the accuracy of the SVC model
accuracy_svc = metrics.accuracy_score(y_test, y_pred_svc)
print("SVC AdaBoost Accuracy:", accuracy_svc)

# Create a Voting Classifier combining both models with soft voting
voting_clf = VotingClassifier(estimators=[
    ('default_ada', model_default),
    ('svc_ada', model_svc)
], voting='soft')

# Train Voting Classifier
voting_clf.fit(X_train, y_train)

# Predict using Voting Classifier
y_pred_voting = voting_clf.predict(X_test)

# Calculate and print the accuracy of the Voting Classifier
accuracy_voting = metrics.accuracy_score(y_test, y_pred_voting)
print("Voting Classifier Accuracy:", accuracy_voting)

# Evaluation Metrics for Default Model
print("\nClassification Report for Default Model:")
print(classification_report(y_test, y_pred_default))

# Evaluation Metrics for SVC Model
print("\nClassification Report for SVC Model:")
print(classification_report(y_test, y_pred_svc))

# Evaluation Metrics for Voting Classifier
print("\nClassification Report for Voting Classifier:")
print(classification_report(y_test, y_pred_voting))

# Confusion Matrices
confusion_default = confusion_matrix(y_test, y_pred_default)
confusion_svc = confusion_matrix(y_test, y_pred_svc)
confusion_voting = confusion_matrix(y_test, y_pred_voting)

# Plotting Confusion Matrices
fig, ax = plt.subplots(1, 3, figsize=(18, 5))
metrics.ConfusionMatrixDisplay(confusion_default).plot(ax=ax[0], cmap='Blues')
ax[0].set_title('Confusion Matrix - Default AdaBoost')

metrics.ConfusionMatrixDisplay(confusion_svc).plot(ax=ax[1], cmap='Blues')
ax[1].set_title('Confusion Matrix - SVC AdaBoost')

metrics.ConfusionMatrixDisplay(confusion_voting).plot(ax=ax[2], cmap='Blues')
ax[2].set_title('Confusion Matrix - Voting Classifier')

plt.show()

# Accuracy Comparison Histogram (Bar Chart)
plt.figure(figsize=(8, 6))
models = ['Default AdaBoost', 'SVC AdaBoost', 'Voting Classifier']
accuracies = [accuracy_default * 100, accuracy_svc * 100, accuracy_voting * 100]

plt.bar(models, accuracies, color=['skyblue', 'lightgreen', 'salmon'])
plt.title('Accuracy Comparison: AdaBoost vs SVC vs Voting Classifier\n')
plt.xlabel('Model')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 100)
plt.grid(axis='y')

# Adding data labels on top of bars
for index, value in enumerate(accuracies):
    plt.text(index, value + 1, f'{value:.2f}%', ha='center')

plt.show()
