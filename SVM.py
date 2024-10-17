import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics

# Create a synthetic dataset
X, y = make_blobs(n_samples=100, centers=2, random_state=42, cluster_std=1.0)

# Function to visualize confusion matrix
def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# Function to plot decision boundaries
def plot_decision_boundaries(X, y, model, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='k', s=100)
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(scatter, label='Classes')
    plt.show()

# Step 1: Classification before adding noise
print("Classification before adding noise:")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train the SVM model (Linear)
svm_model_linear = SVC(kernel='linear', C=1)
svm_model_linear.fit(X_train, y_train)

# Make predictions
y_pred_linear = svm_model_linear.predict(X_test)

# Evaluate the model
print("Accuracy before adding noise (Linear):", metrics.accuracy_score(y_test, y_pred_linear))
print("F1 Score before adding noise (Linear):", metrics.f1_score(y_test, y_pred_linear, average='macro'))
print("Precision before adding noise (Linear):", metrics.precision_score(y_test, y_pred_linear, average='macro'))
print("Recall before adding noise (Linear):", metrics.recall_score(y_test, y_pred_linear, average='macro'))

# Confusion Matrix for Linear model
conf_matrix_linear = metrics.confusion_matrix(y_test, y_pred_linear)
print("\nConfusion Matrix (Before Noise - Linear):")
print(conf_matrix_linear)

# Plot decision boundaries for Linear model
plot_decision_boundaries(X_train, y_train, svm_model_linear, title='Decision Boundaries (Linear Model)')

# Step 2: Classification after adding noise
print("\nClassification after adding noise:")

# Add random noise to the data
np.random.seed(42)
noise = np.random.normal(0, 1.0, X.shape)  # More significant noise
X_noisy = X + noise

# Split the noisy data into training and testing sets
X_train_noisy, X_test_noisy, y_train, y_test = train_test_split(X_noisy, y, test_size=0.3, random_state=42)

# Create and train the SVM model on noisy data (Linear)
svm_model_noisy_linear = SVC(kernel='linear', C=1)
svm_model_noisy_linear.fit(X_train_noisy, y_train)

# Make predictions on the noisy test set (Linear)
y_pred_noisy_linear = svm_model_noisy_linear.predict(X_test_noisy)

# Evaluate the model
print("Accuracy after adding noise (Linear):", metrics.accuracy_score(y_test, y_pred_noisy_linear))
print("F1 Score after adding noise (Linear):", metrics.f1_score(y_test, y_pred_noisy_linear, average='macro'))
print("Precision after adding noise (Linear):", metrics.precision_score(y_test, y_pred_noisy_linear, average='macro'))
print("Recall after adding noise (Linear):", metrics.recall_score(y_test, y_pred_noisy_linear, average='macro'))

# Confusion Matrix for Noisy Linear model
conf_matrix_noisy_linear = metrics.confusion_matrix(y_test, y_pred_noisy_linear)
print("\nConfusion Matrix (After Noise - Linear):")
print(conf_matrix_noisy_linear)

# Plot decision boundaries for Noisy Linear model
plot_decision_boundaries(X_train_noisy, y_train, svm_model_noisy_linear, title='Decision Boundaries (Noisy Linear Model)')

# Create and train the SVM model on noisy data (Non-Linear)
svm_model_noisy_non_linear = SVC(kernel='rbf', C=1)
svm_model_noisy_non_linear.fit(X_train_noisy, y_train)

# Make predictions on the noisy test set (Non-Linear)
y_pred_noisy_non_linear = svm_model_noisy_non_linear.predict(X_test_noisy)

# Evaluate the non-linear model
print("Accuracy after adding noise (Non-Linear):", metrics.accuracy_score(y_test, y_pred_noisy_non_linear))
print("F1 Score after adding noise (Non-Linear):", metrics.f1_score(y_test, y_pred_noisy_non_linear, average='macro'))
print("Precision after adding noise (Non-Linear):", metrics.precision_score(y_test, y_pred_noisy_non_linear, average='macro'))
print("Recall after adding noise (Non-Linear):", metrics.recall_score(y_test, y_pred_noisy_non_linear, average='macro'))

# Confusion Matrix for Noisy Non-Linear model
conf_matrix_noisy_non_linear = metrics.confusion_matrix(y_test, y_pred_noisy_non_linear)
print("\nConfusion Matrix (After Noise - Non-Linear):")
print(conf_matrix_noisy_non_linear)

# Visualize confusion matrices
plot_confusion_matrix(conf_matrix_linear, title='Confusion Matrix (Before Noise - Linear)')
plot_confusion_matrix(conf_matrix_noisy_linear, title='Confusion Matrix (After Noise - Linear)')
plot_confusion_matrix(conf_matrix_noisy_non_linear, title='Confusion Matrix (After Noise - Non-Linear)')
