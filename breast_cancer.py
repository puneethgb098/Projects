from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt

# Load the breast cancer dataset
breast_cancer = datasets.load_breast_cancer()

print(list(breast_cancer.keys()))
print(breast_cancer['data'].shape)
print(breast_cancer['target'])
print(breast_cancer['DESCR'])

# Select features and target
X = breast_cancer["data"]
y = breast_cancer["target"]

# Train a logistic regression classifier
clf = LogisticRegression()
clf.fit(X, y)

# Make a prediction
example = clf.predict([[13.08, 15.71, 85.63, 520.0, 0.1075, 0.127, 0.04568, 0.0311, 0.1967, 0.06811, 0.1852, 0.7477, 1.383, 14.67, 0.004097, 0.01898, 0.01698, 0.00649, 0.01678, 0.002425, 14.5, 20.49, 96.09, 630.5, 0.1312, 0.2776, 0.189, 0.07283, 0.3184, 0.08183]])
print(example)

# Generate input data for visualization
X_new = np.linspace(0, 30, 1000)  # Adjust the range based on your data
X_test = np.array([[x] * 30 for x in X_new])  # Repeat the X_new values for all features
y_prob = clf.predict_proba(X_test)

# Plot the visualization
plt.plot(X_new, y_prob[:, 0], "r-", label="Malignant")
plt.plot(X_new, y_prob[:, 1], "g-", label="Benign")
plt.legend()
plt.show()
