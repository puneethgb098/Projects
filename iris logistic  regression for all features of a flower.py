from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt

iris = datasets.load_iris()

print(list(iris.keys()))
print(iris['data'].shape)
print(iris['target'])
print(iris['DESCR'])

#takes all the features
X = iris["data"]  
y = iris["target"]

# Train a logistic regression classifier
clf = LogisticRegression()
clf.fit(X, y)

#sepal lenght,width,petal length,width
example = clf.predict([[7.9,4.4,6.9,2.5]])
print(example)

# Using matplotlib to plot the visualization
X_new = np.linspace(0, 8, 1000)  # Adjust the range based on your data
X_test = np.array([[x, x, x, x] for x in X_new])  # Repeat the X_new values for all features
y_prob = clf.predict_proba(X_test)

plt.plot(X_new, y_prob[:, 0], "r-", label="Iris setosa")
plt.plot(X_new, y_prob[:, 1], "g-", label="Iris versicolor")
plt.plot(X_new, y_prob[:, 2], "b-", label="Iris virginica")
plt.legend()
plt.show()
