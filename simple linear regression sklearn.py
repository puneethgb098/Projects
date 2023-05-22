#  data
#x     y
#1     3
#2     4
#3     2 
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error


diabetes = datasets.load_diabetes()

#(['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename', 'data_module'])

diabetes_X = diabetes.data #[:,np.newaxis,2]  if you add this in diabetes_X = diabetes.data[:,np.newaxis,2] then remove the comment of plt
# the above [:,np.newaxis,2] is removed because it only takes 2 inputs which increases the mean squred error 

diabetes_X = np.array(([1],[2],[3],)) #replaces the original feature data from the diabetes dataset with a 
#new array [1, 2, 3]. This line overrides the previous assignment and uses a simple array for demonstration purposes.

diabetes_X_train = diabetes_X
diabetes_X_test = diabetes_X

diabetes_Y_train = np.array([3,2,4])                              
diabetes_Y_test = np.array([3,2,4])                                
                                                                  
model = linear_model.LinearRegression()                         

model.fit(diabetes_X_train,diabetes_Y_train)

diabetes_Y_predicted = model.predict(diabetes_X_test)

print("Mean squared error is:", mean_squared_error(diabetes_Y_predicted, diabetes_Y_test))

print("Weights:", model.coef_)
print("intercept", model.intercept_)

plt.scatter(diabetes_X_test,diabetes_Y_test)
plt.plot(diabetes_X_test,diabetes_Y_predicted)


plt.show()
#Mean squared error is: 3035.060115291269
#Weights: [941.43097333]
#intercept 153.39713623331644