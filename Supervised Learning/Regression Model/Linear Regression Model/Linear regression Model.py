import matplotlib.pyplot as plt
import numpy as np 
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

diabetes = datasets.load_diabetes()

# data_x = diabetes.data[:, np.newaxis, 2] #for only one independent var.(feature) we can plot graph but its predection accuracy is low
data_x = diabetes.data
data_x_train = data_x[:-30]
data_x_test = data_x[-30:]

data_y_train = diabetes.target[:-30]
data_y_test = diabetes.target[-30:]

model = linear_model.LinearRegression()
model.fit(data_x_train, data_y_train)

data_y_predicted = model.predict(data_x_test)

# plt.scatter(data_x_test, data_y_test)
# plt.plot(data_x_test, data_y_predicted)
# plt.show()

print("The Mean of square error is: ", mean_squared_error(data_y_test, data_y_predicted))
print("Coefficient are: ",model.coef_)
print("Intercept: ",model.intercept_)