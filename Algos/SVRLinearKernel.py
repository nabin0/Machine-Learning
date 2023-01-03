import pandas as pd
"""### Load Data"""
path = r"https://drive.google.com/uc?export=download&id=1xxDtrZKfuWQfl-6KA9XEd_eatitNPnkB" 
df = pd.read_csv(path)
df.head()
"""## Split Data"""
X = df.drop('price', axis=1)
y = df['price']
print('Shape of X = ', X.shape)
print('Shape of y = ', y.shape)
 

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=51)
print('Shape of X_train = ', X_train.shape)
print('Shape of y_train = ', y_train.shape)
print('Shape of X_test = ', X_test.shape)print('Shape of y_test = ', y_test.shape)
 
"""## Feature Scaling"""
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)
"""## Support Vector Regression - ML Model Training"""
from sklearn.svm import SVR
svr_linear = SVR(kernel='linear')
svr_linear.fit(X_train, y_train)
svr_linear.score(X_test, y_test)
 

"""## Predict the value of Home and Test"""
X_test[0]
svr_linear.predict([X_test[0]])
y_pred = svr_linear.predict(X_test)
y_pred
 
from sklearn.metrics import mean_squared_error
import numpy as np
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print('MSE = ', mse)
print('RMSE = ', rmse)
