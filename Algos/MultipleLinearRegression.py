import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# reading csv
df = pd.read_csv("banglore.csv")
shape = df.shape
df.head()
x=df.drop("price", axis=1) 
y=df["price"]


# train and test split
x_train,x_test,y_train,y_test= train_test_split(x,y, test_size = 0.2, random_state =0)
#shapes of splitted data
print("x_train:",x_train.shape)
print("X test:",x_test.shape) 
print("Y_train:",y_train.shape)
print("_test:",y_test.shape)

# fitting data through Linear regression and yielding accuracy
linreg = LinearRegression() 
linreg.fit(x_train,y_train)
y_predict=linreg.predict(x_test) 
Accuracy = linreg.score(x_test,y_predict)*100
Accuracy
