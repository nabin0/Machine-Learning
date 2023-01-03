from sklearn import datasets
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
df = datasets.load_wine()
df.keys() 
df['data']
 

# Split data for training and testing
x_train, x_test, y_train, y_test = train_test_split(df.data, df.target, test_size=0.2, random_state=0)
model  =  LogisticRegression (max_iter=10008)
model.fit(x_train, y_train)
y_predict = model.predict(x_test)
accuracy  =  accuracy_score (y_test, y_predict)*100
