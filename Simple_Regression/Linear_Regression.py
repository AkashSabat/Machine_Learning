import numpy as npy
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('F:\Programming_Languages\Machine_Learning_Repo\Simple_Regression\Salary_DataSet.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values

#divide data into training and testing data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train , y_test =train_test_split(X,y,test_size =0.3,random_state=0)

#standard and fit the data for better prediction
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_test = sc_X.fit_transform(X_test)
X_train = sc_X.fit_transform(X_train)

#regression
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train,y_train)

y_predict = reg.predict(X_test)

# Red points are actual Data 
# Blue line is a predicted data

#For Training Data
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,reg.predict(X_train), color='blue')
plt.title("Linear Regression Salary vs Experience")
plt.xlabel("Years of Experience")
plt.ylabel("Salaries of Employees")
plt.show()

#For Testing Data
plt.scatter(X_test,y_test,color='red')
plt.plot(X_train,reg.predict(X_train), color='blue')
plt.title("Linear Regression Salary vs Experience")
plt.xlabel("Years of Experience")
plt.ylabel("Salaries of Employees")
plt.show()