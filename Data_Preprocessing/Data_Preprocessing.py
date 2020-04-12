import numpy as npy
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('F:\Programming_Languages\Machine_Learning_Repo\Data_Preprocessing\DataSet.csv')
X = dataset.iloc[:,:-1].values
XX = dataset.iloc[:,:-1].values
XXX = dataset.iloc[:,:-1].values
y = dataset.iloc[:,3].values

#fill the missing value by Mean attribute
from sklearn.impute import SimpleImputer
imp_mean = SimpleImputer(missing_values=npy.nan, strategy='mean')
imp_mean=imp_mean.fit(X[:,1:3])
SimpleImputer()
X[:,1:3]= imp_mean.transform(X[:,1:3])

#fill the missing value by Median attribute
from sklearn.impute import SimpleImputer
imp_mean = SimpleImputer(missing_values=npy.nan, strategy='median')
imp_mean=imp_mean.fit(XX[:,1:3])
SimpleImputer()
XX[:,1:3]= imp_mean.transform(XX[:,1:3])

#fill the missing value by Most_Frequent attribute
from sklearn.impute import SimpleImputer
imp_mean = SimpleImputer(missing_values=npy.nan, strategy='most_frequent')
imp_mean=imp_mean.fit(XXX[:,1:3])
SimpleImputer()
XXX[:,1:3]= imp_mean.transform(XXX[:,1:3])

#Converting the string data into Numerical form
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = preprocessing.LabelEncoder()
labelencoder_X = labelencoder_X.fit(X[:,0])
LabelEncoder()
X[:,0] = labelencoder_X.transform(X[:,0])


#Dummy value Concept
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_X = preprocessing.LabelEncoder()
labelencoder_X = labelencoder_X.fit(X[:,0])
LabelEncoder()
X[:,0] = labelencoder_X.transform(X[:,0])
transformer = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), [0])],remainder='passthrough')
X = npy.array(transformer.fit_transform(X), dtype=npy.float)
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(y)


#divide data into training and testing data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train , y_test =train_test_split(X,y,test_size =0.2)

#standard and fit the data for better prediction
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_test = sc_X.fit_transform(X_test)
X_train = sc_X.fit_transform(X_train)





