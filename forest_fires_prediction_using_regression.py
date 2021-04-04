# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 15:48:51 2021

@author: CRACA
"""

#Data_Preprocessing --> ForestFire

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("forest.csv")
o_data = data.iloc[:,:-1].values
target = data.iloc[:,-1].values

o_data = pd.DataFrame(o_data)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le =LabelEncoder()
o_data[2] = le.fit_transform(o_data[2])
o_data[3] = le.fit_transform(o_data[2]) 

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(o_data, target , test_size = 0.2 , random_state = 1)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

#******************************************************************************
print("\n\n")
print("FOREST FIRE PREDICTION DEPENDING ON 4 VARIABLE MODELS ")



##*****************************************************************************
print("\n\n")
print("PREDICTIONS ON THE BASIS OF LINEAR REGRESSION MODEL")

#Linear_Regression_Model-->>ForestFire

from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()
linear_regressor.fit(x_train,y_train)

#prediction_of_linear_regression_model

#on_testing_data
linear_regression_prediction_test = linear_regressor.predict(x_test)


#Errors

from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
 
# Mean_squared_error 
print("\nMEAN_SQUARE_ERROR")
print(mean_squared_error(linear_regression_prediction_test,y_test))

#Mean_absolute_error
print("\nMEAN_ABSOLUTE_ERROR")
print(mean_absolute_error(linear_regression_prediction_test,y_test))

#r2_score
print("\nR2_SCORE")
print(r2_score(linear_regression_prediction_test,y_test))

print("\n------------------------------------------------")

#******************************************************************************

print("\n")
print("PREDICTIONS ON THE BASIS OF POLYNOMIAL REGRESSION MODEL(best model)")
#polynomial_regression-->ForetFire

#for_degree_2
from sklearn.preprocessing import PolynomialFeatures
polynomial_0 = PolynomialFeatures()
polynomial_x_train_0 = polynomial_0.fit_transform(x_train)
polynomial_x_test_0 = polynomial_0.fit_transform(x_test)

p_linear_regresssor_0 = LinearRegression()
p_linear_regresssor_0.fit(polynomial_x_train_0,y_train)

predict_p_linear_regressor_0 = p_linear_regresssor_0.predict(polynomial_x_test_0)


#for_degree_3
polynomial_1 = PolynomialFeatures(degree = 3)
polynomial_x_train_1 = polynomial_1.fit_transform(x_train)
polynomial_x_test_1 = polynomial_1.fit_transform(x_test)

p_linear_regresssor_1 = LinearRegression()
p_linear_regresssor_1.fit(polynomial_x_train_1,y_train)

predict_p_linear_regressor_1 = p_linear_regresssor_1.predict(polynomial_x_test_1)


#Errors
print("\nFOR_DEGREE_2(best prediction)")
#for_polynomial_regression_degree_2

# Mean_squared_error 
print("\nMEAN_SQUARE_ERROR")
print(mean_squared_error(predict_p_linear_regressor_0,y_test))

#Mean_absolute_error
print("\nMEAN_ABSOLUTE_ERROR")
print(mean_absolute_error(predict_p_linear_regressor_0,y_test))

#r2_score
print("\nR2_SCORE")
print(r2_score(predict_p_linear_regressor_0,y_test))

print("\n***************************")

#for_polynomial_regression_degree3
print("\nFOR_DEGREE_3")
# Mean_squared_error
print("\nMEAN_SQUARE_ERROR")
print(mean_squared_error(predict_p_linear_regressor_1,y_test))

#Mean_absolute_error
print("\nMEAN_ABSOLUTE_ERROR")
print(mean_absolute_error(predict_p_linear_regressor_1,y_test))

#r2_score
print("\nR2_SCORE")
print(r2_score(predict_p_linear_regressor_1,y_test))

print("\n-----------------------------------------------------------")

#******************************************************************************
print("\n")
print("PREDICTIONS ON THE BASIS OF DECISION TREE REGRESSION MODEL")
#Decision_Tree_Regression-->forestfire

from sklearn.tree import DecisionTreeRegressor
decision_tree_regressor = DecisionTreeRegressor(random_state=0)
decision_tree_regressor.fit(x_train,y_train)

#prediction_of_decision_tree_regressor

#on_testing_data
decision_tree_prediction_test = decision_tree_regressor.predict(x_test)

#Errors
 
# Mean_squared_error
print("\nMEAN_SQUARE_ERROR") 
print(mean_squared_error(decision_tree_prediction_test,y_test))

#Mean_absolute_error
print("\nMEAN_ABSOLUTE_ERROR")
print(mean_absolute_error(decision_tree_prediction_test,y_test))

#r2_score
print("\nR2_SCORE")
print(r2_score(decision_tree_prediction_test,y_test))

print("\n-----------------------------------------------------------")

#*****************************************************************************
print("\n")
print("PREDICTIONS ON THE BASIS OF RANDOM FOREST REGRESSION MODEL")

#Random_Forest_Regression-->forestfire
from sklearn.ensemble import RandomForestRegressor
random_forest_regressor = RandomForestRegressor(n_estimators=10,random_state=0)
random_forest_regressor.fit(x_train,y_train)

#prediction_of_random_forest_regressor

#on_testing_data
random_forest_prediction_test = random_forest_regressor.predict(x_test)

#Errors
 
# Mean_squared_error 
print("\nMEAN_SQUARE_ERROR") 
print(mean_squared_error(random_forest_prediction_test,y_test))

#Mean_absolute_error
print("\nMEAN_ABSOLUTE_ERROR")
print(mean_absolute_error(random_forest_prediction_test,y_test))

#r2_score
print("\nR2_SCORE")
print(r2_score(random_forest_prediction_test,y_test))
print("\n-----------------------------------------------------------")

print("\n final outcome --> The best predictions are made by POLYNOMIAL REGRESSION WITH DEGREE 2")


print("\n-----------------------*************------------------")

#*****************************************************************************
#END OF CODE

