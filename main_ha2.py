from download_data import download_data
import numpy as np
import matplotlib.pyplot as plt
from GD import gradientDescent
# NOTE: removed rescaleMatrix() import since we're using our own normalization
#from dataNormalization import rescaleMatrix
 

#NOTICE: Fill in the codes between "%PLACEHOLDER#start" and "PLACEHOLDER#end"

# There are two PLACEHODERS IN THIS SCRIPT

# parameters

################PLACEHOLDER1 #start##########################
# test multiple learning rates and report their convergence curves. 
ALPHA = 0.1
MAX_ITER = 500
################PLACEHOLDER1 #end##########################

#% step-1: load data and divide it into two subsets, used for training and testing
sat = download_data('sat.csv', [1, 2, 4]).values # three columns: MATH SAT, VERB SAT, UNI. GPA  # convert frame to matrix

################PLACEHOLDER2 #start##########################
# Normalize data
#sat = rescaleMatrix(sat) # please replace this code with your own codes

# use my own normalization formula on sat (all 3 columns)
# instead of rescaleMatrix()

# SIMPLE OPTION: min-max normalization to [0,1]

col_min = sat.min(axis=0)
col_max = sat.max(axis=0)
sat = (sat - col_min) / (col_max - col_min + 1e-8)

# ALTERNATIVE OPTION: mean normalization to [-1,1]
#col_mean = sat.mean(axis=0)
#col_min = sat.min(axis=0)
#col_max = sat.max(axis=0)
#sat = (sat - col_mean) / (col_max - col_min + 1e-8)
# subtracts each column's mean, scales by that column's range
# 1e-8 avoids divide-by-zero if a column is constant
################PLACEHOLDER2 #end##########################

 
# training data;
satTrain = sat[0:60, :]
# testing data; 
satTest = sat[60:len(sat),:]

#% step-2: train a linear regression model using the Gradient Descent (GD) method
# ** theta and xValues have 3 columns since have 2 features: y = (theta * x^0) + (theta * x^1) + (theta * x^2)
theta = np.zeros(3) 

xValues = np.ones((60, 3)) 
xValues[:, 1:3] = satTrain[:, 0:2]
yValues = satTrain[:, 2]
# call the GD algorithm, placeholders in the function gradientDescent()
[theta, arrCost] = gradientDescent(xValues, yValues, theta, ALPHA, MAX_ITER)

 
#visualize the convergence curve
plt.plot(range(0,len(arrCost)),arrCost);
plt.xlabel('iteration')
plt.ylabel('cost')
plt.title('alpha = {}  theta = {}'.format(ALPHA, theta))
plt.show()

#% step-3: testing
testXValues = np.ones((len(satTest), 3)) 
testXValues[:, 1:3] = satTest[:, 0:2]
tVal =  testXValues.dot(theta)
 

#% step-4: evaluation
# calculate average error and standard deviation
tError = np.sqrt([x**2 for x in np.subtract(tVal, satTest[:, 2])])
print('results: {} ({})'.format(np.mean(tError), np.std(tError)))