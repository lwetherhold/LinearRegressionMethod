
import numpy as np
# X          - single array/vector
# y          - single array/vector
# theta      - single array/vector
# alpha      - scalar
# iterations - scarlar

def gradientDescent(X, y, theta, alpha, numIterations):
    '''
    # This function returns a tuple (theta, Cost array)
    '''
    m = len(y)
    arrCost =[];
    # NOTE: transposedX is unused (used X.T directly instead)
    #transposedX = np.transpose(X) # transpose X into a vector  -> XColCount X m matrix
    for iteration in range(0, numIterations):
        ################PLACEHOLDER3 #start##########################
        #: write your codes to update theta, i.e., the parameters to estimate. 
	# Replace the following variables if needed 

        # inside loop, compute prediction error
        
        # predicted y_hat from current theta and X
        # each prediction is the row of X dotted with theta
        # y_hat = X * theta
        y_hat = X.dot(theta)

        # subtract true y
        error = y_hat - y # compute error_old

        # compute the gradient vector of the cost function with respect to theta
        # G = (1/m) * X^T * error
        G = X.T.dot(error) / m # compute gradient from error_old

        # update parameters w/ gradient descent step
        # theta = theta - alpha * gradient
        theta = np.subtract(theta, alpha*G) # update theta
        ################PLACEHOLDER3 #end##########################

        ################PLACEHOLDER4 #start##########################
        # calculate the current cost with the new theta;

        # recompute y_hat and error_new with updated theta
        y_hat_new = X.dot(theta)
        error_new = y_hat_new - y

        # compute cost from error_new after updating theta
        # cost = (1/(2*m)) * sum(error^2)
        atmp = np.sum(error_new**2) / (2*m)

        # print cost
        #print(atmp)

        # append numeric cost to arrCost
        arrCost.append(atmp)
        ################PLACEHOLDER4 #end##########################

    return theta, arrCost