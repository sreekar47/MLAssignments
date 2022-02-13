import numpy as np
import pandas as pd
from argparse import ArgumentParser

parser = ArgumentParser(description='linear regression using Gradient descent')
parser.add_argument('-d', '--data', type=str, metavar='', required=True, help='Input file path')
parser.add_argument('-l', '--eta', type=float, metavar='', required=True, help='Learning rate')
parser.add_argument('-t', '--threshold', type=float, metavar='', required=True, help='Threshold')
args = parser.parse_args()

#read arguments to variable from argparse
file_loc = args.data
eta = args.eta
threshold = args.threshold


dataset = pd.read_csv(file_loc,header=None)
csv_data = np.genfromtxt(file_loc, delimiter=',')
# df = pd.DataFrame(dataset)
X = dataset.iloc[:, :].values
# print(X)

rows = X.shape[0]   # number of rows of the input file
columns =X.shape[1] # number of colums of the input file

# initialising all the required variables
new_weights = np.zeros(columns)    # array to store the new weights in each iteration
old_weights = np.zeros(columns)    # array to store the final weights of an iteration
gradients = np.zeros(columns)    # array to store the gradients
x_values = np.zeros(columns)    # array to store the features
y = 0    # variable to store the actual output value

change_in_error = 1   # giving a random value initially
sse = 0    # initiating sum of squared errors


# function to return the predicted output based on the current weights and features
def predy(weightsarray, arr_x):
    y_pred = 0
    for w_val, x_val in zip(weightsarray, arr_x):
        y_pred += w_val*x_val
    return y_pred
    


# function to extract the features and true y
def features(csv_data, row, columns):
    x_values[0] = 1
    j = 0
    while j < columns-1:
        x_values[j + 1] = csv_data[row, j]
        j += 1
    y = csv_data[i, columns - 1]
    # print(type(x_values))
    # print(type(y))
    return x_values,y


# function to make copy of weights for next iteration
def storeweights(new_weights, old_weights, columns):
    j = 0
    while j < columns:
        old_weights[j] = new_weights[j]
        j += 1
    return old_weights



#main loop
iterator = 0 
while change_in_error > threshold:    
    gradients = np.zeros(columns)
    oldsse = sse
    sse = 0    

    i = 0
    while i < rows:
        old_weights = storeweights(new_weights,old_weights, columns)    
        x_values,y = features(csv_data, i,columns)    
        y_pred = predy(old_weights, x_values)    

        j = 0
        while j < columns:
            gradients[j] += x_values[j]*(y-y_pred) 
            j += 1

        sse = sse+(y-y_pred)**2  
        i += 1

    j = 0
    while j < columns:
        new_weights[j] += gradients[j] * eta    
        j += 1
    print(str(iterator)+','+str(round(old_weights[0],9))+','+str(old_weights[1])+','+str(old_weights[2])+','+str(sse))
    
    change_in_error = abs(oldsse-sse) 
    iterator += 1
