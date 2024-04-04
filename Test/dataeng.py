# data_eng.py
# Produces a smaller dataset starting from the mnist, which is suitable for
# the proposed experiment

import numpy as np
import os
from PIL import Image
import sys

__location__  = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
args          = sys.argv
topology_csv  = __location__ + "/" + args[1]	

def load_data():
    data_train       = np.genfromtxt(__location__ + '/mnist_train.csv', delimiter = ',')
    x_train, y_train = data_train[:,1:], data_train[:,0]
    
    data_test        = np.genfromtxt(__location__ + '/mnist_test.csv', delimiter = ',')
    x_test, y_test   = data_test[:,1:], data_test[:,0]
    
    x_train          = x_train.reshape(-1,28,28)
    x_test           = x_test.reshape(-1,28,28)
    
    return (x_train,y_train),(x_test,y_test)
	
	
(x_train,y_train),(x_test,y_test) = load_data()
    
# data loading
onehot = lambda y : np.concatenate([
	(y == cifra)[:,None] * 1.
for cifra in np.arange(10)
], axis = 1)

# read the topology
topology = np.genfromtxt(topology_csv, delimiter = ',')
# width of the first layer
w        = int(topology[0])

# re-engineer data 
x_train_small  = np.array([ np.array(Image.fromarray(x).resize((w,w))) for x in x_train]).reshape(-1,14*14)
x_test_small   = np.array([ np.array(Image.fromarray(x).resize((w,w))) for x in x_test]).reshape(-1,14*14)
y_train_small  = onehot(y_train)
y_test_small   = onehot(y_test)

# save data
np.savetxt(__location__ + "/X_train_small.csv", x_train_small, delimiter=",")
np.savetxt(__location__ + "/Y_train_small.csv", y_train_small, delimiter=",")
np.savetxt(__location__ + "/X_test_small.csv", x_test_small, delimiter=",")
np.savetxt(__location__ + "/Y_test_small.csv", y_test_small, delimiter=",")

# log
print( "Operation complete" )
print( "X_train.shape = %s" % str(x_train_small.shape))
print( "Y_train.shape = %s" % str(y_train_small.shape))
print( "X_test.shape = %s"  % str(x_test_small.shape))
print( "Y_test.shape = %s"  % str(y_test_small.shape)) 