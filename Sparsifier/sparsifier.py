from PIL import Image
from MLP.mlp import *                  # Mini-Package for Multi Layer Perceptron
import jax
import sys 
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


# load the neural network

# define the local distance on the manifold of parameters
def d(params_1, mask_1, params_2, mask_2 ):
        omega_sample = np.random.choice(255, size = (10000,14*14))
        return jnp.sum( (batched_predict(params_1, mask_1, omega_sample) - batched_predict(params_2, mask_2, omega_sample) )**2 )
d = jax.jit(d)
d_grad = jax.grad(d)
d_grad = jax.jit(d_grad)

# trovo il peso piú piccolo
import numpy as np

# construct a copy of the parameters
def clone_params(params):
    return [  [ np.array(p[0]).copy(),np.array(p[1]).copy()] for p in params]


########################################################################################################################################
# Adjust Function : minimize the distance on the plane locally isomorphic to the manifold of paremeters.
# given 
#       - a set of parameters not adjusted
#       - a mask which represent the sparsity pattern of the parameters
#       - a set of parameters that we would like to mimic semantically
#       - given the mask of the parameters that we would like to mimic (only 1)
# 
# Returns
#       - a set of parameters, with the same sparsity pattern of the 
#         input parameters but optimized to have minimal local distance
#         in the manifold of parameters with respect to the cmp_params (params that we want to mimic)
#

def get_full_mask(params):
    return [ np.array(p[0])*0. + 1 for p in params]
    

def adjust(params, mask, cmp_params, full_mask):
    new_params = clone_params(params)
    alfa       = 1e-11
    while alfa > 1e-14 :
        gradiente = d_grad(params,mask,cmp_params,full_mask)
        new_new_params = clone_params(new_params)
        for n,g in zip(new_new_params,gradiente):
            n[0] = n[0] - alfa * g[0]
            n[1] = n[1] - alfa * g[1]
            
        if(d(new_new_params,mask,cmp_params,full_mask) < d(new_params,mask,cmp_params,full_mask)):
            new_params = new_new_params
            alfa *= 1.001
        else:
            alfa *= 0.5
    return new_params
########################################################################################################################################


########################################################################################################################################
# Prune Function : function for increasing the sparsity pattern of parameters
# given 
#       - a set of parameters
#
# The function compute every possible variation of the initial set of parameters obtainable
# by, for each variation, put a different parameter to 0.
# The optimal candidate, is the variation that minimizes the local distance with the
# original set of parameters
#
# returns
#       - a set of parameters, with the same sparsity pattern of the 
#         input parameters but optimized to have minimal local distance
#         in the manifold of parameters with respect to the cmp_params (params that we want to mimic)
#
def prune(params, mask, cmp_params, full_mask, doAdjust = True):
    minimo = 1e16
    minimo_idx = 0
    minimo_i   = 0
    minimo_j   = 0
    
    for idx,p in enumerate(params):
        #print(idx)
        for i,row in enumerate(p[0]):
            #print("\t%d" % i , minimo)
            for j,el in enumerate(row):
                if params[idx][0][i,j] != 0.:
                    new_params = clone_params(params)
                    new_params[idx][0][i,j] = 0.
                    distanza =  d(cmp_params,full_mask,new_params,mask)
                    if(distanza < minimo):
                        minimo = distanza
                        minimo_idx = idx
                        minimo_i   = i
                        minimo_j   = j
                    if minimo == 0:         # if the minimum becomes zero then the current weight doees not affect distance in P space
                        # non ha bisogno di aggiustamento
                        mask[minimo_idx][minimo_i,minimo_j]          = 0. # setto la maschera  a 0
                        return new_params
    new_params = clone_params(params)
    new_params[minimo_idx][0][minimo_i,minimo_j] = 0.
    
    mask[minimo_idx][minimo_i,minimo_j]          = 0. # setto la maschera  a 0
    # ha bisogno di aggiustamento
    if doAdjust:
        new_params = adjust(new_params, mask, cmp_params,full_mask)
    
    return new_params
########################################################################################################################################

def load_data():
    data_train       = np.genfromtxt(__location__ + '/mnist_train.csv', delimiter = ',', max_rows = 10000)
    x_train, y_train = data_train[:,:-1], data_train[:,0]
    
    data_test        = np.genfromtxt(__location__ + '/mnist_test.csv', delimiter = ',', max_rows = 1000)
    x_test, y_test   = data_test[:,:-1], data_test[:,0]
    
    x_train          = x_train.reshape(-1,28,28)
    x_test           = x_test.reshape(-1,28,28)
    
    return (x_train,y_train),(x_test,y_test)

    
    
def main():
    args          = sys.argv
    input_folder  = __location__ + "/" + args[1]	
    x_test_file   = __location__ + "/" + args[2]	
   
    print("Load the validation set for the user to evaluate to goodness")
    data          = np.genfromtxt(x_test_file, delimiter = ',', max_rows = 1000)
    x_test,y_test = data[:,1:],data[:,0]
    x_test = x_test.reshape(-1,28,28)
    x_test = np.array([ np.array(Image.fromarray(x).resize((14,14))) for x in x_test]).reshape(-1,14*14) # da togliere sta cosa, fare il resize nel file
    
    onehot = lambda y : np.concatenate([
        (y == cifra)[:,None] * 1.
    for cifra in np.arange(10)
    ], axis = 1)
    
    y_test        = onehot(y_test)

    print("Load the parameters from the folder")
    params = load_network_params(input_folder)
    full_mask = get_full_mask(params)
    mask = get_full_mask(params)

    print("Accuracy in validation:", accuracy(params,mask,  x_test, y_test))
    print("Construct the mask for the sparsification")
    


    params_perturbed = [ (p[0] + np.random.normal(size = p[0].shape) * .00001 ,p[1]) for p in params]
    
    print("Compute the local distance between a random perturbation of the input network and the input network itself")
    print(">>>", d(params,full_mask,params_perturbed,full_mask))
    
    print("Starting sparsification loop")
    print("At every iteration the networks gets")
    print("\t 1. Pruned  --- to remove the least influential parameter")
    print("\t 2. Adjusted --- to mitigate the effect of the removal of the parameter using the other ones")
    new_params = clone_params(params)
    for i in range(500):
        NZ = np.sum([ (p[0] != 0).sum() for p in new_params]) # non zero entries
        print( "validation accuracy = %.3f" % accuracy(new_params,mask,  x_test, y_test) , " | non zero elements = %d" % NZ)
        new_params = prune(new_params, mask, params, full_mask, True)


if __name__ == "__main__":
    main()
    