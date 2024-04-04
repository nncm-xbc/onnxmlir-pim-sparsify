# CTOProject
Simple (optimized) neural network compiler for the course of CTO.
## Sparsifier
Different algorithms are able to produce, in an efficent way, sparse neural networks. Nonetheless, they often deal with in-training sparsification,
helping the training to converge in the direction of a sparse result. The rationale behind this work is to offer a sparsification algorithm 
able to deal with the problem of a-posteriori sparsification: how can we make a neural network sparser, in order to be less computationally heavy on 
simpler architectures where massive parallelism is not possible, and most of all without training AGAIN a new network? The method uses the <b>Manifold assumption</b> about parameters, namely the idea that
every set of parameters induces locally (on the surface of the possible parameters) a distance that quantify the semantic difference represented by the networks.
A greedy algorithm is therefore produced, in order to sparsify parameter by parameter the network and after each "epoch" use the differentiability 
of the aforementioned distance in order to stabilize the noise introduced by the removal of a parameter.

## Compiler
After sparsification, a fundamental key was the production of a compiler able to exploit the gained sparsity.
The presenteed compiler is thought to do mainly two things
1. Minimize the amount of data transfer during the flow from the input layer to the ooutput layer. The idea is to make the number of data 
   movements $\mathcal O(R)$ , where $R$ is the number of registers in the machine (which is way smaller than the cardinality of the address space)
2. Optimize the allocation in memory in order to preserve as far as possible data locality, constructing a suitable metric to evaluate 
   how much such criteria is respected. This optimization happens through a <b>Simulated Annealing Procedure</b>

# Test enviroment
The ```Test``` folder contains a training application for the generation of small (dense) MLP neural network.
The enviroment provides a full use case for the sparsification and compilation, from the training to the execution
of the Manifold Based Sparsifying Algorithm , arriving finally to the production of a fully working assembly code for ARM architecture, using an highly optimized ad hoc compiler.
Technical details for the mathematics and the compiler design can be found in the documentation.

# 1) Generation of a new dataset (Not mandatory)
In the test folder you can find the ```network_topology.csv``` file, which basically describes the structure of the network.
The only "unusual" thing is that the first layer, in the csv, is represented by the square root of the actual number; this design choice
was made to enforce "easily" that the number of neuron of the first layer has to be a perfect square (since MNIST images are in square format).
If you would like to modify the resolution of input images you can run, in the project main folder the following command
<br>
<center>
<code>python3 -m Test.dataeng network_topology.csv</code>
</center>
<br>
<br>

Which will generate for you a dataset that can be used to train a network with the given topology. Feel free to make experiments!

<b> Hint : </b> do you want to try lighter or heavier networks? Just modify the topology csv and run this command. The next command are resilient to modifications of the settings,
as long they are performed properly.
# 2) Training of the network
If you got here you have now a complete dataset to train your model. Training is simple, just run --- again from the folder of the project --- the following
<br>
<center>
<code>python3 -m Test.train params_folder_name network_topology.csv</code>
</center>
<br>
<br>
This will build a folder named "params_folder_name" inside the test folder, containing the result of the training, in terms of weights and biases.

# 3) Sparsify your network
To lunch the procedure, run the following command
<br>
<center>
<code>python3 -m Sparsifier.sparsifier ../Test/parametri ../Test/X_test_small.csv ../Test/Y_test_small.csv </code>
</center>
<br>
<br>
Note that the presence of the dataset is offered only as a proof of concept of the correctness. It's not actually used into the pruning and adjust procedure,
but only for the user to verify that everything was set up fine and proceeding in the correct way. (the provided datasets may even be very small portions of the original dataset,
it is just to check visually during the pruning that everything is ok)




