# CTOProject
Simple (optimized) neural network compiler for the course of CTO

# Test enviroment
The following folder contains a training application for the generation of small (dense) MLP neural network.
The execution can follow different steps

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




