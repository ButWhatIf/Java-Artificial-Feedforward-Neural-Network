# Java-Artificial-Feedforward-Neural-Network
All the classes generate a dense, feedforward neural network in Java. There is currently support for two loss functions, four optimizers,
and eight activation functions. All the optimizers are gradient descent based, though particle swarm is a feature I would like to implement in the future. 
Currently I do not have plans to write the Nesterov variants of each optimizer. See the "example" program to see how to implement this program on an example 
training set, where I train this NN using "training_example.txt", and write the outputs t

Future commits:
- Additional loss functions (in particular, Binary Cross Entropy and Hubert)
- RMSProp gradient descent algorithm
- Particle swarm optimization algorithm.

Current issues: ADAM is currently performing worse than vanilla SGD and momentum gradient descents, but more testing is required to determine if this is due to sample "allergic" data or simply faulty implementation. 
