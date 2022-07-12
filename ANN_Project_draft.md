# Artificial Neural Networks : CTRNN Project

An introduction to continuous time neural networks, their pros-and-cons and an application with reinforcement learning.

The project would be split into three parts: 

1. Given data from a dynamical system, use a continous-time model to derive a model of the system's dynamics (which are unknown) [I was thinking a rocket landing or docking situation, alternatively one could do covid propagation on a graph, but if there is a more Neurosciency topic you would like why not]
2. Exploit that model together with continuous RL to train a policy to solve a control problem using the model that we derived using the data [train a policy that lands the rocket]
3. Test the results on the actual dynamical system using [evaluate the results]

So the student would be given a black box dynamical system together with a dataset of trajectories of the rocket, and would have to implement a continuous time model to model them (I was thinking of a neural-ode model). 

Then using that model the student would then implement either PPO or DDPG and fit a model on that problem (I think here one would have to use a MLP as training continuous time models locally is really hard).