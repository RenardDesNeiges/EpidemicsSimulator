# Artificial Neural Networks : Epidemic Control Project

Building a reinforcement learning environment for training epidemic mitigation policies. (Presumably using Q-learning.)

## Model design:
Model the epidemic dynamics on a graph representing cities and social-groups with the following per-node dynamics (for city $i \in [m]$ for $m$ cities):

$\begin{cases}
\dot{s}_i = \gamma_i r_i - \alpha_i s_i(i_i + \sum_{j\neq i} \tau_j i_j )\\
\dot{i} = \alpha_i s_i(i_i + \sum_{j\neq i} \tau_j i_j ) - (\beta_i + \zeta_i) i_i\\
\dot{r} = \beta_i i_i - \gamma_i r_i\\
\dot{d} = \zeta_i i_i \\
\end{cases}$

where the $_i$ subscript denotes the $i$-th city, the variables are the following:
- $s_i$ the proportion of susceptible population
- $i_i$ the proportion of infected population
- $r_i$ the proportion of recovered population
- $d_i$ the proportion of dead population

the parameters are the following:
- $\alpha_i$ is the transmission rate
- $\beta_i$ is the recovery rate
- $\zeta_i$ is the death rate
- $\gamma_i$ is the immunity-loss rate

Since $s_i +i_i + r_i +d_i = 1$ the model is a $3\cdot m$-th order model.


## Todos :
- Write a dynamics class that implements the simulation
- Write a visualization library
<!-- - Write a yaml parser that creates instances of the dynamics class from files  -->