import networkx as nx
import numpy as np
import random as rd
import matplotlib.pyplot as plt


class LatticeDynamics():
    """ A dynamical model of an epidemic on a L2 lattice
        
    """
    def __init__(self, 
                 size = 20,
                 p_si = 1e-3,
                 p_ir = 1e-3,
                 p_rs = 1e-3,
                 ):
        self.lattice = nx.grid_graph(dim=(size,size)) # we model the world as a fully connected 2d lattice
        self.p_si = p_si
        self.p_ir = p_ir
        self.p_rs = p_rs
        self.dt = 1
        self.reset()
        
    def reset(self):
        nx.set_node_attributes(self.lattice, 
                               's', 
                               'state', 
                               )
    def start_epidemic(self, 
                       initial_infected):
        initial_nodes = rd.sample(self.lattice.nodes,initial_infected)
        for n in initial_nodes:
            self.lattice.nodes[n]['state'] = 'i'
            
    def count_sir(self):
        acc_s = 0
        acc_i = 0
        acc_r = 0
        for n in self.lattice.nodes:
            if self.lattice.nodes[n]['state'] == 's':
                acc_s += 1
            elif self.lattice.nodes[n]['state'] == 'i':
                acc_i += 1
            elif self.lattice.nodes[n]['state'] == 'r':
                acc_r += 1
                
        return acc_s,acc_i,acc_r
        
    def _infected_neighbors(self, node):
        acc = 0
        for n in self.lattice.neighbors(node):
            if self.lattice.nodes[n]['state'] == 'i':
                acc += 1
                
        return acc
    
    def plot_state(self):
        pos_map = {n:np.array(n) for n in self.lattice.nodes}

        color_map = []

        for n in self.lattice.nodes:
            if self.lattice.nodes[n]['state'] == 's':
                color_map.append('green')
            if self.lattice.nodes[n]['state'] == 'i':
                color_map.append('red')
            if self.lattice.nodes[n]['state'] == 'r':
                color_map.append('blue')
        
        nx.draw(self.lattice,pos=pos_map,node_color=color_map, node_size=25)
        plt.show()
    
    def step(self):
        for n in self.lattice.nodes:
            if self.lattice.nodes[n]['state'] == 's':
                if rd.uniform(0,1) < self.p_si*self._infected_neighbors(n):
                    self.lattice.nodes[n]['state'] = 'i'
            elif self.lattice.nodes[n]['state'] == 'i':
                if rd.uniform(0,1) < self.p_ir:
                    self.lattice.nodes[n]['state'] = 'r'
            elif self.lattice.nodes[n]['state'] == 'r':
                if rd.uniform(0,1) < self.p_rs:
                    self.lattice.nodes[n]['state'] = 's'
                    
                    
    def run_episode(self, initial_cases, steps):
        self.reset()
        self.start_epidemic(initial_cases)
        s = []
        i = []
        r = []
        for _ in range(steps):
            _s,_i,_r = self.count_sir()
            s.append(_s)
            i.append(_i)
            r.append(_r)
            self.step()
        s = np.array(s)
        i = np.array(i)
        r = np.array(r)
            
        return s,i,r