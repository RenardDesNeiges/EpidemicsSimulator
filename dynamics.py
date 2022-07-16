import yaml
import networkx as nx
import numpy as np
import random as rd
import matplotlib.pyplot as plt

class ModelDynamics():
    
    """ Initializes the ModelDynamics class, 
        creates a graph and sets epidemic dynamics 
        parameters from a source yaml file
        
        Parameters : 
            yaml_source [string] : path to yaml initialization file

        Returns : 
            None
    """
    def __init__(self, yaml_source):
        # loading the parameters from the yaml file
        doc = open(yaml_source, 'r')    
        _params = yaml.safe_load(doc)
        
        # simulation parameters
        self.alpha = _params['alpha']
        self.beta = _params['beta']
        self.gamma = _params['gamma']
        
        # cities and roads lists
        self.cities = list(_params['cities'].keys())
        self.roads = _params['roads']
        
        # generating a graph from the roads and cities
        self.map = nx.Graph()
        self.map.add_nodes_from(self.cities)
        self.map.add_edges_from(self.roads)
        
        self.pos_map = {}
        for c in self.cities:
            self.map.nodes[c]['pop'] = _params['cities'][c]['pop'] 
            self.pos_map[c] = [_params['cities'][c]['lat'],_params['cities'][c]['lon']]
    
        # initializing the variables        
        nx.set_node_attributes(self.map, 1., "s")   
        nx.set_node_attributes(self.map, 0., "i")   
        nx.set_node_attributes(self.map, 0., "r")   
        nx.set_node_attributes(self.map, 0., "d")   
        
        s = np.sum([self.map.nodes[n]['pop'] for n in self.map.nodes()])

        for e in self.roads:
            tau = 10*(self.map.nodes[e[0]]['pop'] * self.map.nodes[e[1]]['pop'])/s**2
            self.map.edges[e]['tau'] = tau 
    
    """ Draws the map on which the epidemic is simulated
        
        Parameters : 
            None

        Returns : 
            None
    """
    def draw_map(self,):
        nx.draw(self.map,
                with_labels=True,
                pos=self.pos_map,
                node_size=[self.map.nodes[n]['pop']/1000 for n in self.map.nodes()], 
                width=[self.map.edges[e]['tau']*10 for e in self.map.edges()]
                )

    """ Returns the state of the epidemic propagation 
        
        Parameters : 
            None

        Returns : 
            total [dict] : a dict containing the total suceptible, infected, recovered and dead population
            cities [dict] : a dict containing the suceptible, infected, recovered and dead population per city
    """
    def epidemic_parameters(self,):
        cities = {}
        
        for c in self.cities:
            suceptible = self.map.nodes[c]['s'] * self.map.nodes[c]['pop']
            suceptible_total += suceptible
            infected = self.map.nodes[c]['i'] * self.map.nodes[c]['pop']
            infected_total += infected
            recovered = self.map.nodes[c]['r'] * self.map.nodes[c]['pop']
            recovered_total += recovered
            dead = self.map.nodes[c]['d'] * self.map.nodes[c]['pop']
            dead_total += dead
            
            city = {
                'suceptible' : suceptible,
                'infected' : infected,
                'recovered' : recovered,
                'dead' : dead
            }
            cities[c] = city
        
        total = {
            'suceptible' : suceptible_total,
            'infected' : infected_total,
            'recovered' : recovered_total,
            'dead' : dead_total
        }
        
        return total, cities
            
    
    """     Starts the epidemic (infects a given proportion 
            of the population in one or more randomly chosen cities)
        
        Parameters : 
            seed [int] : the random seed 
            sources [int] : the number of cities we want the epidemic to start from
            prop [float] : the propotion of the population we initialy infect in a given city

        Returns : 
            None
    """ 
    def start_epidemic(self,seed=10, sources=1, prop=0.1):
        rd.seed(seed)
        
        start_cities = rd.choices(self.cities, k = sources)
        for c in start_cities:
            self.map.nodes[c]['i'] += prop
            self.map.nodes[c]['s'] -= prop
    
    """ Step forward in the epdidemic
    TODO : implement that
    TODO : figure out how long is a step for RP purposes
        
        Parameters : 
            None

        Returns : 
            None
    """
    def step(self,):
        pass
    
dyn = ModelDynamics('./config/switzerland.yaml')
dyn.start_epidemic()
dyn.draw_map()
plt.show()
