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
        try:    
            # simulation parameters
            self.alpha = _params['alpha']
            self.var_alpha = _params['var_alpha']
            self.beta = _params['beta']
            self.var_beta = _params['var_beta']
            self.eta = _params['eta']
            self.var_eta = _params['var_eta']
            self.gamma = _params['gamma']
            self.var_gamma = _params['var_gamma']
            self.zeta = _params['zeta']
            self.var_zeta = _params['var_zeta']
            self.tau_0 = _params['tau_0']
            self.var_tau_0 = _params['var_tau_0']
            self.dt = _params['dt']
            
            # action parameters
            self.confinement_effectiveness = _params['confinement_effectiveness']
            self.isolation_effectiveness = _params['isolation_effectiveness']
            self.extra_hospital_effectiveness = _params['extra_hospital_effectiveness']
            self.vaccination_effectiveness = _params['vaccination_effectiveness']
            self.env_step_length = _params['env_step_lenght']
            
            # cities and roads lists
            self.cities = list(_params['cities'].keys())
            if _params['roads'] is not None:
                self.roads = _params['roads']
            else:
                self.roads = []
            
            # generating a graph from the roads and cities
            self.map = nx.Graph()
            self.map.add_nodes_from(self.cities)
            self.map.add_edges_from(self.roads)
            
            self.pos_map = {}
            for c in self.cities:
                self.map.nodes[c]['pop'] = _params['cities'][c]['pop'] 
                self.pos_map[c] = [_params['cities'][c]['lat'],_params['cities'][c]['lon']]
        except:
            raise("Invalid YAML scenario file")
    
        # initializing the variables        
        nx.set_node_attributes(self.map, 1., "s")   
        nx.set_node_attributes(self.map, 0., "e")   
        nx.set_node_attributes(self.map, 0., "i")   
        nx.set_node_attributes(self.map, 0., "r")   
        nx.set_node_attributes(self.map, 0., "d")   
        
        # initializing the control parameters
        self.c_confined = {c:1 for c in self.cities}
        self.c_isolated = {c:1 for c in self.cities}
        self.extra_hospital_beds = {c:1 for c in self.cities}
        self.vaccinate = 0
        
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
        suceptible_total = 0
        exposed_total = 0
        infected_total = 0
        recovered_total = 0
        dead_total = 0
        total = 0
        
        for c in self.cities:
            suceptible = int(np.floor(self.map.nodes[c]['s'] * self.map.nodes[c]['pop']))
            suceptible_total += suceptible
            exposed = int(np.floor(self.map.nodes[c]['e'] * self.map.nodes[c]['pop']))
            exposed_total += exposed
            infected = int(np.floor(self.map.nodes[c]['i'] * self.map.nodes[c]['pop']))
            infected_total += infected
            recovered = int(np.floor(self.map.nodes[c]['r'] * self.map.nodes[c]['pop']))
            recovered_total += recovered
            dead = int(np.floor(self.map.nodes[c]['d'] * self.map.nodes[c]['pop']))
            dead_total += dead
            total += self.map.nodes[c]['pop']
            
            city = {
                'suceptible' : suceptible,
                'exposed' : exposed,
                'infected' : infected,
                'recovered' : recovered,
                'dead' : dead,
                'initial population' : self.map.nodes[c]['pop']
            }
            cities[c] = city
        
        total = {
            'suceptible' : suceptible_total,
            'exposed' : exposed_total,
            'infected' : infected_total,
            'recovered' : recovered_total,
            'dead' : dead_total,
            'initial population' : total
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
    def start_epidemic(self,seed=10, sources=1, prop=0.01):
        rd.seed(seed)
        
        start_cities = rd.choices(self.cities, k = sources)
        for c in start_cities:
            self.map.nodes[c]['e'] += prop
            self.map.nodes[c]['s'] -= prop
    
    """ Perform one environment step
        
        Parameters : 
            act [dict] : a dict containing actions to be performed this turn
                    the dict must be structured as 
                    confinement :   city,True/False
                    isolation :     city,True/False
                    hospital :      city,True/False
                    vaccination :   True/False

        Returns : 
            obs [dict] : a dict containing observation from this step
                    the dict is structured as 
                    total   :   list(dict(measurements)) (over 7 days)
                    city    :   dict(list(dict(measurements))) (over 7 days, per city)
                    pop    :    dict(int) initial per-city population (to allow normalization)

    """
    def step(self, act):
        
        # change the model parameters according to the actions
        self.c_confined = {c:(self.confinement_effectiveness if v else 1) 
                           for (c,v) in act['confinement'].items()}
        self.c_isolated = {c:(self.isolation_effectiveness if v else 1) 
                           for (c,v) in act['isolation'].items()}
        self.vaccinate = self.vaccination_effectiveness if act['vaccinate'] else 0
        self.extra_hospital_beds = {c:(self.extra_hospital_effectiveness if v else 1) 
                                    for (c,v) in act['hospital'].items()}
        _total_history = []
        _city_history = []
        # print step through a week of simulation to produce one environment step
        for i in range(self.env_step_length):
            self.step_dyn()
            total, cities = self.epidemic_parameters()
            _total_history.append(total)
            _city_history.append(cities)
        
        # output observations
        _total_infected = [t['infected'] for t in _total_history][0::3]
        _total_dead = [t['dead'] for t in _total_history][0::3]
        _total = {
            'infected' :    _total_infected,
            'dead' :        _total_dead,
        }
        _city_infected = {c:([t[c]['infected'] for t in _city_history][0::3]) for c in self.cities}
        _city_dead = {c:([t[c]['dead'] for t in _city_history][0::3]) for c in self.cities}
        _city = {
            'city' :    _city_infected,
            'dead' :        _city_dead,
        }
        _pop = {c:self.map.nodes[c]['pop'] for c in self.cities}
        
        obs = {
            'total' :   _total,
            'city'  :   _city,
            'pop'   :   _pop
        }
        return obs
    
    """ Step forward in the epidemic dynamics
        
        Parameters : 
            None

        Returns : 
            None
    """
    def step_dyn(self):
        ds = {}
        de = {}
        di = {}
        dr = {}
        dd = {}
        
        for c in self.cities:
            
            # query the variables from the graph
            s = self.map.nodes[c]['s']
            e = self.map.nodes[c]['e']
            i = self.map.nodes[c]['i']
            r = self.map.nodes[c]['r']
            d = self.map.nodes[c]['d']
            
            # compute the derivative terms
            
            # city - to city contagion
            stoch_t0 = np.max([np.random.normal(self.tau_0,self.var_tau_0),0]) 
            sum_term = self.c_isolated[c]*stoch_t0 * np.sum([self.map.nodes[a]['i']*self.map.edges[(a,c)]['tau']*self.c_isolated[a] for a in nx.neighbors(self.map,c)])
            
            # incidence rate
            stoch_alpha = np.max([np.random.normal(self.alpha*self.c_confined[c],self.var_alpha),0]) 
            new_exposed = stoch_alpha * (s * i  + sum_term)
            
            # vaccination
            stoch_mu = self.vaccinate/self.map.nodes[c]['pop'] 
            new_vaccinated = np.fmax(np.fmin(float(stoch_mu*s), float(stoch_mu)),0)
            
            # exposure to infection flow
            stoch_eta = np.max([np.random.normal(self.eta,self.eta),0])
            new_infected = stoch_eta * e
            
            # exposure to recovered flow
            stoch_beta = np.max([np.random.normal(self.beta,self.var_beta),0])
            new_recovered = stoch_beta * i
            
            # death rate
            stoch_zeta = np.max([np.random.normal(self.zeta,self.var_zeta),0]) 
            new_deaths = stoch_zeta * i * i * self.extra_hospital_beds[c]
            
            # loss of immunity rate
            stoch_gamma = np.max([np.random.normal(self.gamma,self.var_gamma),0])
            new_suceptible = stoch_gamma * r  

            # compute the derivatives
            ds[c] = new_suceptible - new_exposed - new_vaccinated
            de[c] = new_exposed - new_infected
            di[c] = new_infected - new_recovered - new_deaths
            dr[c] = new_recovered - new_suceptible + new_vaccinated
            dd[c] = new_deaths
            
        for c in self.cities: 
            # Euler integration step
            self.map.nodes[c]['s'] += ds[c]*self.dt
            self.map.nodes[c]['e'] += de[c]*self.dt
            self.map.nodes[c]['i'] += di[c]*self.dt
            self.map.nodes[c]['r'] += dr[c]*self.dt
            self.map.nodes[c]['d'] += dd[c]*self.dt