import matplotlib.pyplot as plt
import networkx as nx
import random
import numpy as np
import pandas as pd
import pickle as pk
from tqdm import tqdm

data_path = "data/"

N=100 #size of the population
n_groups = 2
prop = np.array([0.5, 0.5])

groups = {"id":[n for n in range(n_groups)],
          "prop" : prop,
          "size" : prop*N,
          "nodes": []
        }

#G = nx.Graph() #Graph
#nds = range(0,N) #nodes
#G.add_nodes_from(nds)

G = nx.complete_graph(N)

# Distribution of individuals
selected = []
for g in groups["id"]:
    nodes = []
    for _ in range(int(groups["size"][g])):
        i = random.randint(0,N - 1)
        while i in selected:
            i = random.randint(0,N - 1)            
        nodes.append(i)
        selected.append(i)
        G.nodes[i]["group"] = g
    groups["nodes"].append(nodes)

#nx.draw(G)

# Distribution of states
# 0: e+
# 1: e-
# 2: e0
for i in G.nodes():
    G.nodes[i]["e"] = random.randint(0,2)


outcome_intra = np.matrix([[ 1, -1,  1],
                     [-1, -1, -1],
                     [ 1, -1,  0]])

outcome_inter = np.matrix([[-1, -1, -1],
                     [-1,  1,  1],
                     [-1,  1,  0]])

update_intra = np.matrix([[ 0, 1, 0],
                          [ 1, 1, 1],
                          [ 0, 1, 2]])

update_inter = np.matrix([[ 0, 0, 0],
                          [ 0, 1, 1],
                          [ 0, 1, 2]])

# SIMULATION
T = N**2

states = dict()
outcomes_intra = dict()
outcomes_inter = dict()
for i in G.nodes():
    states.update({i : [G.nodes[i]["e"]]})
    outcomes_intra.update({i : []})
    outcomes_inter.update({i : []})

for t in tqdm(range(T)):
    i = random.randint(0,N - 1)
    j = random.choice(list(G.edges(0)))[1]
    if G.nodes[i]["group"] == G.nodes[j]["group"]:
        e_i = G.nodes[i]["e"]
        e_j = G.nodes[j]["e"]
        
        outcomes_intra[i].append(outcome_intra[e_i,e_j]) 
        outcomes_intra[j].append(outcome_intra[e_j,e_i]) 
        states[i].append(update_intra[e_i,e_j]) 
        states[j].append(update_intra[e_j,e_i])
        
        G.nodes[i]["e"] = update_intra[e_i,e_j]
        G.nodes[j]["e"] = update_intra[e_j,e_i]
    else:
        e_i = G.nodes[i]["e"]
        e_j = G.nodes[j]["e"]
        
        outcomes_inter[i].append(outcome_inter[e_i,e_j]) 
        outcomes_inter[j].append(outcome_inter[e_j,e_i]) 
        states[i].append(update_inter[e_i,e_j]) 
        states[j].append(update_inter[e_j,e_i])
        
        G.nodes[i]["e"] = update_inter[e_i,e_j]
        G.nodes[j]["e"] = update_inter[e_j,e_i]


with open(f"{data_path}groups.pkl", "wb") as f:
    pk.dump(groups, f)

with open(f"{data_path}outcomes_intra.pkl", "wb") as f:
    pk.dump(outcomes_intra, f)

with open(f"{data_path}outcomes_inter.pkl", "wb") as f:
    pk.dump(outcomes_inter, f)
    
with open(f"{data_path}network.pkl", "wb") as f:
    pk.dump(G, f)