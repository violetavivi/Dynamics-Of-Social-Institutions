import pickle as pk
import pandas as pd
import matplotlib.pyplot as plt

data_path = "data/"

with open(f"{data_path}groups.pkl", "rb") as f:
    groups = pk.load(f)
    
with open(f"{data_path}outcomes_inter.pkl", "rb") as f:
    outcomes_inter = pk.load(f)
    
with open(f"{data_path}outcomes_intra.pkl", "rb") as f:
    outcomes_intra = pk.load(f)

with open(f"{data_path}network.pkl", "rb") as f:
    G = pk.load(f)
   
def plot_dist_out(g, outcomes_dict, type):
    int_temp = []
    for i in groups["nodes"][g]:
        int_temp = int_temp + outcomes_dict[i]
    df1 = pd.DataFrame({"Outcome": int_temp})
    count = df1["Outcome"].value_counts().reset_index()
    count.columns = ["Group", "Freq"]
    count.Group = count.Group.astype(int)
    count.sort_values(by = "Group", inplace = True)
    count["label"] = count.Group.astype(str)
    fig, ax = plt.subplots()
    ax.bar(count.label, count.Freq)
    ax.set_xlabel("Outcome")
    ax.set_ylabel("Freq")
    ax.set_title(f"{type}-group interaction")
    fig.show()
    
def plot_dist_states(g):
    states_temp = []
    for i in groups["nodes"][g]:
        states_temp.append(G.nodes[i]["e"])
    df = pd.DataFrame({"State": states_temp})
    count = df.State.value_counts().reset_index()
    count.columns = ["Group", "Freq"]
    count.Group = count.Group.astype(int)
    count.sort_values(by = "Group", inplace = True)
    count["label"] = count.Group.astype(str)
    fig, ax = plt.subplots()
    ax.bar(count.label, count.Freq)    
    ax.set_title(f"{type}-group interaction")
    fig.show()
    
plot_dist_out(g = 0, outcomes_dict = outcomes_inter, type = "Inter")
plot_dist_out(g = 0, outcomes_dict = outcomes_intra, type = "Intra")

plot_dist_states(g = 1)

