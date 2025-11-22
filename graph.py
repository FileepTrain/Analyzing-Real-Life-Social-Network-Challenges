import networkx as nx
#import nx_cugraph as nxcg
import matplotlib.pyplot as plt
import os 

G = nx.read_edgelist(path='./facebook_combined.txt')

#os.environ['NX_CUGRAPH_AUTOCONFIG'] = "True"

nx.draw(G)
plt.show()
