import networkx as nx
#import nx_cugraph as nxcg
import matplotlib.pyplot as plt
import os 
from time import time
G = nx.read_edgelist(path='./facebook_combined.txt')

#os.environ['NX_CUGRAPH_AUTOCONFIG'] = "True"
def main():
    start = time()
    #takes about 2 minutes to get page_rank and centrality
    page_rank = nx.pagerank(G)
    centrality = nx.betweenness_centrality(G)
    print(str(abs(time() - start)))
    # nx.draw(G)
    # plt.show()

if __name__ == '__main__':
    main()


