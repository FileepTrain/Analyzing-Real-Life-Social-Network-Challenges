import networkx as nx
#import nx_cugraph as nxcg
import matplotlib.pyplot as plt
import os 
from time import time
G = nx.read_edgelist(path='./facebook_combined.txt')

#os.environ['NX_CUGRAPH_AUTOCONFIG'] = "True"
def main():
    #takes about 2 minutes to get page_rank and centrality
    page_rank = sorted(nx.pagerank(G).items(), key=lambda item: item[1], reverse=True)
    centrality = sorted(nx.betweenness_centrality(G).items(), key=lambda item: item[1], reverse=True)
    ranks = [rank for node, rank in page_rank]
    cen = [centrality for node, centrality in centrality]
    fig, axes =plt.subplots(1, 2)
    
    print('='*8+'Page_rank'+'='*8)
    print(f'max rank: {page_rank[0][1]}')
    print(f'average: {sum(ranks)/len(page_rank)}')
    print('\tTop 10 rank')
    for node, rank in page_rank[:10]:
        print(f'node: {node} - {rank:.5f}')
    axes[0].boxplot(ranks)
    axes[0].set_title('Rank')
    axes[0].set_yscale('log')
    
    print('='*8+'Centrality'+'='*8)
    print(f'max: {centrality[0][1]}')
    print(f'average: {sum(cen)/len(centrality)}')
    print('\tTop 10 centrality')
    for node, centrality in centrality[:10]:
        print(f'node {node} - {centrality:.2f}')
    axes[1].boxplot(cen)
    axes[1].set_title('Centrality')
    axes[1].set_yscale('log')
    
    plt.figure(figsize=(16,12))
    pos = nx.spring_layout(G, k=3, iterations=10)
    nx.draw(G, with_labels=False, node_size=50, pos =pos)
    plt.show()

if __name__ == '__main__':
    main()


