import networkx as nx
import nx_cugraph as nxcg # gpu  of network x
import matplotlib.pyplot as plt
import numpy as np
import os 
import json
G = nx.read_edgelist(path='./facebook_combined.txt')

os.environ['NX_CUGRAPH_AUTOCONFIG'] = "True"
def main():
    #takes about 2 minutes to get page_rank and centrality
    page_rank = sorted(nxcg.pagerank(G).items(), key=lambda item: item[1], reverse=True)
    centrality = sorted(nxcg.betweenness_centrality(G).items(), key=lambda item: item[1], reverse=True)
    clustering = sorted(nxcg.clustering(G).items(), key=lambda item: item[1], reverse=True)
    if(not os.path.exists('rank.json')):
        with open('rank.json', mode='w') as f:
            f.write(json.dumps(dict(page_rank)))
    if(not os.path.exists('centrality.json')):
        with open('centrality.json', mode='w') as f:
            f.write(json.dumps(dict(centrality)))
    if(not os.path.exists('clustering.json')):
        with open('clustering.json', mode='w') as f:
            f.write(json.dumps(dict(clustering)))

    ranks = [rank for node, rank in page_rank]
    cen = [centrality for node, centrality in centrality]
    cl = [c for node, c in clustering]
    fig, axes =plt.subplots(1, 3)
    
    print('='*8+'Page_rank'+'='*8)
    print(f'max rank: {page_rank[0][1]}')
    print(f'min rank: {page_rank[-1][1]}')
    print(f'std: {np.std([r for n, r in page_rank])}')
    print(f'average: {np.average(ranks)}')
    print('\tTop 10 rank')
    for node, rank in page_rank[:10]:
        print(f'node: {node} - {rank:.5f}')
    axes[0].boxplot(ranks)
    axes[0].set_title('Rank')
    axes[0].set_yscale('log')
    
    print('='*8+'Centrality'+'='*8)
    print(f'max: {centrality[0][1]}')
    print(f'min rank: {centrality[-1][1]}')
    print(f'std: {np.std([c for n, c in centrality])}')
    print(f'average: {np.average(cen)}')
    print('\tTop 10 centrality')
    for node, centrality in centrality[:10]:
        print(f'node {node} - {centrality:.2f}')
    axes[1].boxplot(cen)
    axes[1].set_title('Centrality')
    axes[1].set_yscale('log')

    print('='*8+'Clustering'+'='*8)
    print(f'max rank: {clustering[0][1]}')
    print(f'min rank: {clustering[-1][1]}')
    print(f'std: {np.std([c for n, c in clustering])}')
    print(f'average: {np.average(cl)}')
    print('\tTop 10 clustering')
    for node, cluster in clustering[:10]:
        print(f'node: {node} - {cluster:.2f}')
    axes[2].boxplot(cl)
    axes[2].set_title('Clusterin')
    
    plt.figure(figsize=(14,10))
    pos = nx.spring_layout(G, k=10, scale=2, iterations=10)
    nx.draw_networkx_nodes(G, edgecolors='black', node_color = 'lightblue', pos = pos, node_size=50)
    nx.draw_networkx_edges(G, pos = pos, width=0.1)
    plt.margins(0.0)
    plt.show()

if __name__ == '__main__':
    main()


