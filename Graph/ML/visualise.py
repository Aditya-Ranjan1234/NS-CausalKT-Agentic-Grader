import networkx as nx
import matplotlib.pyplot as plt

G = nx.read_graphml("/Users/gnanendranaidun/Downloads/data_rar/ACL 2017/graphs_test/ML/knowledge_graph.graphml")

plt.figure(figsize=(24, 18))
pos = nx.spring_layout(G, seed=42, k=0.35)
nx.draw_networkx_nodes(G, pos, node_size=80)
nx.draw_networkx_edges(G, pos, arrows=True, alpha=0.25, width=0.7)
nx.draw_networkx_labels(G, pos, font_size=6)
plt.axis("off")
plt.show()