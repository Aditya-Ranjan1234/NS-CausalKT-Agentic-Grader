import networkx as nx
from pyvis.network import Network

# Load graph
G = nx.read_graphml("graphs_test/DSA/knowledge_graph.graphml")

print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())

# Create visualization
net = Network(
    height="50vh",
    width="50%",
    bgcolor="#111111",
    font_color="white",
    directed=True
)

# Physics engine
net.barnes_hut()

# Add graph
net.from_nx(G)

# Extra controls
net.show_buttons(filter_=['physics'])

# IMPORTANT FIX
net.write_html("graph.html", notebook=False)

print("Interactive graph saved as graph.html")