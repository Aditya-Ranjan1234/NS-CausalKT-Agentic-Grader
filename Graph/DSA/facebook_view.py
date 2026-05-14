import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import animation

# Load graph from a GraphML file
# Replace with your actual file path
graphml_file = "graphs_test/DSA/knowledge_graph.graphml"

G = nx.read_graphml(graphml_file)

# Optional: convert node labels if needed
# Example: if node IDs are integers stored as strings
# G = nx.relabel_nodes(G, lambda x: int(x))

# Select a node for ego graph
# Use an existing node from the graph
node = list(G.nodes())[0]

radius = 2
ego = nx.ego_graph(G, node, radius)

# Generate 3D layout
pos = nx.spring_layout(ego, dim=3, seed=25519)

# Convert positions into arrays for plotting
nodes = np.array([pos[v] for v in ego.nodes()])
edges = np.array([(pos[u], pos[v]) for u, v in ego.edges()])

point_size = max(20, 1000 // np.sqrt(len(ego)))


def init():
    ax.clear()

    # Draw nodes
    ax.scatter(
        nodes[:, 0],
        nodes[:, 1],
        nodes[:, 2],
        alpha=0.8,
        s=point_size,
        edgecolors="white",
    )

    # Draw edges
    for edge in edges:
        ax.plot(
            edge[:, 0],
            edge[:, 1],
            edge[:, 2],
            color="tab:gray",
            alpha=0.5,
        )

    ax.grid(False)
    ax.set_axis_off()


def _frame_update(idx):
    ax.view_init(elev=idx * 0.9, azim=idx * 1.8)


fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection="3d")

fig.tight_layout()

anim = animation.FuncAnimation(
    fig,
    func=_frame_update,
    init_func=init,
    interval=50,
    cache_frame_data=False,
    frames=120,
)

plt.show()