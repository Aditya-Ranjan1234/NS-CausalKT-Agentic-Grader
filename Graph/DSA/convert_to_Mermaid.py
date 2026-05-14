import networkx as nx

# Load graph
G = nx.read_graphml("graphs_test/DSA/knowledge_graph.graphml")

# Limit size
MAX_EDGES = 100

edges = list(G.edges())[:MAX_EDGES]

lines = ["graph TD"]

for u, v in edges:
    u_clean = str(u).replace(" ", "_")
    v_clean = str(v).replace(" ", "_")

    lines.append(f'    {u_clean} --> {v_clean}')

mermaid_text = "\n".join(lines)

with open("graph.mmd", "w") as f:
    f.write(mermaid_text)

print("Saved graph.mmd")