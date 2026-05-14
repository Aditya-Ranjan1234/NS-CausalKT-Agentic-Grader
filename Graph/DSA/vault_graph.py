# import networkx as nx
# import os

# G = nx.read_graphml("graphs_test/DSA/knowledge_graph.graphml")

# # os.makedirs("vault", exist_ok=True)

# # for node in G.nodes():
# #     filename = os.path.join("vault", f"{node}.md")

# #     neighbors = list(G.neighbors(node))

# #     with open(filename, "w") as f:
# #         f.write(f"# {node}\n\n")

# #         if neighbors:
# #             f.write("## Related Concepts\n\n")

# #             for n in neighbors:
# #                 f.write(f"- [[{n}]]\n")

# first_node = list(G.nodes())[0]

# print("Node ID:", first_node)
# print("Attributes:", G.nodes[first_node])


import networkx as nx
import os
import re

# Load graph
G = nx.read_graphml("graphs_test/DSA/knowledge_graph.graphml")

# Vault folder
VAULT_DIR = "Knowledge_Graph_Vault"

os.makedirs(VAULT_DIR, exist_ok=True)

# Clean filenames
def safe_filename(name):
    return re.sub(r'[\\/:"*?<>|]+', '_', name)

# Create notes
for node, attrs in G.nodes(data=True):

    concept = attrs.get("label", str(node))

    filename = safe_filename(concept) + ".md"
    path = os.path.join(VAULT_DIR, filename)

    # Outgoing neighbors
    neighbors = list(G.neighbors(node))

    with open(path, "w", encoding="utf-8") as f:

        f.write(f"# {concept}\n\n")

        f.write("---\n")
        f.write(f"concept: {concept}\n")
        f.write(f"degree: {G.degree(node)}\n")
        f.write("---\n\n")

        f.write("## Related Concepts\n\n")

        if neighbors:

            for nbr in neighbors:

                nbr_label = G.nodes[nbr].get("label", str(nbr))

                f.write(f"- [[{nbr_label}]]\n")

        else:
            f.write("No related concepts.\n")

print(f"\nVault created: {VAULT_DIR}")
print("Open this folder in Obsidian.")