import torch
import torch.nn as nn
import torch.nn.functional as F

class SymbolicConceptLayer(nn.Module):
    def __init__(self, num_concepts, d_model):
        super(SymbolicConceptLayer, self).__init__()
        self.num_concepts = num_concepts
        # Gating network
        self.gate_proj = nn.Linear(num_concepts + 1, num_concepts)
        
        # Simple GCN weights for f_sym propagation
        self.W_sym = nn.Parameter(torch.randn(num_concepts, num_concepts))
        nn.init.xavier_uniform_(self.W_sym)

    def compute_phi(self, mu_t, edge_index):
        """
        mu_t: (batch, seq, num_concepts) concept mastery
        edge_index: (2, num_edges) edge list where edge_index[0, k] -> edge_index[1, k] means 0 is prereq of 1
        """
        if edge_index is None or edge_index.size(1) == 0:
            return torch.ones(mu_t.size(0), mu_t.size(1), 1, device=mu_t.device)

        # Rule R1: c_j -> c_i means if c_i is mastered, c_j must be mastered.
        # phi = min(1, 1 - mu_i + mu_j)
        # edge_index[0] = j (prereq), edge_index[1] = i (dependent)
        
        j_idx = edge_index[0]
        i_idx = edge_index[1]
        
        mu_i = mu_t[..., i_idx] # (batch, seq, num_edges)
        mu_j = mu_t[..., j_idx] # (batch, seq, num_edges)
        
        phi_e = torch.clamp(1.0 - mu_i + mu_j, max=1.0) # (batch, seq, num_edges)
        
        # Average over all edges
        phi_t = phi_e.mean(dim=-1, keepdim=True) # (batch, seq, 1)
        return phi_t
        
    def forward(self, mu_t, edge_index, adj_matrix):
        """
        mu_t: Neural concept state (batch, seq, num_concepts)
        edge_index: (2, num_edges)
        adj_matrix: (num_concepts, num_concepts) symmetric normalized adjacency with self loops
        """
        phi_t = self.compute_phi(mu_t, edge_index)
        
        # Symbolic correction gate
        # g = sigmoid(W_g * [mu_t; phi_t])
        gate_input = torch.cat([mu_t, phi_t.expand(-1, -1, mu_t.size(1)) if phi_t.size(1)==1 else phi_t], dim=-1) # (batch, seq, num_concepts + 1)
        # wait, phi_t is (batch, seq, 1). So cat gives (batch, seq, num_concepts + 1)
        gate_input = torch.cat([mu_t, phi_t], dim=-1)
        g = torch.sigmoid(self.gate_proj(gate_input)) # (batch, seq, num_concepts)
        
        # f_sym: standard dense GCN pass: A_hat * mu_t * W
        # mu_t is (batch, seq, num_concepts)
        # A_hat is (num_concepts, num_concepts)
        
        # mu_t * W_sym:
        h_sym = torch.matmul(mu_t, self.W_sym) # (batch, seq, num_concepts)
        # multiply by adjacency:
        # We need to compute A_hat @ h_sym. Since A_hat is (C,C), we can do:
        # einsum: 'cc, bsc -> bsc'  Wait, A_hat @ h_sym over concept dim.
        f_sym = torch.matmul(h_sym, adj_matrix.t()) # (batch, seq, num_concepts)
        f_sym = torch.sigmoid(f_sym) # bounded in [0,1]
        
        mu_tilde = g * mu_t + (1 - g) * f_sym
        
        return mu_tilde, phi_t

    def compute_symbolic_loss(self, mu_tilde, edge_index):
        if edge_index is None or edge_index.size(1) == 0:
            return torch.tensor(0.0, device=mu_tilde.device)
            
        j_idx = edge_index[0]
        i_idx = edge_index[1]
        
        mu_i = mu_tilde[..., i_idx]
        mu_j = mu_tilde[..., j_idx]
        
        # L_sym = - sum [mu_j * log(mu_i) + (1-mu_j)*log(1-mu_i)]
        # This is cross entropy: encouraging mu_i to match mu_j in probability distribution
        # But paper says: "mastering c_i should correlate with mastering c_j"
        # The formula in paper: L_sym = - Sum [ mu_j * log mu_i + (1-mu_j)*log(1-mu_i) ]
        # wait, if mu_j is 1, it forces log mu_i -> mu_i=1. If mu_j=0, forces mu_i=0. 
        # so mu_i MUST follow mu_j. This makes sense for prereq.
        
        loss = - (mu_j * torch.log(mu_i + 1e-8) + (1.0 - mu_j) * torch.log(1.0 - mu_i + 1e-8))
        return loss.mean()
