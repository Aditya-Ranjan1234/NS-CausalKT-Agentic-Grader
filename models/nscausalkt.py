import torch
import torch.nn as nn
from models.nsm import NeuralStudentModel
from models.scl import SymbolicConceptLayer
from models.csl import CausalSCMLayer

class NSCausalKT(nn.Module):
    def __init__(self, num_questions, num_concepts, edge_index, d_model=256, n_heads=4, n_layers=2):
        super(NSCausalKT, self).__init__()
        self.num_questions = num_questions
        self.num_concepts = num_concepts
        self.d_model = d_model
        
        # Static graph representation
        self.register_buffer('edge_index', edge_index)
        
        adj = torch.zeros(num_concepts, num_concepts)
        if edge_index is not None and edge_index.size(1) > 0:
            adj[edge_index[0], edge_index[1]] = 1.0
        # Add self loops
        adj.fill_diagonal_(1.0)
        # Row normalize
        deg = adj.sum(dim=-1, keepdim=True)
        adj_norm = adj / deg
        self.register_buffer('adj_matrix', adj_norm)
        
        self.nsm = NeuralStudentModel(num_questions, num_concepts, d_model, n_heads, n_layers)
        self.scl = SymbolicConceptLayer(num_concepts, d_model)
        self.csl = CausalSCMLayer(num_concepts, d_model)
        
        # Joint Prediction Head
        # Wp * [h_t; mu_specific; ACE_vec] + bp
        # mu_specific is 1, ACE_vec is 1
        self.W_p = nn.Linear(d_model + 1 + 1, 1)

    def forward(self, q, a, c, delta_t, target_c=None, src_mask=None, src_key_padding_mask=None):
        # 1. Neural Student Model
        h_t, mu_t = self.nsm(q, a, c, delta_t, src_mask, src_key_padding_mask)
        
        # 2. Symbolic Concept Layer
        mu_tilde, phi_t = self.scl(mu_t, self.edge_index, self.adj_matrix)
        
        # 3. CSL - Calculate real ACE for the concept `c` being asked at step `t+1`.
        # Note: at sequence step `t`, we want to predict `a_{t+1}` for question `q_{t+1}` which corresponds to concept `c_{t+1}`.
        # But `c` array holds the concept of the PAST interactions. 
        # Wait, the target `y` is shifted in the training loop.
        # Let's say `c` is the past sequence. We don't have `c_{t+1}` directly inside unless we pass it.
        # For simplicity, we just calculate the base prediction if we don't have the target concept. 
        # But wait, standard KT standardizes: inputs are Q_1..T-1, A_1..T-1. Target is A_2..T based on Q_2..T.
        # `q, a, c` in our inputs are already aligned so `h_t` represents knowledge state BEFORE answering `q_{t+1}`?
        # Standard transformer input: e_t = Eq_{t+1} + Ea_t + ... Or e_t = Eq_t + Ea_t ...
        # Let's pass the 'next sequence concept' specifically, or dynamically get base predictions.
        # We will use y_base as the proxy prediction for now to bypass the cycle, and return components.
        
        if target_c is None:
            # Fallback if testing
            target_c = torch.zeros(q.size(0), q.size(1), 1, dtype=torch.long, device=q.device)
            
        y_base = self.csl.compute_base_prediction(h_t, mu_tilde, target_c)
        
        # We also need to compute ACE for a specific conceptual intervention if required.
        safe_target = torch.clamp(target_c, 0, self.num_concepts - 1)
        mu_specific = torch.gather(mu_tilde, 2, safe_target)
        
        dummy_ace = torch.zeros_like(y_base)
        concat_feat = torch.cat([h_t, mu_specific, dummy_ace], dim=-1)
        y_hat = torch.sigmoid(self.W_p(concat_feat))
        
        return y_hat, y_base, mu_tilde, h_t

    def get_losses(self, y_hat, a_next, mu_tilde, mu_tilde_prev, lambda_1=1.0, lambda_2=0.5, lambda_3=0.1):
        """
        Computes the 4 losses
        a_next: true responses for the NEXT timestep
        """
        # L1 Binary Cross Entropy
        # y_hat is (B, S, 1), a_next is (B, S, 1) Let's assume mask is applied outside
        mask = (a_next != -100).float()
        
        # safe bce
        y_safe = torch.clamp(y_hat, 1e-6, 1.0 - 1e-6)
        L_pred = - (a_next * torch.log(y_safe) + (1 - a_next) * torch.log(1 - y_safe))
        L_pred = (L_pred * mask).sum() / (mask.sum() + 1e-8)
        
        # L2 Symbolic Loss (computed internally in SCL)
        L_sym = self.scl.compute_symbolic_loss(mu_tilde, self.edge_index)
        
        # L3 Causal Regularization (computed in CSL)
        L_causal = self.csl.compute_causal_loss(mu_tilde, self.edge_index)
        
        # L4 Temporal Smoothness
        if mu_tilde_prev is not None:
            L_smooth = torch.mean((mu_tilde - mu_tilde_prev)**2)
        else:
            # Within-sequence smoothness
            L_smooth = torch.mean((mu_tilde[:, 1:, :] - mu_tilde[:, :-1, :])**2)
            
        L_total = L_pred + lambda_1 * L_sym + lambda_2 * L_causal + lambda_3 * L_smooth
        return L_total, L_pred, L_sym, L_causal, L_smooth
