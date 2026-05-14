import torch
import torch.nn as nn

class DKT(nn.Module):
    """
    Standard Deep Knowledge Tracing (DKT) Baseline Model.
    Uses an LSTM to trace student knowledge representations without Causal graphs.
    """
    def __init__(self, num_concepts, d_model=256, n_layers=1, dropout=0.2):
        super(DKT, self).__init__()
        self.num_concepts = num_concepts
        # num_concepts * 2 accounts for concept_id answered incorrectly vs correctly.
        self.interaction_emb = nn.Embedding(num_concepts * 2 + 1, d_model, padding_idx=0)
        
        self.lstm = nn.LSTM(d_model, d_model, num_layers=n_layers, batch_first=True, dropout=dropout if n_layers > 1 else 0.0)
        
        self.out_proj = nn.Linear(d_model, num_concepts)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, a, c, delta_t, target_c=None, src_mask=None, src_key_padding_mask=None):
        # In dataset.py, 'a' was shifted: a_shifted = a + 1. (0 is pad, 1 is incorrect, 2 is correct).
        a_binary = torch.clamp(a - 1, min=0)
        interaction_idx = c + self.num_concepts * a_binary
        
        # Zero out padding interaction
        pad_mask = (c == 0)
        interaction_idx[pad_mask] = 0
        
        X = self.interaction_emb(interaction_idx)
        out, _ = self.lstm(X)
        out = self.dropout(out)
        
        # Logits for all concepts at each step
        logits = self.out_proj(out)
        
        if target_c is not None:
             safe_target = torch.clamp(target_c, 0, self.num_concepts - 1)
             target_logits = torch.gather(logits, 2, safe_target)
             y_hat = torch.sigmoid(target_logits)
        else:
             y_hat = torch.sigmoid(logits)
             
        # Dummy returns to perfectly match the NSCausalKT tuple signature
        # allowing seamless interchange in eval scripts
        return y_hat, y_hat, torch.sigmoid(logits), out
