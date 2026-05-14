import torch
import torch.nn as nn

class SAKT(nn.Module):
    """
    Standard Self-Attentive Knowledge Tracing (SAKT) Baseline Model.
    Uses a Transformer-style Multi-Head Attention to trace student knowledge.
    """
    def __init__(self, num_questions, num_concepts, d_model=256, n_heads=4, n_layers=1, dropout=0.2, max_seq=200):
        super(SAKT, self).__init__()
        self.num_concepts = num_concepts
        self.d_model = d_model
        
        # Interaction embedding (concept + result)
        self.interaction_emb = nn.Embedding(num_concepts * 2 + 1, d_model, padding_idx=0)
        # Question/Concept embedding
        self.concept_emb = nn.Embedding(num_concepts + 1, d_model, padding_idx=0)
        # Position embedding
        self.pos_emb = nn.Embedding(max_seq, d_model)
        
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dim_feedforward=d_model * 4, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=n_layers)
        
        self.out_proj = nn.Linear(d_model, num_concepts)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, a, c, delta_t, target_c=None, src_mask=None, src_key_padding_mask=None):
        batch_size, seq_len = c.size()
        
        # Interaction index: 1-indexed concepts
        # a is 1 (incorrect) or 2 (correct). Convert to 0 (incorrect) or 1 (correct)
        a_binary = torch.clamp(a - 1, min=0)
        interaction_idx = c + self.num_concepts * a_binary
        interaction_idx[c == 0] = 0
        
        # E is interaction embeddings
        E = self.interaction_emb(interaction_idx) # (B, S, D)
        
        # Q is concept embeddings for current timestep
        Q_emb = self.concept_emb(c) # (B, S, D)
        
        # Position embeddings
        pos = torch.arange(seq_len).unsqueeze(0).to(c.device)
        P = self.pos_emb(pos) # (1, S, D)
        
        # In SAKT, we attend to PAST interactions to predict CURRENT outcome
        # So we shift interaction embeddings by 1 to represent history
        history_E = torch.zeros_like(E)
        history_E[:, 1:, :] = E[:, :-1, :]
        
        x = history_E + P
        
        # Causal mask to ensure we only look at past
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(c.device)
        
        # Transformer processing
        # mask=mask ensures attention only on previous steps
        out = self.transformer(x, mask=mask)
        out = self.dropout(out)
        
        # Final logits
        logits = self.out_proj(out)
        
        if target_c is not None:
             safe_target = torch.clamp(target_c, 0, self.num_concepts - 1)
             target_logits = torch.gather(logits, 2, safe_target)
             y_hat = torch.sigmoid(target_logits)
        else:
             y_hat = torch.sigmoid(logits)
             
        return y_hat, y_hat, torch.sigmoid(logits), out
