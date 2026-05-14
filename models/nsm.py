import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class NeuralStudentModel(nn.Module):
    def __init__(self, num_questions, num_concepts, d_model=256, n_heads=4, n_layers=2):
        super(NeuralStudentModel, self).__init__()
        self.d_model = d_model
        
        # We split d_model into 4 parts as per paper: e_t = Eq + Ea + Ec + Edelta
        # However, it's easier to embed all to d_model and sum, or concat to d_model.
        # Paper says: e_t = Embed(q_t) + Embed(a_t) + Embed(c_t) + delta_t (concat)
        # Let's use concatenation: d_model must be divisible by 4.
        self.d_part = d_model // 4
        
        self.q_emb = nn.Embedding(num_questions + 1, self.d_part, padding_idx=0)
        self.a_emb = nn.Embedding(3, self.d_part, padding_idx=0) # 0: pad, 1: incorrect, 2: correct
        self.c_emb = nn.Embedding(num_concepts + 1, self.d_part, padding_idx=0)
        self.delta_proj = nn.Linear(1, self.d_part)
        
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # GRU Refinement
        self.gru = nn.GRU(input_size=d_model, hidden_size=d_model, batch_first=True)
        
        # Concept projection head
        self.concept_proj = nn.Linear(d_model, num_concepts)
        
    def forward(self, q, a, c, delta_t, src_mask=None, src_key_padding_mask=None):
        """
        q, a, c: (batch_size, seq_len)
        delta_t: (batch_size, seq_len, 1)
        """
        # Embeddings
        # Convert dense representations
        eq = self.q_emb(q)
        ea = self.a_emb(a)
        ec = self.c_emb(c)
        ed = self.delta_proj(delta_t)
        
        # e_t in R^d
        e_t = torch.cat([eq, ea, ec, ed], dim=-1)
        
        e_t = self.pos_encoder(e_t)
        
        # Transformer
        # We need causal mask (seq_len, seq_len)
        seq_len = e_t.size(1)
        if src_mask is None:
            src_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(e_t.device)
            
        H = self.transformer(e_t, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        
        # GRU
        H_gru, _ = self.gru(H)
        
        # Concept projection
        mu = torch.sigmoid(self.concept_proj(H_gru))
        
        return H_gru, mu
    
