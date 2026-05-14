import torch
from sklearn.metrics import roc_auc_score, mean_squared_error, accuracy_score
import numpy as np

def evaluate_metrics(y_true, y_pred):
    """
    Standard KT metrics calculation
    """
    if len(y_true) == 0:
         return {"auc": 0.0, "rmse": 0.0, "acc": 0.0}
         
    # Only keep valid predictions (filter out -100)
    valid_idx = y_true != -100
    yt = y_true[valid_idx]
    yp = y_pred[valid_idx]
    
    if len(np.unique(yt)) < 2:
        auc = 0.5
    else:
        auc = roc_auc_score(yt, yp)
        
    rmse = np.sqrt(mean_squared_error(yt, yp))
    acc = accuracy_score(yt, (yp > 0.5).astype(int))
    
    return {"auc": auc, "rmse": rmse, "acc": acc}

def evaluate_ic_k(model, dataset, k=5, device='cpu'):
    """
    Intervention Correctness (IC@k)
    Simulate "teach concept c_j" by giving k synthetic correct answers.
    Measure if prediction improves on dependant concepts c_i
    """
    model.eval()
    improvements = 0
    total = 0
    
    # We do a simplified version using the synthetic edge index from model.
    edge_index = model.edge_index
    if edge_index is None or edge_index.size(1) == 0:
        return 0.0
        
    j_idx = edge_index[0]
    i_idx = edge_index[1]
    
    # Check on a few edge pairs
    with torch.no_grad():
        # Get a dummy input from dataset
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
        for i, batch in enumerate(loader):
            if i > 50: break # test on max 50 samples to save time
            
            # get base prediction
            q = batch['q'].to(device)
            a = batch['a'].to(device)
            c = batch['c'].to(device)
            dt = batch['delta_t'].to(device)
            
            y_hat, _, _, _ = model(q, a, c, dt)
            base_pred = y_hat.squeeze()
            
            # intervene: teach concept j (add k correct interactions to seq)
            # Find an edge to test
            edge_to_test = np.random.randint(edge_index.size(1))
            c_j = j_idx[edge_to_test].item()
            c_i = i_idx[edge_to_test].item()
            
            # Synthetic history + intervention
            # We append 'k' correct interactions of c_j
            q_inter = list(q[0].cpu().numpy())
            a_inter = list(a[0].cpu().numpy())
            c_inter = list(c[0].cpu().numpy())
            dt_inter = list(dt[0].cpu().numpy())
            
            # Find first padding 0
            pad_idx = -1
            for idx, x in enumerate(q_inter):
                if x == 0:
                    pad_idx = idx
                    break
            if pad_idx == -1: pad_idx = len(q_inter)
            
            # Ensure we have space for k elements, else shift
            if pad_idx + k > len(q_inter):
                pad_idx = len(q_inter) - k
            
            for _ in range(k):
                q_inter[pad_idx] = 9999 # dummy question
                a_inter[pad_idx] = 2 # correct
                c_inter[pad_idx] = c_j
                dt_inter[pad_idx] = [0.0]
                pad_idx += 1
                
            q_do = torch.tensor([q_inter]).to(device)
            a_do = torch.tensor([a_inter]).to(device)
            c_do = torch.tensor([c_inter]).to(device)
            dt_do = torch.tensor([dt_inter]).to(device)
            
            y_hat_do, _, _, _ = model(q_do, a_do, c_do, dt_do)
            do_pred = y_hat_do.squeeze()
            
            # Wait, we need to evaluate exactly predicting concept `c_i`. 
            # We assume y_hat is aligned with `c_i`. But wait, y_hat predicts exactly the next position.
            # A more robust check computes the causal effect directly using computation graph or checks 
            # the internal CSL components.
            # Simplified: if the intervention increases the average concept mastery mu_tilde for c_i
            _, _, mu_tilde, _ = model(q, a, c, dt)
            _, _, mu_tilde_do, _ = model(q_do, a_do, c_do, dt_do)
            
            base_mastery = mu_tilde[0, -1, c_i].item()
            intervened_mastery = mu_tilde_do[0, -1, c_i].item()
            if intervened_mastery > base_mastery:
                improvements += 1
            total += 1
                
    if total == 0: return 0.0
    return improvements / total
