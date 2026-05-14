import os
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import download_assist09, preprocess_assist09, KTDataset
from models.nscausalkt import NSCausalKT
from torch.utils.data import DataLoader

def simulate_counterfactual(model, h_t, mu_tilde, W_sym, adj_matrix, edge_index, intervene_c, target_c, device):
    """
    Simulates do(c = 1.0) and observes the change in predicting target_c.
    """
    with torch.no_grad():
        csl = model.csl
        
        # 1. Base prediction (No intervention)
        tc = torch.full((1, 1, 1), target_c, dtype=torch.long, device=device)
        concat_base = torch.cat([h_t[:, -1:, :], torch.gather(mu_tilde[:, -1:, :], 2, tc)], dim=-1)
        dummy_ace = torch.zeros((1, 1, 1), device=device)
        y_original = torch.sigmoid(model.W_p(torch.cat([concat_base, dummy_ace], dim=-1)))
        
        # 2. Intervention: do(intervene_c = 1.0)
        mu_do_1 = mu_tilde[:, -1:, :].clone()
        mu_do_1[..., intervene_c] = 1.0
        
        # Propagate through causal graph
        mu_do_1 = csl.propagate_concept_graph(mu_do_1, edge_index, W_sym, adj_matrix)
        
        # 3. Intervened prediction
        concat_intervened = torch.cat([h_t[:, -1:, :], torch.gather(mu_do_1, 2, tc)], dim=-1)
        y_intervened = torch.sigmoid(model.W_p(torch.cat([concat_intervened, dummy_ace], dim=-1)))
        
        return y_original.item(), y_intervened.item()

def run_inference(args):
    device = torch.device(args.device)
    os.makedirs('visualizations', exist_ok=True)
    
    print("Loading ASSISTments 2009 dataset logic...")
    csv_path = download_assist09('data')
    df, num_q, num_c, skill_map = preprocess_assist09(csv_path)
    inv_skill_map = {v: k for k, v in skill_map.items()}
    
    # Isolate specific user
    users = df['user_id_mapped'].unique()
    target_user = users[args.user_idx % len(users)] # fallback safety
    user_df = df[df['user_id_mapped'] == target_user]
    
    if len(user_df) == 0:
        print(f"User index {args.user_idx} not found. Attempting fallback.")
        return
        
    dataset = KTDataset(df[df['user_id_mapped'] == target_user], max_len=200)
    loader = DataLoader(dataset, batch_size=1)
    batch = next(iter(loader))
    
    # Surrogate edges
    edges = [[i, i+1] for i in range(1, num_c - 1)]
    edge_index = torch.tensor(edges, dtype=torch.long).t().to(device)

    # Initialize Model
    model = NSCausalKT(num_q, num_c, edge_index, d_model=256, n_layers=2).to(device)
    checkpoint_path = 'checkpoints/latest.pt'
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: No trained checkpoint found at {checkpoint_path}. Train the model first!")
        return
        
    print(f"Loading checkpoint {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    q = batch['q'].to(device)
    a = batch['a'].to(device)
    c = batch['c'].to(device)
    dt = batch['delta_t'].to(device)
    target_c = batch['target_c'].to(device)
    seq_len = batch['seq_len'].item()
    
    with torch.no_grad():
        y_hat, y_base, mu_tilde, h_t = model(q, a, c, dt, target_c=target_c)

    print(f"\n--- Root Cause Analysis (RCA) for Student {target_user} ---")
    
    # Extract final mastery state
    final_mastery = mu_tilde[0, seq_len-1, :].cpu().numpy()
    
    # Find weaknesses (lowest masteries > 0 so we avoid unencountered padding)
    encountered_concepts = torch.unique(c[0, :seq_len]).cpu().numpy()
    
    weaknesses = []
    for concept_id in encountered_concepts:
        if concept_id == 0: continue
        weaknesses.append({'id': concept_id, 'skill': inv_skill_map.get(concept_id, "Unknown"), 'mastery': final_mastery[concept_id]})
        
    weaknesses = sorted(weaknesses, key=lambda x: x['mastery'])
    
    print("\n[Weakest Concepts (Root Causes of failure)]")
    for w in weaknesses[:5]:
        print(f"  - {w['skill']} (ID: {w['id']}): {w['mastery']*100:.1f}% mastery")
        
    print("\n[Strongest Concepts]")
    for w in weaknesses[-5:]:
        print(f"  - {w['skill']} (ID: {w['id']}): {w['mastery']*100:.1f}% mastery")

    print(f"\n--- Counterfactual Simulation (do-calculus) ---")
    # Identify a future/target concept to test (e.g. the last question they faced)
    testing_target = c[0, seq_len-1].item()
    if testing_target == 0 and encountered_concepts.size > 1:
        testing_target = encountered_concepts[-1]
    
    # Assume weak concepts are foundational. We intervene on the absolute weakest concept!
    if len(weaknesses) > 0:
        weakest_c = weaknesses[0]['id']
        weakest_skill = weaknesses[0]['skill']
        target_skill = inv_skill_map.get(testing_target, 'Unknown')
        
        # Build surrogate edges identically to train.py
        edges = [[i, i+1] for i in range(1, num_c - 1)]
        edge_index = torch.tensor(edges, dtype=torch.long).t().to(device)
        W_sym = model.scl.W_sym.to(device)
        adj = torch.ones(num_c, num_c, device=device) # rough surrogate
        
        y_orig, y_inter = simulate_counterfactual(model, h_t, mu_tilde, W_sym, adj, edge_index, weakest_c, testing_target, device)
        
        print(f"Scenario: Student attempting '{target_skill}'.")
        print(f"Original Predicted Passing Probability: {y_orig*100:.1f}%")
        print(f"Simulating Intervention --> do({weakest_skill} = 100%)")
        print(f"New Causal Predicted Passing Probability: {y_inter*100:.1f}%")
        
        # Bar Chart
        plt.figure(figsize=(8, 5))
        sns.barplot(x=['Original State', f'Intervened in\n{weakest_skill}'], y=[y_orig, y_inter], palette=['#e74c3c', '#2ecc71'])
        plt.title(f"Causal Intervention Effect on '{target_skill}' Mastery")
        plt.ylabel("Passing Probability")
        plt.ylim(0, 1.0)
        out_bar = f'visualizations/student_{target_user}_counterfactual.png'
        plt.savefig(out_bar, bbox_inches='tight')
        print(f"Saved Counterfactual Bar Chart -> {out_bar}")
    
    # Heatmap Generation
    print("\nGenerating Concept Mastery Timeline Heatmap...")
    plt.figure(figsize=(12, 8))
    
    # Only plot encountered concepts to keep it readable, over time
    history_mu = mu_tilde[0, :seq_len, encountered_concepts].cpu().numpy()
    concept_labels = [inv_skill_map.get(cid, "UNK") for cid in encountered_concepts]
    
    sns.heatmap(history_mu.T, cmap="viridis", yticklabels=concept_labels[:15]) # cap at 15 for visibility
    plt.title(f"Vibrational Concept Mastery over Time (Student {target_user})")
    plt.xlabel("Interaction Timestep")
    plt.ylabel("Assessed Skills")
    out_map = f'visualizations/student_{target_user}_heatmap.png'
    plt.savefig(out_map, bbox_inches='tight')
    print(f"Saved Mastery Heatmap -> {out_map}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--user_idx', type=int, default=1, help='Index of user (0-4150) to diagnose')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    run_inference(args)
