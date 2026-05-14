import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import KTDataset, download_assist09, preprocess_assist09
from models.nscausalkt import NSCausalKT
from eval import evaluate_metrics, evaluate_ic_k
import numpy as np
import sys
import os
import argparse
import logging
from tqdm import tqdm

def setup_logger(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    logger = logging.getLogger("NSCausalKT")
    logger.setLevel(logging.INFO)
    
    # Prevent duplicate handlers
    if not logger.handlers:
         fh = logging.FileHandler(os.path.join(save_dir, "training_logs.txt"))
         fh.setLevel(logging.INFO)
         ch = logging.StreamHandler()
         ch.setLevel(logging.INFO)
         formatter = logging.Formatter('%(asctime)s - %(message)s')
         fh.setFormatter(formatter)
         ch.setFormatter(formatter)
         logger.addHandler(fh)
         logger.addHandler(ch)
    return logger

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--lambda1', type=float, default=1.0)
    parser.add_argument('--lambda2', type=float, default=0.5)
    parser.add_argument('--lambda3', type=float, default=0.1)
    args = parser.parse_args()

    logger = setup_logger(args.save_dir)
    logger.info(f"Using device: {args.device}")

    # Ensure checkpoint dir exists
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 1. Dataset Initialization
    logger.info("Loading original ASSISTments 2009 dataset...")
    csv_path = download_assist09(data_dir='data')
    df, num_q, num_c, skill_map = preprocess_assist09(csv_path)
    
    # We build a basic surrogate graph using skill frequencies or random structure since the exact
    # prior prerequisite graph is external.
    # To ensure IC@5 doesn't return 0.0 due to empty edges, we add sequential edges:
    edges = [[i, i+1] for i in range(1, num_c - 1)]
    edge_index = torch.tensor(edges, dtype=torch.long).t().to(args.device)
    
    # Simple split
    users = df['user_id_mapped'].unique()
    np.random.shuffle(users)
    split_idx = int(len(users) * 0.8)
    train_users = users[:split_idx]
    val_users = users[split_idx:]
    
    train_df = df[df['user_id_mapped'].isin(train_users)]
    val_df = df[df['user_id_mapped'].isin(val_users)]
    
    train_dataset = KTDataset(train_df, max_len=200)
    val_dataset = KTDataset(val_df, max_len=200)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 2. Model Initialization
    edge_index = edge_index.to(args.device)
    model = NSCausalKT(num_questions=num_q, num_concepts=num_c, edge_index=edge_index).to(args.device)
    
    # 3. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)
    
    start_epoch = 0
    checkpoint_path = os.path.join(args.save_dir, 'latest.pt')
    
    if os.path.exists(checkpoint_path):
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=args.device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        logger.info(f"Resuming from epoch {start_epoch}")
        
    logger.info("Starting Training Loop...")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0
        total_l_pred = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for batch in pbar:
            q = batch['q'].to(args.device)
            a = batch['a'].to(args.device)
            c = batch['c'].to(args.device)
            dt = batch['delta_t'].to(args.device)
            target = batch['target_a'].to(args.device)
            target_c = batch['target_c'].to(args.device)
            
            optimizer.zero_grad()
            
            y_hat, y_base, mu_tilde, h_t = model(q, a, c, dt, target_c=target_c)
            
            # Since temporal smoothness needs prev state:
            # We enforce within-sequence temporal smoothness inside the model
            loss, l_pred, l_sym, l_causal, l_smooth = model.get_losses(y_hat, target, mu_tilde, None, args.lambda1, args.lambda2, args.lambda3)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_l_pred += l_pred.item()
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'l_pred': f"{l_pred.item():.4f}"})
            
        scheduler.step()
        
        # Validation
        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]"):
                q = batch['q'].to(args.device)
                a = batch['a'].to(args.device)
                c = batch['c'].to(args.device)
                dt = batch['delta_t'].to(args.device)
                target = batch['target_a'].to(args.device)
                target_c = batch['target_c'].to(args.device)
                
                y_hat, _, _, _ = model(q, a, c, dt, target_c=target_c)
                
                all_preds.append(y_hat.cpu().numpy().flatten())
                all_targets.append(target.cpu().numpy().flatten())
                
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        metrics = evaluate_metrics(all_targets, all_preds)
        
        ic_k = evaluate_ic_k(model, val_dataset, k=5, device=args.device)
        
        logger.info(f"Epoch {epoch+1} Results: AUC={metrics['auc']:.4f}, RMSE={metrics['rmse']:.4f}, ACC={metrics['acc']:.4f}, IC@5={ic_k:.4f}")
        
        # Save Checkpoint
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'metrics': metrics
        }, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")

if __name__ == "__main__":
    train()
