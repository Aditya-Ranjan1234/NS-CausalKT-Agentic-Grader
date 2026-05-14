import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import KTDataset, download_assist09, preprocess_assist09
from models.dkt import DKT
from models.sakt import SAKT
from eval import evaluate_metrics
import numpy as np
import sys
import os
import argparse
import logging
from tqdm import tqdm

def setup_logger(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    logger = logging.getLogger("DKT_Benchmark")
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
         fh = logging.FileHandler(os.path.join(save_dir, "benchmark_logs.txt"))
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
    parser.add_argument('--model', type=str, default='dkt', choices=['dkt', 'sakt'], help='Model to train')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    logger = setup_logger(args.save_dir)
    logger.info(f"[BENCHMARK] Using device: {args.device}")

    # 1. Dataset Initialization (Identical to NSCausalKT to ensure fairness)
    logger.info("Loading original ASSISTments 2009 dataset...")
    csv_path = download_assist09(data_dir='data')
    df, num_q, num_c, skill_map = preprocess_assist09(csv_path)
    
    users = df['user_id_mapped'].unique()
    # To keep exact split, ideally we use a set seed.
    np.random.seed(42)
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
    
    # 2. Model Initialization (Baseline)
    if args.model == 'dkt':
        model = DKT(num_concepts=num_c, d_model=256, n_layers=1).to(args.device)
    else:
        model = SAKT(num_questions=num_q, num_concepts=num_c, d_model=256).to(args.device)
    
    # 3. Optimizer & Criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Using BCE for DKT standard loss
    
    start_epoch = 0
    checkpoint_path = os.path.join(args.save_dir, f'{args.model}_latest.pt')
    
    logger.info(f"Starting {args.model.upper()} Benchmark Training Loop...")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train {args.model.upper()}]")
        for batch in pbar:
            q = batch['q'].to(args.device)
            a = batch['a'].to(args.device)
            c = batch['c'].to(args.device)
            dt = batch['delta_t'].to(args.device)
            target = batch['target_a'].to(args.device)
            target_c = batch['target_c'].to(args.device)
            
            optimizer.zero_grad()
            
            y_hat, _, _, _ = model(q, a, c, dt, target_c=target_c)
            
            # Predict only on valid (non-padded) targets
            mask = (target != -100).float()
            y_safe = torch.clamp(y_hat, 1e-6, 1.0 - 1e-6)
            loss = - (target * torch.log(y_safe) + (1 - target) * torch.log(1 - y_safe))
            loss = (loss * mask).sum() / (mask.sum() + 1e-8)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        # Validation
        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val {args.model.upper()}]"):
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
        
        logger.info(f"BENCHMARK [{args.model.upper()}] | Epoch {epoch+1} Results: AUC={metrics['auc']:.4f}, RMSE={metrics['rmse']:.4f}, ACC={metrics['acc']:.4f}")
        
        # Save Checkpoint
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'metrics': metrics
        }, checkpoint_path)

if __name__ == "__main__":
    train()
