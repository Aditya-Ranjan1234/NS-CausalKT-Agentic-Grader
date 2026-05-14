import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import os
import urllib.request
import zipfile

def download_assist09(data_dir='data'):
    """
    Downloads the original ASSISTments 2009 skill builder dataset.
    """
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, 'skill_builder_data.csv')
    
    if os.path.exists(csv_path):
        print(f"Data already exists at {csv_path}")
        return csv_path
        
    print("Downloading original ASSISTments 2009 dataset...")
    # This is a widely used mirror for the raw, original dataset:
    url = "https://raw.githubusercontent.com/bighuang624/DKT/master/data/assist2009_updated/assist2009_updated.csv"
    
    # Alternatively, the dataset is often distributed as a zip. We will try a known mirror.
    # Since raw github links might 404, we provide instructions if download fails.
    try:
        # We will use the deepkt mirror which is stable
        stable_url = "https://raw.githubusercontent.com/ckyeungac/DeepKT/master/data/assistments09/skill_builder_data_corrected.csv"
        urllib.request.urlretrieve(stable_url, csv_path)
        print("Download complete.")
    except Exception as e:
        print(f"Failed to download automatically: {e}")
        print("Please manually download 'skill_builder_data.csv' from the official ASSISTments 2009-2010 site:")
        print("https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data")
        print(f"And place it at: {os.path.abspath(csv_path)}")
        raise FileNotFoundError("Missing original dataset.")
        
    return csv_path

def preprocess_assist09(csv_path):
    print("Preprocessing real ASSISTments 09 logs...")
    df = pd.read_csv(csv_path, encoding='latin1')
    
    # Filter rows with missing skills
    df = df.dropna(subset=['skill_name'])
    
    # Create mappings
    users = df['user_id'].unique()
    skills = df['skill_name'].unique()
    questions = df['problem_id'].unique()
    
    user_map = {u: i for i, u in enumerate(users)}
    skill_map = {s: i+1 for i, s in enumerate(skills)} # 0 is padding
    q_map = {q: i+1 for i, q in enumerate(questions)}
    
    df['user_id_mapped'] = df['user_id'].map(user_map)
    df['concept_id'] = df['skill_name'].map(skill_map)
    df['question_id'] = df['problem_id'].map(q_map)
    
    # Parse time logic if needed (original dataset doesn't have reliable timestamp, order_id is used)
    if 'ms_first_response' in df.columns:
        df['time_delta'] = df['ms_first_response'].fillna(0)
    else:
        df['time_delta'] = 0
        
    num_q = len(questions)
    num_c = len(skills) + 1
    
    return df, num_q, num_c, skill_map

class KTDataset(Dataset):
    def __init__(self, df, max_len=200):
        super(KTDataset, self).__init__()
        self.max_len = max_len
        
        # Expects df with user_id_mapped, question_id, concept_id, correct, time_delta
        self.users = df['user_id_mapped'].unique()
        self.user_groups = df.groupby('user_id_mapped')
        
    def __len__(self):
        return len(self.users)
        
    def __getitem__(self, idx):
        u = self.users[idx]
        group = self.user_groups.get_group(u)
        
        # sort by interaction sequence if ordered
        if 'order_id' in group.columns:
             group = group.sort_values('order_id')
             
        q = group['question_id'].values
        c = group['concept_id'].values
        a = group['correct'].values
        if 'time_delta' in group.columns:
            delta = np.log1p(np.clip(group['time_delta'].values, 0, None))
        else:
            delta = np.zeros_like(q, dtype=float)
            
        # Truncate
        seq_len = len(q)
        if seq_len > self.max_len:
            q = q[-self.max_len:]
            c = c[-self.max_len:]
            a = a[-self.max_len:]
            delta = delta[-self.max_len:]
            seq_len = self.max_len
            
        q_pad = np.zeros(self.max_len, dtype=int)
        c_pad = np.zeros(self.max_len, dtype=int)
        a_pad = np.zeros(self.max_len, dtype=int) 
        delta_pad = np.zeros(self.max_len, dtype=float)
        
        a_shifted = a + 1
        
        q_pad[:seq_len] = q
        c_pad[:seq_len] = c
        a_pad[:seq_len] = a_shifted
        delta_pad[:seq_len] = delta
        
        target_a = np.full(self.max_len, -100, dtype=float)
        target_c = np.full(self.max_len, 0, dtype=int)
        
        # Shift the target: predict a[t+1] using state at t
        if seq_len > 1:
            target_a[:seq_len-1] = a[1:]
            target_c[:seq_len-1] = c[1:]
        
        return {
            'q': torch.tensor(q_pad, dtype=torch.long),
            'c': torch.tensor(c_pad, dtype=torch.long),
            'a': torch.tensor(a_pad, dtype=torch.long),
            'delta_t': torch.tensor(delta_pad, dtype=torch.float).unsqueeze(-1),
            'target_a': torch.tensor(target_a, dtype=torch.float).unsqueeze(-1),
            'target_c': torch.tensor(target_c, dtype=torch.long).unsqueeze(-1),
            'seq_len': torch.tensor(seq_len, dtype=torch.long)
        }
