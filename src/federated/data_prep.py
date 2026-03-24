"""
Data Preparation for Federated Content Moderation
Dataset: HateXplain (Mathew et al., 2021)
"""
import os, sys, json, re, pickle
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from src.config import (DATA_RAW_PATH, DATA_PROCESSED_PATH, NUM_CLIENTS, VOCAB_SIZE, MAX_SEQ_LEN)
from loguru import logger

STOPWORDS = set(['i','me','my','we','our','you','your','he','him','his','she','her','it','its',
'they','them','their','what','which','who','this','that','these','those','am','is','are','was',
'were','be','been','being','have','has','had','do','does','did','a','an','the','and','but','if',
'or','as','of','at','by','for','with','about','into','through','to','from','up','in','out','on',
'off','over','then','here','there','when','where','how','all','both','each','more','most','other',
'some','no','nor','not','only','same','so','than','too','can','will','just','should','now'])

def load_hatexplain(path):
    logger.info(f"Loading HateXplain from {path}")
    with open(path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    rows = []
    for post_id, post in raw.items():
        text = ' '.join(post.get('post_tokens', []))
        label_votes = [a['label'] for a in post.get('annotators', [])]
        majority_label = Counter(label_votes).most_common(1)[0][0]
        binary_label = 0 if majority_label == 'normal' else 1
        all_targets = []
        for a in post.get('annotators', []):
            all_targets.extend(a.get('target', []))
        target_counts = Counter([t for t in all_targets if t != 'None'])
        primary_target = target_counts.most_common(1)[0][0] if target_counts else 'None'
        rows.append({'post_id': post_id, 'text': text, 'label': binary_label,
                     'raw_label': majority_label, 'target_group': primary_target})
    df = pd.DataFrame(rows)
    logger.info(f"Loaded {len(df)} posts | labels: {df['label'].value_counts().to_dict()}")
    logger.info(f"Target groups: {df['target_group'].value_counts().head(6).to_dict()}")
    return df

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def tokenize(text):
    return [t for t in text.lower().split() if t not in STOPWORDS and len(t) > 1]

def build_vocab(texts, max_size=VOCAB_SIZE):
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, _ in counter.most_common(max_size - 2):
        vocab[word] = len(vocab)
    logger.info(f"Vocabulary size: {len(vocab)}")
    return vocab

def text_to_indices(text, vocab, max_len=MAX_SEQ_LEN):
    tokens = tokenize(text)
    indices = [vocab.get(t, vocab['<UNK>']) for t in tokens]
    if len(indices) < max_len:
        indices += [vocab['<PAD>']] * (max_len - len(indices))
    return indices[:max_len]

def dirichlet_split(df, num_clients, alpha=0.5):
    np.random.seed(42)
    labels = df['label'].values
    client_indices = [[] for _ in range(num_clients)]
    for cls in range(len(np.unique(labels))):
        cls_indices = np.where(labels == cls)[0]
        np.random.shuffle(cls_indices)
        proportions = np.random.dirichlet(alpha=np.repeat(alpha, num_clients))
        proportions = (proportions * len(cls_indices)).astype(int)
        proportions[-1] = len(cls_indices) - proportions[:-1].sum()
        start = 0
        for cid, count in enumerate(proportions):
            client_indices[cid].extend(cls_indices[start:start + count].tolist())
            start += count
    client_dfs = []
    for cid, indices in enumerate(client_indices):
        np.random.shuffle(indices)
        cdf = df.iloc[indices].reset_index(drop=True)
        client_dfs.append(cdf)
        logger.info(f"Client {cid}: {len(cdf)} samples | toxic={round(cdf['label'].mean()*100,1)}%")
    return client_dfs

def prepare_data():
    os.makedirs(DATA_PROCESSED_PATH, exist_ok=True)
    df = load_hatexplain(os.path.join(DATA_RAW_PATH, 'dataset.json'))
    df['clean_text'] = df['text'].apply(clean_text)
    df = df[df['clean_text'].str.len() > 5].reset_index(drop=True)
    train_df, test_df = train_test_split(df, test_size=0.15, random_state=42, stratify=df['label'])
    train_df = train_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)
    logger.info(f"Train: {len(train_df)} | Test: {len(test_df)}")
    vocab = build_vocab(train_df['clean_text'].tolist())
    train_df['encoded'] = train_df['clean_text'].apply(lambda x: text_to_indices(x, vocab))
    test_df['encoded']  = test_df['clean_text'].apply(lambda x: text_to_indices(x, vocab))
    client_dfs = dirichlet_split(train_df, NUM_CLIENTS, alpha=0.5)
    client_stats = {}
    for cid, cdf in enumerate(client_dfs):
        ctrain, cval = train_test_split(cdf, test_size=0.2, random_state=42)
        save_dir = os.path.join(DATA_PROCESSED_PATH, f'client_{cid}')
        os.makedirs(save_dir, exist_ok=True)
        ctrain.to_pickle(os.path.join(save_dir, 'train.pkl'))
        cval.to_pickle(os.path.join(save_dir, 'val.pkl'))
        client_stats[f'client_{cid}'] = {
            'train_samples': len(ctrain), 'val_samples': len(cval),
            'toxic_ratio': round(ctrain['label'].mean(), 3),
            'top_targets': ctrain['target_group'].value_counts().head(3).to_dict()
        }
        logger.success(f"Client {cid}: {len(ctrain)} train / {len(cval)} val")
    test_df.to_pickle(os.path.join(DATA_PROCESSED_PATH, 'test.pkl'))
    with open(os.path.join(DATA_PROCESSED_PATH, 'vocab.pkl'), 'wb') as f:
        pickle.dump(vocab, f)
    stats = {'dataset': 'HateXplain (Mathew et al., 2021)', 'total': len(df),
             'train': len(train_df), 'test': len(test_df), 'vocab_size': len(vocab),
             'label_dist': df['label'].value_counts().to_dict(),
             'target_groups': df['target_group'].value_counts().head(10).to_dict(),
             'clients': client_stats}
    with open(os.path.join(DATA_PROCESSED_PATH, 'dataset_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    logger.success("=" * 50)
    logger.success(f"Dataset  : HateXplain (Mathew et al., 2021)")
    logger.success(f"Total    : {len(df)} real social media posts")
    logger.success(f"Vocab    : {len(vocab)} tokens")
    logger.success(f"Train    : {len(train_df)} | Test: {len(test_df)}")
    logger.success(f"Clients  : {NUM_CLIENTS} non-IID partitions")
    logger.success("=" * 50)
    return stats

if __name__ == '__main__':
    prepare_data()