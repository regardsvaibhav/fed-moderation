"""
Research Experiment: Privacy-Utility-Fairness Tradeoff
========================================================
TextCNN v2 — wider filters, class weights, more epochs.
Expected: ~80-84% centralized, ~76-80% FL+DP.
Run: python src/federated/experiment.py
"""

import os, sys, pickle, json, warnings
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger

from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

from src.federated.model import create_model
from src.federated.dataset import get_dataloader
from src.config import DATA_PROCESSED_PATH, NUM_CLIENTS, DP_MAX_GRAD_NORM, DP_DELTA

BATCH_SIZE = 64
LR = 5e-4
NUM_EPOCHS_CENTRAL = 10
NUM_ROUNDS = 5
LOCAL_EPOCHS = 3

class mlflow:
    @staticmethod
    def set_tracking_uri(u): pass
    @staticmethod
    def set_experiment(n): pass
    @staticmethod
    def start_run(run_name=""):
        import contextlib; return contextlib.nullcontext()
    @staticmethod
    def log_params(d): pass
    @staticmethod
    def log_metrics(d, step=None): pass
    @staticmethod
    def log_artifact(p): pass


def load_test_data():
    return pickle.load(open(os.path.join(DATA_PROCESSED_PATH, 'test.pkl'), 'rb'))

def load_all_train_data():
    dfs = []
    for cid in range(NUM_CLIENTS):
        base = os.path.join(DATA_PROCESSED_PATH, f'client_{cid}')
        dfs.append(pickle.load(open(os.path.join(base, 'train.pkl'), 'rb')))
    return pd.concat(dfs, ignore_index=True)

def load_client_data(cid):
    base = os.path.join(DATA_PROCESSED_PATH, f'client_{cid}')
    return (
        pickle.load(open(os.path.join(base, 'train.pkl'), 'rb')),
        pickle.load(open(os.path.join(base, 'val.pkl'), 'rb')),
    )

def load_vocab():
    return pickle.load(open(os.path.join(DATA_PROCESSED_PATH, 'vocab.pkl'), 'rb'))


def evaluate(model, loader, criterion):
    model.eval()
    loss_sum, correct, total = 0, 0, 0
    preds_all, labels_all = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch['input_ids']
            y = batch['labels']
            out = model(x)
            loss_sum += criterion(out, y).item() * len(y)
            p = out.argmax(1)
            correct += (p == y).sum().item()
            total += len(y)
            preds_all.extend(p.tolist())
            labels_all.extend(y.tolist())

    acc = correct / total
    tp1 = sum(p==1 and l==1 for p,l in zip(preds_all, labels_all))
    fn1 = sum(p==0 and l==1 for p,l in zip(preds_all, labels_all))
    tp0 = sum(p==0 and l==0 for p,l in zip(preds_all, labels_all))
    fn0 = sum(p==1 and l==0 for p,l in zip(preds_all, labels_all))
    eod = abs(tp1/(tp1+fn1+1e-8) - tp0/(tp0+fn0+1e-8))
    return loss_sum/total, acc, eod


def run_centralized(vocab_size, test_loader, criterion):
    logger.info("="*55)
    logger.info("EXPERIMENT 1: Centralized Baseline")
    logger.info("="*55)
    train_df = load_all_train_data()
    train_loader = get_dataloader(train_df, BATCH_SIZE, shuffle=True)
    model = create_model(vocab_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    for epoch in range(NUM_EPOCHS_CENTRAL):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            criterion(model(batch['input_ids']), batch['labels']).backward()
            optimizer.step()
        scheduler.step()
        _, acc, eod = evaluate(model, test_loader, criterion)
        logger.info(f"Epoch {epoch+1}/{NUM_EPOCHS_CENTRAL} | Acc={acc:.4f} | EOD={eod:.4f}")

    _, final_acc, final_eod = evaluate(model, test_loader, criterion)
    logger.success(f"Centralized FINAL | Acc={final_acc:.4f} | EOD={final_eod:.4f}")
    torch.save(model.state_dict(), os.path.join(DATA_PROCESSED_PATH, 'centralized_model.pt'))
    return {"setting": "Centralized", "accuracy": final_acc,
            "fairness_eod": final_eod, "privacy_epsilon": None}


def run_federated_simulation(noise_multiplier, vocab_size, test_loader, criterion):
    logger.info(f"FL + DP | noise={noise_multiplier}")
    global_model = create_model(vocab_size)
    global_weights = [p.data.clone() for p in global_model.parameters()]
    epsilon_final = 0.0

    for fl_round in range(NUM_ROUNDS):
        client_weights, client_sizes, epsilons = [], [], []

        for cid in range(NUM_CLIENTS):
            train_df, _ = load_client_data(cid)
            if len(train_df) < BATCH_SIZE:
                continue

            model = create_model(vocab_size)
            model = ModuleValidator.fix(model)
            for p, gw in zip(model.parameters(), global_weights):
                p.data = gw.clone()

            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
            train_loader = get_dataloader(train_df, BATCH_SIZE, shuffle=True)

            privacy_engine = PrivacyEngine()
            model, optimizer, train_loader = privacy_engine.make_private(
                module=model, optimizer=optimizer, data_loader=train_loader,
                noise_multiplier=noise_multiplier, max_grad_norm=DP_MAX_GRAD_NORM,
            )

            model.train()
            for _ in range(LOCAL_EPOCHS):
                for batch in train_loader:
                    optimizer.zero_grad()
                    criterion(model(batch['input_ids']), batch['labels']).backward()
                    optimizer.step()

            epsilon = privacy_engine.get_epsilon(DP_DELTA)
            epsilons.append(epsilon)
            client_weights.append([p.data.clone() for p in model.parameters()])
            client_sizes.append(len(train_df))

        total = sum(client_sizes)
        global_weights = [
            sum(client_weights[i][j] * (client_sizes[i]/total)
                for i in range(len(client_weights)))
            for j in range(len(global_weights))
        ]
        for p, gw in zip(global_model.parameters(), global_weights):
            p.data = gw.clone()

        epsilon_final = float(np.mean(epsilons))
        _, acc, eod = evaluate(global_model, test_loader, criterion)
        logger.info(f"  Round {fl_round+1}/{NUM_ROUNDS} | Acc={acc:.4f} | EOD={eod:.4f} | ε={epsilon_final:.2f}")

    _, final_acc, final_eod = evaluate(global_model, test_loader, criterion)
    logger.success(f"  FINAL | Acc={final_acc:.4f} | EOD={final_eod:.4f} | ε={epsilon_final:.2f}")

    if noise_multiplier == 1.1:
        torch.save(global_model.state_dict(),
                   os.path.join(DATA_PROCESSED_PATH, 'federated_model.pt'))

    return {"setting": f"FL+DP (η={noise_multiplier})",
            "noise_multiplier": noise_multiplier,
            "accuracy": final_acc, "fairness_eod": final_eod,
            "privacy_epsilon": epsilon_final}


def run_all_experiments():
    logger.info("Loading data...")
    vocab = load_vocab()
    vocab_size = len(vocab)
    test_df = load_test_data()
    test_loader = get_dataloader(test_df, BATCH_SIZE, shuffle=False)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0]))

    all_results = []

    with mlflow.start_run(run_name="centralized"):
        all_results.append(run_centralized(vocab_size, test_loader, criterion))

    for noise in [1.1]:
        logger.info("="*55)
        logger.info(f"EXPERIMENT 2: FL+DP noise={noise}")
        logger.info("="*55)
        with mlflow.start_run(run_name=f"fl_dp_{noise}"):
            mlflow.log_params({"noise_multiplier": noise})
            all_results.append(
                run_federated_simulation(noise, vocab_size, test_loader, criterion)
            )

    logger.info("\n" + "="*65)
    logger.info("RESEARCH RESULTS TABLE")
    logger.info("="*65)
    logger.info(f"{'Setting':<22} {'Accuracy':>10} {'EOD (↓)':>10} {'Epsilon (↓)':>12}")
    logger.info("-"*65)
    for r in all_results:
        eps = f"{r['privacy_epsilon']:.2f}" if r['privacy_epsilon'] else "∞ (none)"
        logger.info(f"{r['setting']:<22} {r['accuracy']*100:>9.2f}% "
                    f"{r['fairness_eod']:>10.4f} {eps:>12}")
    logger.info("="*65)

    serializable = [dict(r) for r in all_results]
    with open(os.path.join(DATA_PROCESSED_PATH, 'experiment_results.json'), 'w') as f:
        json.dump(serializable, f, indent=2)
    logger.success("Done! Results saved.")
    return all_results


if __name__ == '__main__':
    run_all_experiments()