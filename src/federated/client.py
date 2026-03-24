"""
Federated Learning Client
==========================
Each client:
  1. Loads its own local data partition
  2. Trains the model locally with Opacus Differential Privacy
  3. Sends updated weights to the FL server
  4. Receives aggregated weights back

Run (3 separate terminals after starting server):
  python src/federated/client.py --client-id 0
  python src/federated/client.py --client-id 1
  python src/federated/client.py --client-id 2
"""

import os, sys, argparse, pickle, warnings
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import flwr as fl
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

from src.federated.model import create_model
from src.federated.dataset import ModerationDataset, get_dataloader
from src.config import (
    DATA_PROCESSED_PATH, FL_SERVER_ADDRESS,
    VOCAB_SIZE, BATCH_SIZE, LOCAL_EPOCHS, LEARNING_RATE,
    DP_MAX_GRAD_NORM, DP_NOISE_MULTIPLIER, DP_DELTA
)
from loguru import logger


def load_client_data(client_id: int):
    base = os.path.join(DATA_PROCESSED_PATH, f'client_{client_id}')
    train_df = pickle.load(open(os.path.join(base, 'train.pkl'), 'rb'))
    val_df   = pickle.load(open(os.path.join(base, 'val.pkl'),   'rb'))
    return train_df, val_df


def evaluate_model(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X, y in loader:
            logits = model(X)
            loss = criterion(logits, y)
            preds = logits.argmax(dim=1)

            total_loss += loss.item() * len(y)
            correct += (preds == y).sum().item()
            total += len(y)
            all_preds.extend(preds.tolist())
            all_labels.extend(y.tolist())

    accuracy = correct / total
    avg_loss = total_loss / total

    # Fairness metric: Equal Opportunity Difference
    # (TPR for toxic class vs TPR for safe class)
    tp_toxic = sum(p == 1 and l == 1 for p, l in zip(all_preds, all_labels))
    fn_toxic = sum(p == 0 and l == 1 for p, l in zip(all_preds, all_labels))
    tp_safe  = sum(p == 0 and l == 0 for p, l in zip(all_preds, all_labels))
    fn_safe  = sum(p == 1 and l == 0 for p, l in zip(all_preds, all_labels))

    tpr_toxic = tp_toxic / (tp_toxic + fn_toxic + 1e-8)
    tpr_safe  = tp_safe  / (tp_safe  + fn_safe  + 1e-8)
    eod = abs(tpr_toxic - tpr_safe)  # 0 = perfectly fair

    return avg_loss, accuracy, eod


class ModerationClient(fl.client.NumPyClient):
    def __init__(self, client_id: int, noise_multiplier: float = DP_NOISE_MULTIPLIER):
        self.client_id = client_id
        self.noise_multiplier = noise_multiplier

        # Load data
        train_df, val_df = load_client_data(client_id)
        self.train_loader = get_dataloader(train_df, BATCH_SIZE, shuffle=True)
        self.val_loader   = get_dataloader(val_df,   BATCH_SIZE, shuffle=False)

        # Load vocab size
        vocab_path = os.path.join(DATA_PROCESSED_PATH, 'vocab.pkl')
        vocab = pickle.load(open(vocab_path, 'rb'))
        self.vocab_size = len(vocab)

        # Model + optimizer
        self.model = create_model(self.vocab_size)
        self.model = ModuleValidator.fix(self.model)  # Opacus compatibility fix
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.criterion = nn.CrossEntropyLoss()

        # Attach Opacus Privacy Engine
        self.privacy_engine = PrivacyEngine()
        (
            self.model,
            self.optimizer,
            self.train_loader,
        ) = self.privacy_engine.make_private(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.train_loader,
            noise_multiplier=self.noise_multiplier,
            max_grad_norm=DP_MAX_GRAD_NORM,
        )

        logger.info(f"Client {client_id} ready | "
                    f"train={len(train_df)} | noise={noise_multiplier}")

    def get_parameters(self, config):
        return [p.detach().numpy() for p in self.model.parameters()]

    def set_parameters(self, parameters):
        for p, new_p in zip(self.model.parameters(), parameters):
            p.data = torch.tensor(new_p)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()

        for epoch in range(LOCAL_EPOCHS):
            total_loss, correct, total = 0, 0, 0
            for X, y in self.train_loader:
                self.optimizer.zero_grad()
                logits = self.model(X)
                loss = self.criterion(logits, y)
                loss.backward()
                self.optimizer.step()

                preds = logits.argmax(dim=1)
                total_loss += loss.item() * len(y)
                correct += (preds == y).sum().item()
                total += len(y)

            acc = correct / total
            epsilon = self.privacy_engine.get_epsilon(DP_DELTA)
            logger.info(f"Client {self.client_id} | Epoch {epoch+1} | "
                        f"Loss={total_loss/total:.4f} | Acc={acc:.4f} | ε={epsilon:.2f}")

        epsilon = self.privacy_engine.get_epsilon(DP_DELTA)
        return self.get_parameters(config={}), total, {
            "accuracy": float(acc),
            "epsilon": float(epsilon),
            "client_id": self.client_id,
        }

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy, eod = evaluate_model(self.model, self.val_loader, self.criterion)
        epsilon = self.privacy_engine.get_epsilon(DP_DELTA)

        logger.info(f"Client {self.client_id} EVAL | "
                    f"Acc={accuracy:.4f} | EOD={eod:.4f} | ε={epsilon:.2f}")

        return float(loss), len(self.val_loader.dataset), {
            "accuracy": float(accuracy),
            "eod": float(eod),          # fairness metric
            "epsilon": float(epsilon),  # privacy budget spent
        }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--client-id', type=int, required=True, choices=[0, 1, 2])
    parser.add_argument('--noise-multiplier', type=float, default=DP_NOISE_MULTIPLIER)
    args = parser.parse_args()

    client = ModerationClient(
        client_id=args.client_id,
        noise_multiplier=args.noise_multiplier
    )

    fl.client.start_numpy_client(
        server_address=FL_SERVER_ADDRESS,
        client=client,
    )