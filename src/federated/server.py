"""
Federated Learning Server
==========================
- FedAvg aggregation strategy
- Collects accuracy, fairness (EOD), and privacy (epsilon) per round
- Logs all metrics to MLflow
- Saves final aggregated model weights

Run FIRST before starting clients:
  python src/federated/server.py
"""

import os, sys, pickle, json
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from typing import List, Tuple, Optional, Dict, Union
import numpy as np
import torch
import flwr as fl
from flwr.common import Metrics, Parameters, Scalar
from flwr.server.strategy import FedAvg

import mlflow
from loguru import logger

from src.federated.model import create_model
from src.config import (
    FL_SERVER_ADDRESS, NUM_ROUNDS, NUM_CLIENTS,
    DATA_PROCESSED_PATH, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT
)


# ── Metrics Aggregation ─────────────────────────────────────────────────────
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate metrics from all clients using weighted average."""
    total_samples = sum(n for n, _ in metrics)

    aggregated = {}
    for key in ['accuracy', 'eod', 'epsilon']:
        if all(key in m for _, m in metrics):
            aggregated[key] = sum(n * m[key] for n, m in metrics) / total_samples

    return aggregated


# ── Custom FedAvg Strategy with Logging ────────────────────────────────────
class LoggingFedAvg(FedAvg):
    def __init__(self, vocab_size: int, run_name: str = "federated", **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.run_name = run_name
        self.round_results = []

    def aggregate_evaluate(
        self,
        server_round: int,
        results,
        failures,
    ):
        """After each round, log aggregated metrics."""
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(
            server_round, results, failures
        )

        if aggregated_metrics:
            acc = aggregated_metrics.get('accuracy', 0)
            eod = aggregated_metrics.get('eod', 0)
            eps = aggregated_metrics.get('epsilon', 0)

            logger.info(
                f"Round {server_round:2d} | "
                f"Acc={acc:.4f} | EOD={eod:.4f} | ε={eps:.2f}"
            )

            # Log to MLflow
            mlflow.log_metrics({
                f"accuracy": acc,
                f"fairness_eod": eod,
                f"privacy_epsilon": eps,
            }, step=server_round)

            self.round_results.append({
                "round": server_round,
                "accuracy": acc,
                "fairness_eod": eod,
                "privacy_epsilon": eps,
            })

        return aggregated_loss, aggregated_metrics

    def aggregate_fit(self, server_round, results, failures):
        """After fitting, save model weights."""
        aggregated = super().aggregate_fit(server_round, results, failures)

        if aggregated and server_round == NUM_ROUNDS:
            # Save final model
            weights = fl.common.parameters_to_ndarrays(aggregated[0])
            model = create_model(self.vocab_size)
            for p, w in zip(model.parameters(), weights):
                p.data = torch.tensor(w)

            save_path = os.path.join(DATA_PROCESSED_PATH, 'federated_model.pt')
            torch.save(model.state_dict(), save_path)
            logger.success(f"Final federated model saved → {save_path}")
            mlflow.log_artifact(save_path)

        return aggregated


# ── Main Server ─────────────────────────────────────────────────────────────
def run_server(noise_multiplier: float = 1.1, run_name: str = "federated_baseline"):
    # Load vocab size
    vocab_path = os.path.join(DATA_PROCESSED_PATH, 'vocab.pkl')
    vocab = pickle.load(open(vocab_path, 'rb'))
    vocab_size = len(vocab)

    # Setup MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "num_rounds": NUM_ROUNDS,
            "num_clients": NUM_CLIENTS,
            "noise_multiplier": noise_multiplier,
            "strategy": "FedAvg",
            "local_epochs": 2,
        })

        strategy = LoggingFedAvg(
            vocab_size=vocab_size,
            run_name=run_name,
            fraction_fit=1.0,           # use all clients every round
            fraction_evaluate=1.0,
            min_fit_clients=NUM_CLIENTS,
            min_evaluate_clients=NUM_CLIENTS,
            min_available_clients=NUM_CLIENTS,
            evaluate_metrics_aggregation_fn=weighted_average,
        )

        logger.info(f"Starting FL Server | {NUM_CLIENTS} clients | {NUM_ROUNDS} rounds")
        logger.info(f"Waiting for clients at {FL_SERVER_ADDRESS}...")

        fl.server.start_server(
            server_address=FL_SERVER_ADDRESS,
            config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
            strategy=strategy,
        )

        # Save round results
        results_path = os.path.join(DATA_PROCESSED_PATH, f'results_{run_name}.json')
        with open(results_path, 'w') as f:
            json.dump(strategy.round_results, f, indent=2)

        logger.success(f"Training complete! Results → {results_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise-multiplier', type=float, default=1.1)
    parser.add_argument('--run-name', type=str, default='federated_baseline')
    args = parser.parse_args()

    run_server(
        noise_multiplier=args.noise_multiplier,
        run_name=args.run_name
    )