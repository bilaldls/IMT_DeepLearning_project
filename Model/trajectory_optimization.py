"""
Phase 3: Trajectory optimization using the Phase 1 LSTM/GRU model as approximate dynamics.
Applies random shooting with Cross-Entropy Method (CEM) to find a Δv plan minimizing
final-state error plus control effort.
"""
import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np
import torch
from torch import nn


@dataclass
class OptConfig:
    model_path: str = "phase1_model.pth"
    horizon: int = 50
    elite_frac: float = 0.1
    iterations: int = 20
    action_std_init: float = 0.05
    lambda_reg: float = 0.01
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_json: str = "maneuver_plan.json"


class SequenceModelWrapper(nn.Module):
    """Wrap the trained sequence model to allow single-step predictions with Δv injection."""

    def __init__(self, ckpt: Dict[str, object]):
        super().__init__()
        self.window_size = int(ckpt["window_size"])
        self.mean = torch.tensor(ckpt["mean"], dtype=torch.float32)
        self.std = torch.tensor(ckpt["std"], dtype=torch.float32)
        input_size = int(ckpt["input_size"])
        hidden_size = int(ckpt["hidden_size"])
        num_layers = int(ckpt["num_layers"])
        model_type = ckpt["model_type"]
        if model_type == "lstm":
            self.model = LSTMModel(input_size, hidden_size, num_layers, input_size)
        else:
            self.model = GRUModel(input_size, hidden_size, num_layers, input_size)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

    def forward(self, history: torch.Tensor, delta_v: torch.Tensor) -> torch.Tensor:
        """history: (B, L, F), delta_v: (B, 3) applied to vx, vy, vz before prediction."""
        hist_denorm = history * self.std + self.mean
        hist_denorm[:, -1, 3:6] += delta_v  # apply Δv to the latest velocity
        hist_norm = (hist_denorm - self.mean) / self.std
        pred_norm = self.model(hist_norm)
        return pred_norm


class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


class GRUModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        out = out[:, -1, :]
        return self.fc(out)


def simulate_trajectory(
    model: SequenceModelWrapper,
    init_history: np.ndarray,
    actions: np.ndarray,
) -> np.ndarray:
    """Simulate trajectory over horizon using the learned dynamics and actions Δv."""
    device = next(model.parameters()).device
    history = torch.tensor(init_history, dtype=torch.float32, device=device)
    traj = []
    for t in range(actions.shape[0]):
        dv = torch.tensor(actions[t], dtype=torch.float32, device=device).unsqueeze(0)
        pred_norm = model(history.unsqueeze(0), dv)
        pred = pred_norm.detach().cpu().numpy()[0]
        traj.append(pred)
        # update history window
        history_np = history.cpu().numpy()
        history_np = np.concatenate([history_np[1:], pred[None, :]], axis=0)
        history = torch.tensor(history_np, dtype=torch.float32, device=device)
    return np.array(traj)


def cost_function(traj: np.ndarray, target: np.ndarray, actions: np.ndarray, lambda_reg: float) -> float:
    terminal = traj[-1]
    state_error = np.sum((terminal - target) ** 2)
    control_cost = lambda_reg * np.sum(actions**2)
    return float(state_error + control_cost)


def optimize_actions(
    model: SequenceModelWrapper,
    init_history: np.ndarray,
    target_state: np.ndarray,
    cfg: OptConfig,
) -> Tuple[np.ndarray, float]:
    np.random.seed(cfg.seed)
    horizon = cfg.horizon
    action_dim = 3
    mean = np.zeros((horizon, action_dim))
    std = np.full((horizon, action_dim), cfg.action_std_init)

    best_cost = math.inf
    best_actions = None

    for iteration in range(cfg.iterations):
        num_candidates = 200
        actions_samples = np.random.normal(mean, std, size=(num_candidates, horizon, action_dim))
        costs = []
        for k in range(num_candidates):
            traj = simulate_trajectory(model, init_history, actions_samples[k])
            c = cost_function(traj, target_state, actions_samples[k], cfg.lambda_reg)
            costs.append(c)
        costs = np.array(costs)
        elite_idx = costs.argsort()[: max(1, int(cfg.elite_frac * num_candidates))]
        elite_actions = actions_samples[elite_idx]
        mean = elite_actions.mean(axis=0)
        std = elite_actions.std(axis=0) + 1e-4
        if costs.min() < best_cost:
            best_cost = float(costs.min())
            best_actions = actions_samples[costs.argmin()]
        print(f"Iter {iteration+1}/{cfg.iterations} | best_cost={best_cost:.4f}")

    return best_actions, best_cost


def load_checkpoint(path: str, device: str) -> Dict[str, object]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint {path} not found. Train Phase 1 first.")
    ckpt = torch.load(path, map_location=device)
    return ckpt


def main():
    parser = argparse.ArgumentParser(description="Optimize trajectory using learned dynamics")
    parser.add_argument("--model", default="phase1_model.pth", help="Path to Phase 1 model")
    parser.add_argument("--horizon", type=int, default=50, help="Planning horizon")
    parser.add_argument("--lambda_reg", type=float, default=0.01, help="Δv regularization")
    parser.add_argument("--iterations", type=int, default=20, help="CEM iterations")
    parser.add_argument("--elite_frac", type=float, default=0.1, help="Elite fraction")
    parser.add_argument("--action_std", type=float, default=0.05, help="Initial std for Δv")
    parser.add_argument("--out", default="maneuver_plan.json", help="Output plan path")
    args = parser.parse_args()

    cfg = OptConfig(
        model_path=args.model,
        horizon=args.horizon,
        lambda_reg=args.lambda_reg,
        iterations=args.iterations,
        elite_frac=args.elite_frac,
        action_std_init=args.action_std,
        output_json=args.out,
    )

    ckpt = load_checkpoint(cfg.model_path, cfg.device)
    model = SequenceModelWrapper(ckpt).to(cfg.device)

    # Build initial history from normalization stats (small perturbation around mean)
    L = ckpt["window_size"]
    feature_cols = ckpt.get("feature_cols", None)
    mean = ckpt["mean"]
    std = ckpt["std"]
    init_history = np.tile(mean, (L, 1)) + np.random.normal(scale=0.01, size=(L, len(mean)))
    init_history_norm = (init_history - mean) / std

    target_state = mean + 0.5 * std  # arbitrary reachable target for demo

    best_actions, best_cost = optimize_actions(model, init_history_norm, target_state, cfg)
    traj = simulate_trajectory(model, init_history_norm, best_actions)
    final_state = traj[-1]

    plan = {
        "best_cost": best_cost,
        "target_state": target_state.tolist(),
        "final_state": final_state.tolist(),
        "delta_v_sequence": best_actions.tolist(),
        "horizon": cfg.horizon,
    }
    with open(cfg.output_json, "w") as f:
        json.dump(plan, f, indent=2)
    print(f"Saved maneuver plan to {cfg.output_json}")


if __name__ == "__main__":
    main()