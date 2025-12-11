# Evaluation

This directory contains resources for evaluating the trained Billiards AI agent.

## Setup
Ensure the trained model checkpoint is present at `eval/checkpoints/cqr.joblib`.
If not, please run the training pipeline (see `train/README.md`).

## Agent Implementation
The `NewAgent` in `agent.py` implements the CQR strategy:
1. **Load Model**: Loads the `SGDRegressor` and `StandardScaler` from the checkpoint.
2. **Decision Making**:
   - Generates geometric candidates (Ghost Ball aiming).
   - Extracts features for each candidate.
   - Predicts scores using the trained model.
   - Selects Top-K (e.g., 24) candidates.
   - Simulates the Top-K candidates using `pooltool` physics engine.
   - Chooses the action with the highest simulated reward.

## Running Evaluation
To test the agent against the baseline `BasicAgent` (or another opponent), run:

```bash
python evaluate.py
```

You can modify `evaluate.py` to change the number of games or opponent settings.

## Performance
The agent is expected to achieve >60% win rate against the `BasicAgent` by effectively pruning bad shots and optimizing shot selection using the learned value function.
