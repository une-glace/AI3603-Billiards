# Training Pipeline

This directory contains scripts for training the Contextual Q Regression (CQR) model for the Billiards AI.

## Requirements
- Python 3.10+
- `pooltool`
- `numpy`
- `scikit-learn`
- `joblib`
- `tqdm`

## Files
- `collect_dataset.py`: Runs games using the `PoolEnv` and collects geometric candidate features and their simulation rewards.
- `train_cqr.py`: Trains a `SGDRegressor` on the collected dataset to predict the expected reward of a shot candidate.

## Usage

1. **Collect Data**
   Run the collection script to generate a training dataset.
   ```bash
   python train/collect_dataset.py --games 200 --samples 30 --out train/dataset.npz
   ```
   Parameters:
   - `--games`: Number of games to simulate (default: 50). Recommendation: 100-200.
   - `--samples`: Max candidates to sample per observation (default: 30).
   - `--out`: Output file path.

2. **Train Model**
   Train the regression model using the collected data.
   ```bash
   python train/train_cqr.py --data train/dataset.npz --out eval/checkpoints/cqr.joblib
   ```
   Parameters:
   - `--data`: Path to the input dataset.
   - `--out`: Path to save the trained model checkpoint.

## Methodology
The approach uses a **Contextual Q Regression** strategy:
1. **Candidate Generation**: For each state, we generate "Ghost Ball" aiming candidates for all legal target balls and pockets.
2. **Feature Extraction**: We compute geometric features (cut angle, distance, clearance, velocity).
3. **Reward Simulation**: We simulate these candidates (offline) to get the ground truth reward using the game's reward function.
4. **Regression**: We train a model to predict `Reward = f(Features)`.

This allows the agent to quickly score hundreds of candidates at runtime and only simulate the top few.
