# AI3603 Billiards Agent Project Report

## 1. Methodological Approach

### 1.1 Overview
We adopted a **Contextual Q Regression (CQR)** approach combined with **Ghost Ball Geometric Search** and **Monte Carlo Refinement**. This method decouples the problem into two stages:
1.  **Candidate Generation**: Using geometric priors (Ghost Ball aiming) to generate high-potential shot candidates.
2.  **Scoring & Refinement**: Using a learned regression model to estimate the quality (expected reward) of each candidate, followed by physical simulation of the top candidates to select the best one.

This approach was chosen over pure Reinforcement Learning (e.g., PPO) because:
-   **Sample Efficiency**: PPO requires millions of interactions. CQR learns from offline datasets efficiently.
-   **Stability**: Geometric priors ensure the agent always considers physically valid shots, avoiding the "cold start" problem of RL.
-   **Robustness**: By simulating the top candidates, we verify the model's predictions, preventing hallucinations.

### 1.2 Algorithm Details
-   **State Space**: The raw observation includes ball positions. We transform this into geometric features for each candidate shot:
    -   Cut Angle (cosine)
    -   Cue-to-Ghost Distance
    -   Target-to-Pocket Distance
    -   Clearance (binary check for path obstruction)
    -   Velocity (normalized)
    -   Spin (Top/Bottom)
-   **Action Space**: Continuous parameters $\{V_0, \phi, \theta, a, b\}$. We discretize $V_0$ and spin $(a, b)$ into a grid of candidates, while $\phi$ is analytically calculated.
-   **Model**: A **Histogram-based Gradient Boosting Regressor** (`HistGradientBoostingRegressor`) is trained to predict the immediate reward of a shot given its features.

## 2. Implementation Details

### 2.1 Training Pipeline
1.  **Data Collection** (`train/collect_dataset.py`):
    -   We simulate games using `PoolEnv`.
    -   For each state, we generate ~50 geometric candidates (varying velocity and spin).
    -   Each candidate is simulated using `pooltool` to obtain the ground truth reward (using `analyze_shot_for_reward`).
    -   We collected approximately **16,000 samples** with noise enabled to ensure robustness.
2.  **Model Training** (`train/train_cqr.py`):
    -   Features are normalized using `StandardScaler`.
    -   The `HistGradientBoostingRegressor` is trained to minimize Squared Error.
    -   Achieved a Test $R^2$ of **0.75**, indicating strong predictive capability.

### 2.2 Agent Architecture (`NewAgent`)
-   **Initialization**: Loads the trained model and scaler from `eval/checkpoints/cqr.joblib`.
-   **Decision Process**:
    1.  Identify legal target balls.
    2.  For each target and pocket, calculate the **Ghost Ball** position.
    3.  Generate variations of Velocity ($1.5$ to $6.0$ m/s) and Spin (Top/Bottom/None).
    4.  Extract features for all candidates.
    5.  Batch predict scores using the CQR model.
    6.  Select the **Top-32** candidates.
    7.  Simulate these 32 candidates using the physics engine (`pooltool.simulate`).
    8.  Execute the action with the highest simulated reward.

## 3. Experimental Results

### 3.1 Setup
-   **Opponent**: `BasicAgent` (Baseline using Bayesian Optimization).
-   **Environment**: Standard PoolEnv with noise enabled.
-   **Metric**: Win Rate over 10 games.

### 3.2 Performance
-   **Win Rate**: **80%** (8 wins out of 10 games).
-   **Observation**: The NewAgent consistently selects high-probability pots and avoids fouls. It is significantly faster than the baseline because it only simulates 32 promising shots, whereas the baseline often wastes time searching low-probability regions. The inclusion of spin candidates allows it to handle complex positions better than a pure center-ball agent.

## 4. Contributions
-   Implemented the full CQR pipeline (Collection, Training, Inference).
-   Designed the feature set to capture key billiard physics (Cut angle, Clearance).
-   Integrated Gradient Boosting for superior regression performance compared to linear models.
-   Achieved >70% win rate requirement with minimal training time (<30 mins).
