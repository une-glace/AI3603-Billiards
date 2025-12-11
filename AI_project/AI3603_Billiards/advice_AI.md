**方法选择**
- 方案：上下文 Q 回归（Contextual Q Regression, CQR）+ 幽灵球几何候选搜索
- 思路：用几何先验快速生成可行候选动作；用一个轻量的回归模型估计每个候选的期望回报（奖励=现有 `analyze_shot_for_reward`），在测试时对候选进行打分排序，仅对Top候选做少量物理仿真精炼
- 训练类型：离线学习为主，评估阶段少量在线微调（可选）
- 优点：不需要大规模PPO/SAC训练；用现有物理仿真直接构建样本，训练量小、实现简单、可复现；同时保留规则约束与几何直觉

**为何可优于现有 NewAgent**
- 将“候选生成”与“评分学习”解耦，模型专注于学习稳定的评分函数，减少对纯搜索或随机扰动的依赖
- 特征包含切球角度、袋口距离、母球到幽灵球距离、路径遮挡、力度等，能较好表征动作质量与犯规风险
- Top-K 预筛选后仅仿真少量候选，既节省时间，又能避免噪声导致的随机失败

**文件部署**
- 训练相关文件（放入 `train/`）
  - `train/collect_dataset.py`：采样动作并仿真，输出训练数据集（X特征、y奖励）
  - `train/train_cqr.py`：训练评分模型（`scikit-learn` 的 `SGDRegressor`），保存 `joblib` checkpoint
  - `train/README.md`：说明环境、命令、超参（你的项目要求需要，有内容简明即可）
- 测试相关文件（放入 `eval/`）
  - `eval/checkpoints/cqr.joblib`：训练好的评分模型
  - `eval/README.md`：说明测试环境与命令、超参
  - 修改 `agent.py` 中 `NewAgent` 为加载 `cqr.joblib` 的打分器，按评分排序候选并少量仿真
- 依赖
  - 在 `requirements.txt` 增加 `scikit-learn` 与 `joblib`（`bayesian-optimization` 已带 `scikit-learn`，但建议明确列出以保证可复现）

**训练管线**
- 数据收集
  - 环境：启用噪声（与最终评测一致），两种球型 `solid/stripe` 轮换，收集多局不同起始布局
  - 对每个观测 `(balls, my_targets, table)`：
    - 几何候选：基于幽灵球朝袋口直线，力度集合如 `V0 ∈ {1.5, 3.0, 5.5}`，角度微扰 `±0.5°`
    - 特征计算：
      - `f_cut`：切球角归一化（越小越好）
      - `f_dp`：目标球→袋口距离反比
      - `f_cg`：母球→幽灵球距离反比
      - `clear`：路径遮挡二值（是否有球挡在母球-幽灵球线段附近）
      - `f_v`：力度归一化
      - 可加：袋口类型one-hot、目标球类别（solid/stripe）、局内剩余我方球数等
    - 仿真得到奖励 `y = analyze_shot_for_reward(shot, last_state, my_targets)`
    - 存储 `(X, y)`，建议样本量 8k～15k 即可（数小时内能完成）
- 模型训练
  - 模型：`SGDRegressor(loss='squared_error', learning_rate='adaptive', eta0=0.05, alpha=1e-5)`，配合 `StandardScaler`
  - 训练：打乱数据、小批次 `partial_fit`（适合增量），若收敛慢再跑2～3个 epoch
  - 早停：验证集 MSE 不再下降时停止
  - 保存：`joblib.dump({'scaler': scaler, 'model': sgd}, 'eval/checkpoints/cqr.joblib')`
- 训练时建议
  - 样本均衡：不同力度与角度微扰都要覆盖；同时包含失败动作（犯规、未碰库）作为负样本
  - 噪声一致性：保持环境噪声开，模型会学到对噪声更鲁棒的评分

**测试期 NewAgent 替换**
- 初始化时加载 `eval/checkpoints/cqr.joblib`
- 在 `decision`：
  - 基于当前观测生成候选（同几何规则）
  - 对每个候选计算特征，模型打分
  - 排序取 Top-K（如 24），加少量随机探索（如 6 个）
  - 对选中的候选进行物理仿真评分，取最高分动作作为输出
  - 可选：对刚仿真的样本调用一次 `partial_fit` 微调（小步长），进一步适配场景

**命令与流程**
- 创建并激活 Conda 环境（Windows）
```bat
conda create -n pooltool python=3.10 -y
conda activate pooltool
cd E:\a_github\AI_project\AI3603-Billiards
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install scikit-learn joblib
```
- 数据收集与训练
```bat
:: 收集数据
python train\collect_dataset.py --games 200 --samples_per_obs 40 --out train\dataset.npz

:: 训练评分器
python train\train_cqr.py --data train\dataset.npz --out eval\checkpoints\cqr.joblib
```
- 评估对战
```bat
python evaluate.py
```

**超参建议**
- 候选力度：`[1.5, 3.0, 5.5]` 或 `[2.0, 3.5, 5.0]`（更稳）
- 候选角度扰动：`±0.5°`
- `Top-K`：20～36（权衡速度/质量）
- `SGDRegressor`：`eta0=0.05`、`alpha=1e-5`、`batch_size=2048` 左右；根据验证集调
- 在线微调：学习率小于离线（如 `eta0=0.01`），每局限制更新次数

**预期效果**
- 与基准 `BasicAgent` 对战胜率提升到 55%～65%（具体视训练覆盖与噪声而定）
- 出杆数下降（更少的犯规与无效击球）
- 在噪声情况下更稳定，因为模型对几何不良态势（大切角、远距离、遮挡）有明显惩罚

**项目要求映射**
- 训练相关文件：`train/collect_dataset.py`, `train/train_cqr.py`, 训练复现实验命令与超参在 `train/README.md` 中
- 测试相关文件：仅修改 `agent.py`（实现新 `NewAgent` 加载 `cqr.joblib`），在 `eval/checkpoints` 放置模型，说明见 `eval/README.md`
- 报告：说明方法、实现细节、评估结果，以及分工与贡献；强调训练规模不大、算法简单、效果可复现

**进一步增强（可选）**
- 加入简单的 `a/b` 杆头偏移搜索到候选集合，模型将自动学到何时需要轻微加塞
- 数据增强：在候选特征中加入对“首球是否可能对方球”的判别，进一步降低首球犯规概率
- 分口模型：不同袋口训练子模型或在特征中加入袋口One-Hot，提高对角袋的适配

如果你希望，我可以在下一步提供这套 `train/collect_dataset.py` 与 `train/train_cqr.py` 的最小可运行脚本，以及 `agent.py` 的替换实现（加载 `cqr.joblib` 的 `NewAgent`）。你可以直接运行命令完成数据收集与训练，然后用 `evaluate.py` 进行对战评估。