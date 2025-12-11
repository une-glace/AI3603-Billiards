# AI3603-Billiards
AI3603课程台球大作业

## 关键文件说明

| 文件 | 作用 | 在最终测试中是否可修改 |
|------|------|-----------|
| `poolenv.py` | 台球环境（游戏规则） | ❌ 不可修改 |
| `agent.py` | Agent 定义（在 `NewAgent` 中实现你的算法） | ✅ 可修改 `NewAgent` |
| `evaluate.py` | 评估脚本（运行对战） | ✅ 可修改 `agent_b` |
| `PROJECT_GUIDE.md` | 项目详细指南 | 📖 参考文档 |
| `GAME_RULES.md` | 游戏规则说明 | 📖 参考文档 |
| `train/` | 训练相关代码与数据 | 🆕 包含数据采集与模型训练脚本 |
| `eval/` | 测试相关文件 | 🆕 包含训练好的模型权重 |

对作业内容的视频说明：
说明.mp4：https://pan.sjtu.edu.cn/web/share/da9459405eac6252d01c249c3bcb989f
供大家参考，以文字说明为准。

---

## 🚀 项目实现与使用说明

本项目实现了一个基于 **上下文 Q 回归 (Contextual Q Regression, CQR)** 的台球智能体。该智能体结合了**几何启发式搜索**与**机器学习评分模型**，能够在无需大量在线交互的情况下，高效地选择最佳击球策略。

### 1. 环境准备

请确保安装以下依赖库（除基础要求外，本项目使用了 `scikit-learn` 和 `joblib`）：

```bash
pip install scikit-learn joblib tqdm
```

### 2. 训练流程 (复现步骤)

虽然我们已经提供了训练好的模型 (`eval/checkpoints/cqr.joblib`)，但你也可以通过以下步骤复现训练过程：

#### **步骤一：数据采集**
运行采集脚本，让 Agent 自我对弈并记录几何特征与击球结果。
```bash
python train/collect_dataset.py --games 200 --samples 50 --out train/dataset_final.npz
```
*   这将在 `train/` 目录下生成 `dataset_final.npz` 数据集文件。
*   `--games 200`: 模拟 200 局游戏。
*   `--samples 50`: 每一步采样 50 个候选动作。

#### **步骤二：模型训练**
使用采集的数据训练回归模型 (Gradient Boosting)。
```bash
python train/train_cqr.py --data train/dataset_final.npz --out eval/checkpoints/cqr.joblib
```
*   这将训练模型并保存到 `eval/checkpoints/cqr.joblib`。

### 3. 测试与评估

运行 `evaluate.py` 进行对战测试。默认配置为 **NewAgent (我们的模型)** vs **BasicAgent (基准)**。

```bash
python evaluate.py
```

*   `NewAgent` 会自动加载 `eval/checkpoints/cqr.joblib` 模型。
*   在测试中，`NewAgent` 展现了 **~80% 的胜率** (基于10局测试结果)，并且决策速度显著快于使用贝叶斯优化的 BasicAgent。

### 4. 方法简介

我们的 `NewAgent` 决策逻辑如下：
1.  **候选生成**：基于“幽灵球”原理，生成针对所有合法目标球的几十个候选击球参数（包含不同的力度、切角和加塞/Spin）。
2.  **特征提取**：计算每个候选的几何特征（如切球角度、距离、路径是否通畅等）。
3.  **模型初筛**：使用训练好的 CQR 模型快速预测每个候选的预期得分，筛选出 Top-32 个高潜力动作。
4.  **物理仿真**：仅对这 32 个动作调用物理引擎进行精确模拟。
5.  **最终决策**：选择模拟得分最高的动作执行。

这种方法既保留了物理引擎的精确性，又避免了盲目搜索带来的低效，大大提高了胜率和稳定性。
