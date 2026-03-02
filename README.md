# Teleological Explainability in Deep Learning

Research testbed comparing **teleological** explainability methods against standard efficient-causal methods in two settings:

1. **Image Classification** on CUB-200-2011 (fine-grained bird species)
2. **Reinforcement Learning** in MiniGrid (goal-conditioned navigation)

## Methods Implemented

### Image Setting
| Method | Type | Description |
|--------|------|-------------|
| GradCAM | Baseline | Gradient-weighted class activation maps at layer4 |
| Integrated Gradients | Baseline | Path-integrated gradient attribution |
| KernelSHAP | Baseline | Shapley value attribution via gradient SHAP |
| **Purposive Saliency** | **Teleological** | Per-competitor integrated gradient maps targeting margin logits |
| **Means-End Decomposition** | **Teleological** | Layer-wise discriminative capacity analysis |

### RL Setting
| Method | Type | Description |
|--------|------|-------------|
| Jacobian Saliency | Baseline | Gradient of log-policy w.r.t. observation |
| Value-Difference Saliency | Baseline | Cell-masking impact on value estimates |
| **Sub-Goal Imputation** | **Teleological** | Post-hoc sub-goal segmentation from trajectories |
| **Counterfactual Goal Analysis** | **Teleological** | Goal-conditioned behavioral counterfactuals |

## Repository Structure

```
teleological-xai/
├── README.md
├── requirements.txt
├── configs/
│   ├── image_config.yaml        # All image-setting hyperparameters
│   └── rl_config.yaml           # All RL-setting hyperparameters
├── data/
│   └── (auto-downloaded)
├── models/
│   ├── image/
│   │   ├── train_classifier.py  # Fine-tune ResNet-50 on CUB-200
│   │   └── checkpoint/          # Saved weights
│   └── rl/
│       ├── train_agent.py       # Train PPO agent on MiniGrid
│       └── checkpoint/
├── methods/
│   ├── baselines/
│   │   ├── gradcam.py
│   │   ├── integrated_gradients.py
│   │   ├── shap_explainer.py
│   │   └── rl_jacobian_saliency.py
│   ├── teleological/
│   │   ├── purposive_saliency.py
│   │   ├── means_end_decomposition.py
│   │   ├── subgoal_imputation.py
│   │   └── counterfactual_goal.py
│   └── utils/
│       ├── confusion_sets.py
│       ├── hooks.py              # Activation extraction
│       ├── mutual_information.py
│       └── trajectory_utils.py
├── evaluation/
│   ├── image_eval.py
│   ├── rl_eval.py
│   ├── human_eval_protocol.md
│   └── metrics.py
├── experiments/
│   ├── run_image_experiments.py
│   ├── run_rl_experiments.py
│   └── analysis/
│       ├── plot_results.py
│       └── statistical_tests.py
└── tests/
    ├── test_purposive_saliency.py
    ├── test_means_end.py
    ├── test_subgoal.py
    └── test_counterfactual_goal.py
```

## Quick Start

### 1. Install Dependencies
```bash
# Create and activate the virtual environment
python3 -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

> **Note:** All commands below assume the venv is activated. Alternatively, prefix each command with `.venv/bin/python` or `.venv/bin/pytest`.

### 2. Image Setting

#### Train ResNet-50 on CUB-200-2011
```bash
# Download CUB-200-2011 from https://www.vision.caltech.edu/datasets/cub_200_2011/
# Extract to data/CUB_200_2011/

python models/image/train_classifier.py \
    --data-dir data/CUB_200_2011 \
    --save-dir models/image/checkpoint \
    --num-epochs 60 \
    --batch-size 16
```
Target: ≥82% top-1 test accuracy.

#### Run Image Experiments
```bash
python experiments/run_image_experiments.py \
    --config configs/image_config.yaml \
    --data-dir data/CUB_200_2011 \
    --checkpoint models/image/checkpoint/best.pth \
    --results-dir results/image
```

### 3. RL Setting

#### Train PPO Agents
```bash
python models/rl/train_agent.py \
    --env MiniGrid-DoorKey-8x8-v0 \
    --timesteps 2000000 \
    --save-dir models/rl/checkpoint

python models/rl/train_agent.py \
    --env MiniGrid-KeyCorridorS4R3-v0 \
    --timesteps 5000000 \
    --save-dir models/rl/checkpoint

python models/rl/train_agent.py \
    --env MiniGrid-MultiRoom-N6-v0 \
    --timesteps 5000000 \
    --save-dir models/rl/checkpoint
```

#### Run RL Experiments
```bash
python experiments/run_rl_experiments.py \
    --config configs/rl_config.yaml \
    --checkpoint-dir models/rl/checkpoint \
    --results-dir results/rl
```

### 4. Run Tests
```bash
python -m pytest tests/ -v
```

## Evaluation Metrics

### Image Setting
- **PBPA** (Part-Based Purposive Alignment): saliency alignment with diagnostic bird parts
- **Deletion AUC**: faithfulness via pixel masking (lower = better)
- **Insertion AUC**: faithfulness via pixel revealing (higher = better)
- **Purposive Specificity (PS)**: distinctness of per-competitor maps (teleological only)
- **Means-End Coherence**: monotonicity and consumption consistency

### RL Setting
- **Segment Boundary F1**: accuracy of sub-goal transition detection
- **Sub-Goal Label Accuracy**: correctness of imputed sub-goal labels
- **Counterfactual Validity**: accuracy of goal-conditioned behavioral predictions
- **GN vs. Entropy**: whether goal necessity predicts behavioral informativeness
- **Human-Proxy Predictability**: 5-step action prediction with/without teleological info

## Key Hypotheses

- **H1**: Purposive saliency achieves competitive/superior PBPA vs. baselines
- **H2**: Purposive specificity PS > 0.3 (per-competitor maps are meaningfully distinct)
- **H3**: Means-end decomposition reveals interpretable layer roles (early=coarse, late=fine)
- **H4**: Sub-goal imputation achieves >0.8 boundary F1 on DoorKey-8x8
- **H5**: Counterfactual goal analysis achieves >85% first-action accuracy
- **H6**: Teleological explanations improve next-5-action predictability

## Hardware Requirements

All experiments runnable on a single GPU with ≥16 GB VRAM (e.g., NVIDIA A100 or V100).
Computationally expensive operations (confusion sets, MI estimation) are precomputed and cached.
