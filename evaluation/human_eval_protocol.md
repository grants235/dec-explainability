# Human Evaluation Protocol

This document specifies the protocol for human evaluation of teleological vs. efficient-causal explanations.
Human evaluation is **not required** for the automated experiments but provides complementary evidence.

## Overview

We recruit N=20 participants (graduate students in CS/biology) for a forced-choice explanation quality evaluation.

## Task Design

### Image Setting: Explanation Quality Rating

**Setup**: Show participants an image of a bird alongside two explanations:
- Explanation A: A standard GradCAM or IG saliency heatmap overlaid on the image
- Explanation B: A purposive saliency annotation map showing which regions serve different discriminative goals

**Question 1 (Understanding)**: "After seeing this explanation, can you predict what other birds this model might confuse with the one shown? [Yes/No/Somewhat]"

**Question 2 (Informativeness)**: "Which explanation (A or B) tells you more about *why* the model made its prediction? [A / B / Equal]"

**Question 3 (Trust)**: "Which explanation would make you more confident in using this model in a real application? [A / B / Equal]"

**Counterbalancing**: Randomize order of A/B presentation. Use 20 images per participant.

### RL Setting: Action Prediction Task

**Setup**: Show participants a sequence of 5 timesteps from a MiniGrid trajectory (as rendered grid images).
At timestep 5, show the agent's current position and ask them to predict the next action.

**Conditions** (between-subjects):
1. **Observation only**: Show the grid image
2. **Observation + Jacobian saliency**: Show the grid with a saliency overlay
3. **Observation + sub-goal label**: Show the grid plus text "Agent is currently trying to: PICKUP(yellow_key)"
4. **Observation + full teleological explanation**: Sub-goal + "If instead pursuing REACH_GOAL directly, the agent would move RIGHT rather than FORWARD"

**Metric**: Fraction of correct next-action predictions across 20 trajectory clips.

## Scoring

- Understanding: Yes=2, Somewhat=1, No=0
- Informativeness/Trust: Preferred=1, Equal=0.5, Not preferred=0 (per method)
- Action prediction: Binary correct/incorrect

## Analysis

- Paired t-test comparing teleological vs. efficient-causal ratings
- Linear mixed-effects model with participant as random effect
- Report Cohen's d effect size

## Materials

Images and trajectories should be selected to span:
- Easy cases (high model confidence)
- Hard cases (small margin between top classes)
- Misclassified cases
- Cases with clear spatial discrimination (diagnostic parts are distinctive)

## Ethical Considerations

- Participants are informed this is an AI explanation study
- No deception is involved
- Data is anonymized
- Participation is voluntary with option to withdraw
