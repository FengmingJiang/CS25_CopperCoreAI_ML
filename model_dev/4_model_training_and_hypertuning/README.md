# Model Evaluation Report: Copper Deposit Classification

We trained 8 classification models to identify porphyry copper deposits using:

- Two model types: Random Forest (RF), Gradient Boosting (GB)

- Two feature sets: All features vs. Selected features (based on importance)

- Two preprocessing strategies: Transformed (log1p + scaling) vs. Not Transformed (scaling)

## Test Set Evaluation Summary

Model | Transform | Feature Set | Accuracy | F1 (Class 1) | Recall (Class 1) | ROC AUC | PR AUC
RF | yes | All | 0.94 | 0.84 | 0.87 | 0.9813 | 0.9260
RF | yes | Selected | 0.94 | 0.84 | 0.89 | 0.9793 | 0.9167
RF | no | All | 0.94 | 0.85 | 0.89 | 0.9822 | 0.9298
RF | no | Selected | 0.93 | 0.84 | 0.89 | 0.9762 | 0.9054
GB | yes | All | 0.95 | 0.87 | 0.89 | 0.9874 | 0.9502
GB | yes | Selected | 0.95 | 0.88 | 0.91 | 0.9832 | 0.9447
GB | no | All | 0.95 | 0.88 | 0.89 | 0.9882 | 0.9530
GB | no | Selected | 0.94 | 0.84 | 0.89 | 0.9828 | 0.9382

## Model Recommendation

In mineral exploration, recall for class 1 (porphyry copper) is critical to avoid missing valuable targets. Class 1 (porphyry copper deposits) recall is emphasized for geological decision-making. Based on this domain-specific requirement, we recommend:

**Recommended Model: Gradient Boosting + Selected Features + Transformed**
- Recall (Class 1): 0.91 (Best for discovery)
- PR AUC: 0.9447
- Use case: Geoscientific targeting where missing true deposits is costly

**Alternative Model: Gradient Boosting + All Features + No Transform**
- PR AUC: 0.9530 (Best overall)
- Balanced performance across all metrics
- Use case: General-purpose deployment with less preprocessing