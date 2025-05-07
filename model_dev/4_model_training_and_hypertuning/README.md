# Model Comparison and Selection Summary

This document summarizes the performance evaluation and final selection of models trained for the binary classification task using Random Forest and Gradient Boosting algorithms under various feature and transformation settings.

## Model Evaluation Summary

| Model ID | Model Description                                 | ROC AUC | PR AUC | Brier ↓ | F1 (Class 1) | Recall (Class 1) |
|----------|---------------------------------------------------|---------|--------|----------|----------------|-------------------|
| **40c**  | Random Forest + All Features + Not Transformed    | **0.9875** | **0.9494** | **0.0390** | **0.88**        | **0.89**          |
| 41a      | Gradient Boosting + All Features + Transformed    | 0.9824  | 0.9330 | 0.0450   | 0.84          | 0.88             |
| 40a      | Random Forest + All Features + Transformed        | 0.9820  | 0.9324 | 0.0449   | 0.84          | 0.88             |
| 41c      | Gradient Boosting + All Features + Not Transformed| 0.9719  | 0.9158 | 0.0475   | 0.84          | 0.83             |
| 41b      | Gradient Boosting + Selected Features + Transformed| 0.9740  | 0.9149 | 0.0481   | 0.83          | 0.87             |
| 40b      | Random Forest + Selected Features + Transformed   | 0.9698  | 0.8965 | 0.0541   | 0.81          | 0.86             |
| 40d      | Random Forest + Selected Features + Not Transformed| 0.9771  | 0.9129 | 0.0511   | 0.82          | 0.88             |
| 41d      | Gradient Boosting + Selected Features + Not Transformed| 0.9637  | 0.8900 | 0.0556   | 0.80          | 0.73             |

> Note: All evaluations were conducted on the held-out test set. Metrics include ROC AUC, Precision-Recall AUC, Brier Score (lower is better), and F1/Recall for the positive class.

---

## Final Model Selection

**Selected Model**: `40c_model_training_rf_with_all_features_not_transformed`

- **Algorithm**: Random Forest Classifier
- **Input**: All features with no transformation
- **Why selected**:
  - Highest ROC AUC (**0.9875**)
  - Best PR AUC (**0.9494**)
  - Lowest Brier Score (**0.0390**) → Excellent probability calibration
  - Strong F1-score and recall for the minority (positive) class

This model provides the best balance between predictive performance and calibrated probability outputs. It will be used as the final model for deployment or downstream decision-making tasks.

---

## File References

- `40c_model_training_rf_with_all_features_not_transformed.ipynb`: Training and evaluation notebook
- `final_model.pkl`: Serialized model for production use

---
