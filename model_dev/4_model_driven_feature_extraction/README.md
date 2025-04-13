After statistical and domain-based feature selection, we chose **not to apply additional model-driven feature extraction techniques** such as PCA or autoencoders. This is because:

- Our features have already been transformed (e.g., `log`, `clip`, `scale`) and carefully curated.
- The current modeling goal prioritizes **interpretability and traceability**.
- Further dimensionality reduction may improve accuracy marginally, but at the cost of geological insight.

Therefore, we proceed with modeling based on the selected and scaled features from `train_dataset_selected.csv`.
