## LendingClub Loan Default Prediction (Keras + scikit-learn)

### Overview
Binary classification to predict whether a borrower will pay back their loan using the LendingClub dataset. The main notebook is `loan.ipynb`. It includes EDA, preprocessing, model training, evaluation, and artifact saving.

### Data
- Input file: `lending_club_loan_two.csv`
- Label: derived from `loan_status` (`Fully Paid` vs `Charged Off`). A `label` column is created (1 = Charged Off).

### Environment
```bash
python -m venv .venv
.\.venv\Scripts\activate  # Windows PowerShell
pip install --upgrade pip
pip install numpy pandas scikit-learn seaborn matplotlib tensorflow
# Optional GPU (only if you have compatible NVIDIA drivers/CUDA): pip install tensorflow[and-cuda]
```

### How to Run
1) Open `loan.ipynb` and run cells top-to-bottom.
2) The notebook performs:
   - EDA with conclusions printed under each plot
   - Target engineering (`loan_status` → `label`)
   - Preprocessing (numeric scaling, categorical encoding with cardinality controls)
   - Train/validation split
   - Keras model training with early stopping and class weights
   - Evaluation (ROC AUC, classification report, confusion matrix, ROC curve)
   - Saving artifacts

### EDA (What you’ll see)
- Class balance bar with a short conclusion
- Top missingness columns with a conclusion
- Numeric distributions by label (subset) with a skewness summary
- Numeric correlation heatmap with high-correlation count
- Categorical default-rate bars (top 6 by mutual information) with a takeaway
- Sampled scatter of two high-variance numerics with a conclusion

### Preprocessing
- Numeric features: `StandardScaler(with_mean=True)`
- Categorical features: `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`
  - Fallback for older scikit-learn: uses `sparse=False`
- To avoid huge OHE matrices, the notebook:
  - Keeps only low-cardinality categoricals (configurable `low_card_limit`)
  - Adds a small whitelist of common LendingClub categoricals
  - Optionally buckets infrequent categories via `min_frequency`

### Training
- Model: Dense feed-forward network (ReLU, Dropout), sigmoid output
- Loss/metrics: Binary cross-entropy, AUC, accuracy
- Class weights computed from training distribution
- Early stopping on validation AUC

### Evaluation & Artifacts
- Printed metrics: ROC AUC, classification report
- Plots: Confusion matrix, ROC curve
- Saved artifacts:
  - Model: `keras_lc_model.keras`
  - Feature names: `feature_names.csv`

### Troubleshooting
- OneHotEncoder error: `TypeError: ... unexpected keyword argument 'sparse'`
  - Use `sparse_output=False` on new scikit-learn; fallback to `sparse=False` on older versions (already handled in the notebook).
- MemoryError during encoding (very large OHE):
  - Increase `min_frequency` (e.g., `0.02`), lower `low_card_limit` (e.g., `20`)
  - Reduce columns kept in the whitelist
  - Consider switching to a sparse end-to-end pipeline or target/WOE encoding
- Keras save error about filepath extension:
  - Use native Keras format: `model.save('keras_lc_model.keras')`

### Optional Enhancements
- Threshold tuning for precision/recall trade-offs
- Calibration curve and Brier score
- Cross-validation and hyperparameter search
- SHAP-based feature attributions
- Sparse pipeline for high-cardinal categoricals or Target/WOE encoding

### Notes
- Seeded for reproducibility (`numpy` and `tensorflow` seeds)
- If you re-run EDA before label engineering, EDA derives a temporary label from `loan_status` automatically


