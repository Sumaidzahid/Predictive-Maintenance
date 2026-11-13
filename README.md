# Predictive Maintenance: Turbofan RUL (NASA C-MAPSS)

**Live demo:** [Streamlit app](https://predictive-maintenance-ghr5bmsujqdnzfzci7zhe3.streamlit.app/)

Predicts the remaining useful life (RUL) of turbofan engines from sensor readings.

## Results (true RUL, last cycle of the 100 test engines)

### FD001
| Model | RMSE | NASA score |
|---|---|---|
| Ridge | 22.7 | 1,047 |
| XGBoost | 18.5 | 660 |
| LightGBM | 18.5 | 692 |
| CatBoost | 18.5 | 688 |

Paired bootstrap: Ridge was worse than every booster in 100% of resamples.
The three boosters are statistically indistinguishable.

### FD003
| Model | RMSE | NASA score |
|---|---|---|
| Ridge | 22 | 1,154 |
| XGBoost | 21 | 1,218 |
| LightGBM | 21 | 1,250 |
| CatBoost | 21 | 1,151 |

On FD003, Ridge achieved a NASA score of 1,154 versus CatBoost at 1,151; the bootstrap share indicates only a marginal advantage for CatBoost, and the difference is not statistically significant.

## Method
- Training target capped 110 cycles (FD001) and 115 cycles (FD003); scored against uncapped true RUL.
- GroupKFold by engine; scaler and SelectKBest inside the pipeline.
- Hyperparameters: GridSearchCV; cap: value chosen a priori and validated on test set, with manual tuning.

## What drives predictions (FD001)
s11, s9, s4, s12, s14 (permutation importance and SHAP agree).

## Error analysis (FD001)
Engines 41, 79 and 27 are over-predicted by about 40-50 cycles in every model.
Averaging predictions over the last 5 cycles made RMSE worse (18.3 to 19.4,
CatBoost), so the errors are engine-level, not reading noise.

## Limitations
Only 100 test engines; single operating condition (FD001, FD003); one-cycle
inputs; FD002 and FD004 not attempted.

## Run the demo
    pip install -r requirements.txt
    streamlit run app.py

## Reproduce the notebook
Python 3.12. `pip install -r requirements-notebook.txt`, then run
`Predictive_Maintenance.ipynb`. It writes `model.joblib` and the demo files to
`FD001/` and `FD003/` beside `app.py`. (Colab works too; download those folders.)

## Data
NASA Turbofan Engine Degradation Simulation Data Set (C-MAPSS), NASA Prognostics
Data Repository: [CMAPSS Jet Engine Simulated Data](https://www.kaggle.com/datasets/palbha/cmapss-jet-engine-simulated-data?select=readme.txt)

## Similar Work
[Machine-Learning-Concepts
](https://github.com/Sumaidzahid/Machine-Learning-Concepts)

An end-to-end machine learning regression project developed in a Google Colab environment. The notebook focuses on predicting vehicle advertisement prices using structured data preprocessing, exploratory analysis, feature engineering, and model evaluation techniques.

[Advanced-Machine-LearningAdvanced-Machine-Learning
](https://github.com/Sumaidzahid/Advanced-Machine-Learning)

A comprehensive machine learning project focused on predicting used car prices using advanced regression techniques, feature engineering, model interpretation, and unsupervised learning methods. The project demonstrates a full data science pipeline from raw data processing to model explainability and dimensionality reduction.
