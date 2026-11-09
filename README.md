
# Predictive Maintenance: Turbofan RUL (NASA C-MAPSS)

**Live demo:** <[Streamlit](https://predictive-maintenance-ghr5bmsujqdnzfzci7zhe3.streamlit.app/)>

Predict the remaining useful life (RUL) of turbofan engines from sensor data.

## Results (FD001, true RUL, last cycle of 100 test engines)

Boosted trees beat Ridge in 100% of paired-bootstrap resamples; the three
boosters are statistically indistinguishable.

## Method
- Training target capped at 110 cycles; scored against uncapped true RUL.
- GroupKFold (by engine), scaler and SelectKBest inside the pipeline.
- Hyperparameters: <how you chose them>

## What drives predictions
s11, s9, s4, s12 (permutation importance and SHAP agree).

## Error analysis
Engines 41, 79, 27 are over-predicted by about 40-50 cycles in every model.
Averaging predictions over recent cycles did not help (RMSE 18.3 to 19.4 at k=5).

## Limitations
100 test engines; single operating condition (FD001); cap value chosen
a priori and checked on test, not tuned; FD002/FD004 not attempted.

## Run it
Install the neccesary requirments and run the app. This will give you a demo
`pip install -r requirements.txt`
`streamlit run app.py`
If you requiure personalization than..
Run "Predictive_Maintenance.ipynb" in colab. The final Models and Results will be stored in FD00* Folder directory. download it and store it besides the app.py file.

If running on computer. The FD00* Folder directory will automatically be stored besides app.py file.

Note: The library versions used in the juypiter notebook have been specified in the file. Python version 3.13.18
## Data
NASA Turbofan Engine Degradation Simulation Data Set (C-MAPSS), NASA
Prognostics Data Repository. <[Dataset Scource](https://www.kaggle.com/datasets/palbha/cmapss-jet-engine-simulated-data?select=readme.txt)>
