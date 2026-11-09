import streamlit as st
from pathlib import Path
import pandas as pd, numpy as np, joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Turbofan RUL", layout="wide")
BASE = Path(__file__).parent

@st.cache_resource
def load_model(folder):
    return joblib.load(BASE / folder / "model.joblib")

@st.cache_data
def load_data(folder):
    return pd.read_csv(BASE / folder / "demo_test.csv"), pd.read_csv(BASE / folder / "demo_rul.csv")
@st.cache_data
def load_results(folder):
    return pd.read_csv(BASE / folder / "results.csv")
@st.cache_data
def final_cycle_predictions(folder, model_name=None):
    bundle = load_model(folder)
    test, rul = load_data(folder)
    test_last = test.sort_values("cycle").groupby("unit").tail(1)
    model = bundle[model_name]
    p = np.clip(model.predict(test_last[bundle["feature"]]), 0, None)
    out = pd.DataFrame({"unit": test_last["unit"].values, "pred": p})
    return out.merge(rul, on="unit")

required = ["model.joblib", "demo_test.csv", "demo_rul.csv"]
available = [d for d in ["FD001", "FD003"]
             if all((BASE / d / f).exists() for f in required)]

if not available:
    st.error("No dataset folders found."); st.stop()
selected_folder = st.sidebar.selectbox("C-MAPSS dataset", available)

bundle = load_model(folder=selected_folder)
model_options = [key for key in bundle.keys() if key != "feature"]
selected_model = st.sidebar.selectbox("Model", model_options)
model, feats = bundle[selected_model], bundle["feature"]
test, rul = load_data(folder=selected_folder)
results = load_results(folder=selected_folder
                                       )
st.title("Turbofan engine: remaining useful life")
rmse = bundle.get("test_rmse")
st.caption(f"NASA C-MAPSS {selected_folder} · {selected_model} model"
           + (f" · test RMSE {rmse:.1f} cycles" if rmse else ""))

unit = st.sidebar.selectbox("Engine", sorted(test["unit"].unique()))
selected_sensors = list(model.named_steps["featsel"].get_feature_names_out(feats))
sensor = st.sidebar.selectbox("Sensor to plot", selected_sensors)

eng = test[test["unit"] == unit].sort_values("cycle")
last = int(eng["cycle"].max())
cyc = st.sidebar.slider("Observed up to cycle", int(eng["cycle"].min()), last, last)
sub = eng[eng["cycle"] <= cyc]

pred = float(np.clip(model.predict(sub[feats].tail(1))[0], 0, None))
true_final_rul = int(rul.loc[rul["unit"] == unit, "true_rul"].iloc[0])
true_at_current_cycle = true_final_rul + (last - cyc)
live_error = pred - true_at_current_cycle

c1, c2, c3 = st.columns(3)
c1.metric("Predicted RUL", f"{pred:.0f} cycles")
c2.metric("True RUL", f"{true_at_current_cycle:.0f} cycles")
c3.metric("Error", f"{live_error:.1f}")

if pred < 30:
    st.warning("Predicted failure is close. Schedule maintenance.")
st.info(f"The model was trained with RUL capped at {110} & {115} for FD001 & FD003, so predictions above these threshold values are not reliable.")

st.write("## Result of the demo engines")
model_row = results[results["Model"] == selected_model]
st.write(f"##### **Model**: `{selected_model}`")
st.write(f"##### Execution time: `{model_row['Execution Time'].values[0]:.1f} seconds`")
st.write(f"##### Test RMSE (last cycle): `{model_row['Test RMSE (last cycle)'].values[0]:.1f} cycles`")
st.write(f"##### Test MAE: `{model_row['Test MAE'].values[0]:.1f} cycles`")
st.write(f"##### Test R2: `{model_row['Test R2'].values[0]:.3f}`")
st.write(f"##### Test NASA score: `{model_row['Test NASA score (lower=better)'].values[0]:.3f}` *Lower score means better performance*")
st.write(f"##### Best params: `{model_row['Best params'].values[0]}`")
# st.dataframe(model_row.drop(columns="Best params"), use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row")

st.write(f"## Engine {unit}, sensor {sensor} over observed cycles")
left1, right1 = st.columns(2)
with left1:
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(sub["cycle"], sub[sensor])
    ax.set_xlabel("Cycle"); ax.set_ylabel(str(sensor))
    st.pyplot(fig)
    plt.close(fig)
with right1:   
    st.write(f"Sensor **{sensor}** over the observed cycles. The model uses **{len(selected_sensors)}** selected sensor "
             "readings from the latest cycle only (no operational settings, no cycle count), "
             f"and was trained on **{100}** **{selected_folder}** engines under a single operating condition.")
    st.write("Rising or falling trends in the top sensors (s11, s9, s4, s12) drive the predictions.")

st.write(f"## Engine {unit}, Predicted vs true RUL at final cycle")
left2, right2 = st.columns(2)
with left2:
    fp = final_cycle_predictions(folder=selected_folder, model_name=selected_model)
    fig1, ax1 = plt.subplots(figsize=(5, 3))
    ax1.scatter(fp["true_rul"], fp["pred"], alpha=0.6, label="demo engines")
    sel = fp[fp["unit"] == unit]
    ax1.scatter(sel["true_rul"], sel["pred"], color="red", s=60, label=f"engine {unit}")
    ax1.plot([0, 150], [0, 150], color="black", linestyle="--", label="perfect prediction")
    ax1.set_xlabel("True RUL (final cycle)"); ax1.set_ylabel("Predicted RUL")
    ax1.legend()
    st.pyplot(fig1)
    plt.close(fig1)

with right2:
    st.write("Each dot is one demo engine at its final cycle. The red dot is the selected engine; "
             "The distance of selected engine from the dashed line is the error in predicted RUL")
    st.write("The training target was capped at 110. "
             "Dots above and below the line indicate (costly error) over- and (safety error) under-prediction, respectively.")