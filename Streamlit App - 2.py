import streamlit as st
import pandas as pd
import joblib
import os
import glob
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import xlsxwriter

st.set_page_config(page_title="RigVisionX Advanced", layout="wide")
st.title("🛢️ RigVisionX: Advanced Reservoir Predictor")

# ========== Utility Functions ==========

def load_model_versions():
    models = glob.glob("models/*.pkl")
    return {os.path.basename(m): m for m in models}

def load_well_trajectory(file):
    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
    elif file.name.endswith('.las'):
        import lasio
        las = lasio.read(file)
        df = las.df()
        df.reset_index(inplace=True)
        df.rename(columns={'DEPT': 'Z', 'X_LOC': 'X', 'Y_LOC': 'Y'}, inplace=True)
    else:
        st.error("Unsupported file format. Upload .csv or .las")
        return None
    return df

# ========== Upload Data ==========
uploaded_file = st.sidebar.file_uploader("📂 Upload your reservoir CSV", type=["csv"])
well_file = st.sidebar.file_uploader("📥 Upload well trajectory (.csv or .las)", type=["csv", "las"])
model_versions = load_model_versions()
model_choice = st.sidebar.selectbox("📦 Select Model Version", list(model_versions.keys()) if model_versions else ["None"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("CSV loaded successfully.")
    X = df.drop(columns=["Production Rate"])
    y = df["Production Rate"]

    # Train model and auto-save
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    param_grid = {
        "n_estimators": [100],
        "max_depth": [3],
        "learning_rate": [0.1]
    }
    model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring="r2", n_jobs=1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    model_path = f"models/rigvisionx_model_v{len(model_versions)+1}.pkl"
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, model_path)

    y_pred = best_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    st.sidebar.success(f"Model trained! R²: {r2:.2f}")

    tab1, tab2, tab3 = st.tabs(["📊 Evaluation", "🧪 Predict", "🧊 3D Visualization"])

    with tab1:
        st.write("### Metrics Summary")
        st.metric("R² Score", f"{r2:.2f}")
        st.metric("MSE", f"{mse:.2f}")
        st.metric("MAE", f"{mae:.2f}")

    with tab2:
        st.subheader("Input Reservoir Parameters")
        input_data = {col: st.number_input(col, value=float(df[col].mean())) for col in X.columns}
        if st.button("Predict Production"):
            input_df = pd.DataFrame([input_data])
            pred = best_model.predict(input_df)[0]
            st.success(f"🎯 Predicted Production: {pred:.2f} barrels/day")

    with tab3:
        st.subheader("3D Reservoir View + Animated Well Trajectory")
        x = np.linspace(0, 1000, 20)
        y_ = np.linspace(0, 1000, 20)
        z = np.linspace(1000, 1500, 5)
        xx, yy, zz = np.meshgrid(x, y_, z, indexing='ij')
        grid_points = pd.DataFrame({
            "x": xx.flatten(),
            "y": yy.flatten(),
            "z": zz.flatten(),
            "porosity": np.random.uniform(0.1, 0.3, size=xx.size),
            "permeability": np.random.uniform(50, 500, size=xx.size)
        })
        grid_points["prediction"] = (
            grid_points["porosity"] * grid_points["permeability"] * 10 + np.random.normal(0, 100, size=xx.size)
        )

        fig = go.Figure()

        fig.add_trace(go.Scatter3d(
            x=grid_points["x"],
            y=grid_points["y"],
            z=grid_points["z"],
            mode="markers",
            marker=dict(size=3, color=grid_points["prediction"], colorscale="Viridis", opacity=0.6),
            name="Production"
        ))

        if well_file:
            well_df = load_well_trajectory(well_file)
            if well_df is not None and {"X", "Y", "Z"}.issubset(well_df.columns):
                for i in range(1, len(well_df)):
                    fig.add_trace(go.Scatter3d(
                        x=well_df["X"][:i+1],
                        y=well_df["Y"][:i+1],
                        z=well_df["Z"][:i+1],
                        mode="lines+markers",
                        marker=dict(size=4, color="red"),
                        line=dict(width=5),
                        name=f"Drilling Frame {i}",
                        visible=True if i == 1 else False
                    ))

                steps = []
                for i in range(1, len(well_df)):
                    step = dict(method="update", args=[{"visible": [True]*(1+i)}], label=f"{i}")
                    steps.append(step)

                sliders = [dict(steps=steps, currentvalue={"prefix": "Frame: "})]
                fig.update_layout(sliders=sliders)

        fig.update_layout(
            scene=dict(
                xaxis_title='X (m)',
                yaxis_title='Y (m)',
                zaxis_title='Depth (m)',
                zaxis=dict(autorange="reversed")
            ),
            height=700
        )
        st.plotly_chart(fig, use_container_width=True)
