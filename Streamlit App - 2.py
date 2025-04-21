import streamlit as st
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import xlsxwriter
import os
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="RigVisionX Reservoir Predictor", layout="wide")
st.title("🛢️ RigVisionX: Reservoir Production Predictor")

# Upload CSV
uploaded_file = st.file_uploader("📂 Upload your reservoir CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("CSV successfully loaded!")

    # Define features and target
    X = df.drop(columns=["Production Rate"])
    y = df["Production Rate"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Hyperparameter tuning
    st.info("🔧 Training model automatically...")
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 4],
        "learning_rate": [0.05, 0.1]
    }

    model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring="r2", n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    # Evaluation
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='r2')

    st.success(f"Model Trained Automatically! ✅ R² Score: {r2:.2f}, MSE: {mse:.2f}")
    st.write("### Best Parameters:", grid_search.best_params_)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Exploration",
        "📈 Predict New Input",
        "📄 Download Report",
        "📋 Evaluation Dashboard",
        "🧊 3D Visualization"
    ])

    with tab1:
        st.subheader("Dataset Overview")
        st.dataframe(df.head())

        st.write("### Histogram for each feature")
        for col in df.columns:
            st.write(f"#### {col}")
            st.bar_chart(df[col])

        st.write("### Correlation Heatmap")
        fig_corr, ax_corr = plt.subplots()
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax_corr)
        st.pyplot(fig_corr)

        st.subheader("Feature Importance")
        fig_feat, ax_feat = plt.subplots()
        xgb.plot_importance(best_model, ax=ax_feat)
        st.pyplot(fig_feat)

        st.subheader("Actual vs Predicted")
        fig_pred, ax_pred = plt.subplots()
        ax_pred.scatter(y_test, y_pred, alpha=0.7)
        ax_pred.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ax_pred.set_xlabel("Actual")
        ax_pred.set_ylabel("Predicted")
        ax_pred.set_title("Actual vs Predicted Production Rate")
        st.pyplot(fig_pred)

    with tab2:
        st.subheader("🧪 Enter New Reservoir Parameters")

        input_data = {}
        for col in X.columns:
            input_data[col] = st.number_input(f"{col}", value=float(df[col].mean()))

        if st.button("Predict Daily Production"):
            input_df = pd.DataFrame([input_data])
            prediction = best_model.predict(input_df)[0]
            st.success(f"💧 Predicted Production Rate: **{prediction:.2f} barrels/day**")

    with tab3:
        st.subheader("📄 Export Model Report")

        if st.button("Generate Excel Report"):
            report_path = "rigvisionx_report.xlsx"
            workbook = xlsxwriter.Workbook(report_path)
            sheet = workbook.add_worksheet("Model Report")

            sheet.write("A1", "Best Parameters")
            for i, (key, val) in enumerate(grid_search.best_params_.items(), start=2):
                sheet.write(f"A{i}", key)
                sheet.write(f"B{i}", val)

            sheet.write("D1", "Metrics")
            sheet.write("D2", "R²")
            sheet.write("E2", r2)
            sheet.write("D3", "MSE")
            sheet.write("E3", mse)
            sheet.write("D4", "MAE")
            sheet.write("E4", mae)
            sheet.write("D5", "RMSE")
            sheet.write("E5", rmse)

            workbook.close()
            with open(report_path, "rb") as f:
                st.download_button("📥 Download Report", f, file_name=report_path)

            os.remove(report_path)

    with tab4:
        st.subheader("📋 Model Evaluation Dashboard")

        st.write("### 📌 Metrics Summary")
        st.metric("R² Score", f"{r2:.2f}")
        st.metric("MSE", f"{mse:.2f}")
        st.metric("MAE", f"{mae:.2f}")
        st.metric("RMSE", f"{rmse:.2f}")

        st.write("### 📉 Residual Plot")
        residuals = y_test - y_pred
        fig_res, ax_res = plt.subplots()
        ax_res.scatter(y_pred, residuals, alpha=0.6)
        ax_res.axhline(0, color='red', linestyle='--')
        ax_res.set_xlabel("Predicted")
        ax_res.set_ylabel("Residual")
        ax_res.set_title("Residuals vs Predicted")
        st.pyplot(fig_res)

        st.write("### 📦 Distribution of Prediction Errors")
        fig_err, ax_err = plt.subplots()
        sns.histplot(residuals, kde=True, ax=ax_err)
        ax_err.set_title("Error Distribution")
        st.pyplot(fig_err)

        st.write("### 🔁 Cross-Validation Scores (R²)")
        st.write(cv_scores)
        st.write(f"Mean R²: {cv_scores.mean():.2f}, Std Dev: {cv_scores.std():.2f}")

    with tab5:
        st.subheader("🧊 3D Reservoir Production Heatmap with Well Trajectories")

        x = np.linspace(0, 1000, 20)
        y = np.linspace(0, 1000, 20)
        z = np.linspace(1000, 1500, 5)
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        grid_points = pd.DataFrame({
            "x": xx.flatten(),
            "y": yy.flatten(),
            "z": zz.flatten(),
            "porosity": np.random.uniform(0.1, 0.3, size=xx.size),
            "permeability": np.random.uniform(50, 500, size=xx.size)
        })
        grid_points["predicted_production"] = (
            grid_points["porosity"] * grid_points["permeability"] * 10 +
            np.random.normal(0, 100, size=xx.size)
        )

        well_trajectories = []
        well_trajectories.append(pd.DataFrame({
            "x": [200] * 10,
            "y": [200] * 10,
            "z": np.linspace(1000, 1500, 10),
            "well_name": ["Well-1"] * 10
        }))
        well_trajectories.append(pd.DataFrame({
            "x": np.linspace(300, 600, 10),
            "y": np.linspace(300, 600, 10),
            "z": np.linspace(1000, 1500, 10),
            "well_name": ["Well-2"] * 10
        }))
        well_trajectories.append(pd.DataFrame({
            "x": np.linspace(100, 900, 10),
            "y": [800] * 10,
            "z": [1450] * 10,
            "well_name": ["Well-3"] * 10
        }))
        wells_df = pd.concat(well_trajectories, ignore_index=True)

        fig = go.Figure()
        fig.add_trace(go.Scatter3d(
            x=grid_points['x'],
            y=grid_points['y'],
            z=grid_points['z'],
            mode='markers',
            marker=dict(
                size=3,
                color=grid_points['predicted_production'],
                colorscale='Viridis',
                colorbar=dict(title='Production'),
                opacity=0.7
            ),
            name='Production Grid'
        ))

        colors = ['red', 'blue', 'green']
        for i, (name, group) in enumerate(wells_df.groupby("well_name")):
            fig.add_trace(go.Scatter3d(
                x=group['x'],
                y=group['y'],
                z=group['z'],
                mode='lines+markers',
                name=name,
                line=dict(width=4, color=colors[i % len(colors)]),
                marker=dict(size=5, symbol='circle', color=colors[i % len(colors)])
            ))

        fig.update_layout(
            title='3D Reservoir with Predicted Production and Well Paths',
            scene=dict(
                xaxis_title='X (m)',
                yaxis_title='Y (m)',
                zaxis_title='Depth (m)',
                zaxis=dict(autorange='reversed')
            ),
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='black',
                borderwidth=1
            ),
            margin=dict(l=0, r=0, b=0, t=30)
        )

        st.plotly_chart(fig, use_container_width=True)

    joblib.dump(best_model, "rigvisionx_best_model.pkl")
