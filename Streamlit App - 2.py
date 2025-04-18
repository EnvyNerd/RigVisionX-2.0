import streamlit as st
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import xlsxwriter
import os

st.set_page_config(page_title="RigVisionX Reservoir Predictor", layout="wide")
st.title("🛢️ RigVisionX: Reservoir Production Predictor")

# Upload CSV
uploaded_file = st.file_uploader("📂 Upload your reservoir CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("CSV successfully loaded!")

    tab1, tab2 = st.tabs(["📊 Data Exploration", "🤖 Model Training"])

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

    with tab2:
        st.subheader("Model Training with XGBoost")

        if st.button("Train Model"):
            # Define features and target
            X = df.drop(columns=["Production Rate"])
            y = df["Production Rate"]

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Hyperparameter tuning
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

            st.success(f"Model Trained! ✅ R² Score: {r2:.2f}, MSE: {mse:.2f}")
            st.write("### Best Parameters:", grid_search.best_params_)

            # Feature Importance
            st.subheader("Feature Importance")
            fig_feat, ax_feat = plt.subplots()
            xgb.plot_importance(best_model, ax=ax_feat)
            st.pyplot(fig_feat)

            # Prediction plot
            st.subheader("Actual vs Predicted")
            fig_pred, ax_pred = plt.subplots()
            ax_pred.scatter(y_test, y_pred, alpha=0.7)
            ax_pred.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            ax_pred.set_xlabel("Actual")
            ax_pred.set_ylabel("Predicted")
            ax_pred.set_title("Actual vs Predicted Production Rate")
            st.pyplot(fig_pred)

            # Save model
            joblib.dump(best_model, "rigvisionx_best_model.pkl")

            # Report Generation
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

                workbook.close()
                with open(report_path, "rb") as f:
                    st.download_button("📥 Download Report", f, file_name=report_path)

                os.remove(report_path)
