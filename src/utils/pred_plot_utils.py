import os
import re
import json
import pickle
import itertools
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

import xgboost as xgb

# ====================
# Constants
# ====================

OUTPUTS_DIR = "outputs"
PLOTS_DIR = os.path.join(OUTPUTS_DIR, "plots")
MODELS_DIR = os.path.join(OUTPUTS_DIR, "models")

BEST_DEGREES_FILE = os.path.join(OUTPUTS_DIR, "best_degrees.json")
BEST_LAGS_FILE = os.path.join(OUTPUTS_DIR, "best_no_lags.json")

FIRST_TIMESTAMP = 1730818413390259424
POLY_DEGREE_RANGE = range(1, 10)
LAG_RANGE = range(3, 21)
TRAIN_SPLIT_Q = 0.7


# ====================
# Files & Directories
# ====================


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def make_timestamp_dir(base_path: str) -> str:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join(base_path, ts)
    ensure_dir(path)
    return path


# ====================
# Data IO
# ====================


def load_files(dir_path):
    return sorted(
        [f for f in os.scandir(dir_path) if f.is_file() and f.name.endswith(".csv")],
        key=lambda f: f.name,
    )


def clean_data_frame(df: pd.DataFrame) -> pd.DataFrame:
    mask = (df["latency"] == 0) & (df["dispatch"] == 0)
    if mask.any():
        idx = mask.idxmax()
        if idx + 1 < len(df) and df.loc[idx + 1, ["latency", "dispatch"]].eq(0).all():
            return df.loc[: idx - 1]
    return df


def read_files(files):
    datasets = []
    for file in files:
        df = pd.read_csv(file.path)
        df = clean_data_frame(df)

        df["time"] = ((df["time"] - FIRST_TIMESTAMP) / 1_000_000_000).round(2)

        match = re.search(r"measures(\d+)\.csv", file.name)
        df["run_id"] = int(match.group(1)) if match else -1

        datasets.append({"data_frame": df, "dataset_name": file.name})

    return datasets


def concat_dataframes(dfs):
    return {
        "data_frame": pd.concat([d["data_frame"] for d in dfs], ignore_index=True),
        "dataset_name": "global",
    }


# ====================
# Cleaning
# ====================


def remove_outliers(df: pd.DataFrame):
    result = df.copy()
    for col in ["wiops", "latency", "dispatch"]:
        q1, q3 = result[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        result = result[(result[col] >= low) & (result[col] <= high)]
    return result.reset_index(drop=True)


# ====================
# Persistence (JSON / Models)
# ====================


def _load_json(path):
    ensure_dir(os.path.dirname(path))
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        with open(path, "w") as f:
            json.dump([], f)
    with open(path, "r") as f:
        return json.load(f)


def _save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def save_model(model, timestamp, name):
    path = os.path.join(MODELS_DIR, "xgb", timestamp)
    ensure_dir(path)
    with open(os.path.join(path, f"{name}.pkl"), "wb") as f:
        pickle.dump(model, f)


# ====================
# Polynomial Degree Logic
# ====================


def get_best_degree(feat_col, pred_col, dataset):
    data = _load_json(BEST_DEGREES_FILE)
    for d in data:
        if (
            d["feat_col"] == feat_col
            and d["pred_col"] == pred_col
            and d["dataset_name"] == dataset
        ):
            return d
    return None


def save_best_degree(entry):
    data = _load_json(BEST_DEGREES_FILE)
    data.append(entry)
    _save_json(BEST_DEGREES_FILE, data)


def find_best_pol_degree_2d(df_train, df_test, feat_col, pred_col, dataset, tol=0.02):
    Xtr = df_train[[feat_col]]
    ytr = df_train[pred_col]
    Xte = df_test[[feat_col]]
    yte = df_test[pred_col]

    scores = []
    for d in POLY_DEGREE_RANGE:
        model = make_pipeline(PolynomialFeatures(d), LinearRegression())
        model.fit(Xtr, ytr)
        r2 = r2_score(yte, model.predict(Xte))
        scores.append((d, r2))

    scores.sort(key=lambda x: x[1], reverse=True)
    best_d, best_score = scores[0]

    for d, s in scores[:5]:
        if d < best_d and best_score - s <= tol:
            best_d, best_score = d, s

    result = {
        "feat_col": feat_col,
        "pred_col": pred_col,
        "dataset_name": dataset,
        "best_degree": best_d,
        "r2score": round(best_score, 3),
    }
    save_best_degree(result)
    return result


# ====================
# Regression Models
# ====================


def create_2d_regression_model(
    df_train, df_test, feat_col, pred_col, dataset, degree=0
):
    if degree == 0:
        entry = get_best_degree(feat_col, pred_col, dataset)
        if entry is None:
            entry = find_best_pol_degree_2d(
                df_train, df_test, feat_col, pred_col, dataset
            )
        degree = entry["best_degree"]

    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(df_train[[feat_col]], df_train[pred_col])

    y_pred = model.predict(df_test[[feat_col]])
    r2 = r2_score(df_test[pred_col], y_pred)

    return model, r2


def create_and_plot_simple_regression(
    df_train_dicts, df_test_dict, columns, force_degrees_dicts=[]
):
    MAIN_DIR_PATH = os.path.join(PLOTS_DIR, "regressions", "2d")
    ensure_dir(MAIN_DIR_PATH)

    time_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    SECONDARY_DIR_PATH = os.path.join(MAIN_DIR_PATH, time_now)
    ensure_dir(SECONDARY_DIR_PATH)

    df_test = df_test_dict["data_frame"]

    for feat_col, pred_col in itertools.combinations(columns, 2):
        FILE_PATH = os.path.join(SECONDARY_DIR_PATH, f"{feat_col}-{pred_col}")
        ensure_dir(FILE_PATH)

        X_test = df_test[feat_col].values.reshape(-1, 1)
        y_test = df_test[pred_col].values

        no_of_samples = len(df_train_dicts)
        average_pol_degree = 0
        average_r2score = 0

        for dftrd in df_train_dicts:
            df_train = remove_outliers(dftrd["data_frame"])
            dataset_name = dftrd["dataset_name"]

            degree = 0
            if force_degrees_dicts:
                for fdd in force_degrees_dicts:
                    if feat_col == fdd["feat_col"] and pred_col == fdd["pred_col"]:
                        degree = fdd["degree"]

            model, r2score = create_2d_regression_model(
                df_train, df_test, feat_col, pred_col, dataset_name, degree
            )

            degree = model.named_steps["polynomialfeatures"].degree
            coefs = model.named_steps["linearregression"].coef_
            intercept = model.named_steps["linearregression"].intercept_

            average_pol_degree += degree
            average_r2score += r2score

            X_line = np.linspace(
                df_train[feat_col].min(), df_train[feat_col].max(), 200
            ).reshape(-1, 1)
            X_line_df = pd.DataFrame(X_line, columns=[feat_col])
            y_line = model.predict(X_line_df)

            scatter_data = [
                {
                    "X_scatter": X_test,
                    "y_scatter": y_test,
                    "color": "red",
                    "label": "Test data",
                },
                {
                    "X_scatter": df_train[feat_col].values,
                    "y_scatter": df_train[pred_col].values,
                    "color": "skyblue",
                    "label": "Train data",
                },
            ]

            plot_results(
                X=X_line,
                y=y_line,
                title=f"Polynomial regression (degree={degree})\nCoefficients: {coefs}, Intercept: {intercept}",
                xlabel=feat_col.upper(),
                ylabel=pred_col.upper(),
                path=FILE_PATH,
                name=dataset_name.split(".")[0]
                if dataset_name != "global"
                else "global",
                scatter=scatter_data,
            )

        average_pol_degree /= no_of_samples
        average_r2score /= no_of_samples
        print(
            f"Average polynomial degree: {round(average_pol_degree)}, Average R2: {round(average_r2score, 3)} for {feat_col}-{pred_col}"
        )


def create_and_plot_3d_regression_model(df_train_dicts, feat_cols, pred_col, degree=2):
    MAIN_DIR_PATH = "./outputs/plots/regressions/3d/"
    if not os.path.isdir(MAIN_DIR_PATH):
        os.mkdir(MAIN_DIR_PATH)

    time_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    SECONDARY_DIR_PATH = os.path.join(MAIN_DIR_PATH, time_now)
    os.mkdir(SECONDARY_DIR_PATH)

    for dfd in df_train_dicts:
        df_train = dfd["data_frame"]

        X = df_train[feat_cols].values
        y = df_train[pred_col].values

        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        model.fit(X, y)

        f1_range = np.linspace(
            df_train[feat_cols[0]].min(), df_train[feat_cols[0]].max(), 30
        )
        f2_range = np.linspace(
            df_train[feat_cols[1]].min(), df_train[feat_cols[1]].max(), 30
        )
        F1, F2 = np.meshgrid(f1_range, f2_range)
        X_grid = np.c_[F1.ravel(), F2.ravel()]
        Z = model.predict(X_grid).reshape(F1.shape)

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")

        ax.scatter(
            df_train[feat_cols[0]],
            df_train[feat_cols[1]],
            df_train[pred_col],
            color="blue",
            label="Actual",
            alpha=0.7,
        )

        ax.plot_surface(F1, F2, Z, color="red", alpha=0.4)

        ax.set_xlabel(feat_cols[0])
        ax.set_ylabel(feat_cols[1])
        ax.set_zlabel(pred_col)
        ax.set_title(f"3D Polynomial Regression (degree={degree})")

        plt.legend()
        plt.tight_layout()
        # plt.show()

        file_name = dfd["dataset_name"].split(".")[0]
        save_figure(plt.gcf(), file_name, SECONDARY_DIR_PATH)
        plt.clf()


# ====================
# Plotting
# ====================


def save_figure(fig, name, path):
    ensure_dir(path)
    fig.savefig(os.path.join(path, name), dpi=300)


def plot_results(X, y, title, xlabel, ylabel, path, name, scatter=None):
    plt.plot(X, y, label="model")
    if scatter:
        for s in scatter:
            plt.scatter(s["X_scatter"], s["y_scatter"], label=s["label"], alpha=0.6)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    save_figure(plt.gcf(), name, path)
    plt.clf()


# ====================
# XGBoost
# ====================


def get_best_no_lags(cols):
    data = _load_json(BEST_LAGS_FILE)
    for d in data:
        if set(d["cols"]) == set(cols):
            return d
    return None


def save_best_no_lags(entry):
    data = _load_json(BEST_LAGS_FILE)
    data.append(entry)
    _save_json(BEST_LAGS_FILE, data)


def find_best_no_lags(global_df, cols, target):
    best = {"no_lags": 0, "score": -np.inf}

    for n in LAG_RANGE:
        df = global_df.copy()
        features = cols.copy()

        for lag in range(1, n):
            name = f"latency_lag{lag}"
            df[name] = df.groupby("run_id")["latency"].shift(lag)
            features.append(name)

        df = remove_outliers(df.dropna())

        mask = df["time"] < df["time"].quantile(TRAIN_SPLIT_Q)
        Xtr, Xte = df.loc[mask, features], df.loc[~mask, features]
        ytr, yte = df.loc[mask, target], df.loc[~mask, target]

        model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)
        model.fit(Xtr, ytr)
        r2 = r2_score(yte, model.predict(Xte))

        if r2 > best["score"]:
            best = {"cols": cols, "no_lags": n, "score": round(r2, 3)}

    save_best_no_lags(best)
    return best


def create_and_plot_xgb_model(global_df_dict, cols, target, out_dir):
    df = global_df_dict["data_frame"].copy()
    best = get_best_no_lags(cols) or find_best_no_lags(df, cols, target)

    features = cols.copy()
    for lag in range(1, best["no_lags"]):
        name = f"latency_lag{lag}"
        df[name] = df.groupby("run_id")["latency"].shift(lag)
        features.append(name)

    df = remove_outliers(df.dropna())

    mask = df["time"] < df["time"].quantile(TRAIN_SPLIT_Q)
    Xtr, Xte = df.loc[mask, features], df.loc[~mask, features]
    ytr, yte = df.loc[mask, target], df.loc[~mask, target]
    t = df.loc[~mask, "time"]

    model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)
    model.fit(Xtr, ytr)

    ts_dir = make_timestamp_dir(out_dir)
    save_model(model, os.path.basename(ts_dir), "-".join(cols))

    y_pred = model.predict(Xte)

    plt.plot(t, yte, label="actual")
    plt.plot(t, y_pred, label="predicted")
    plt.legend()
    save_figure(plt.gcf(), "prediction", ts_dir)
    plt.clf()
