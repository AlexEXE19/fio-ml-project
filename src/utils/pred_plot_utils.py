import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
from sklearn.pipeline import make_pipeline
import xgboost as xgb

import matplotlib.pyplot as plt

import itertools
import json
import os
from datetime import datetime
import re
import pickle


"""
Load '.csv' files from the passed directory
"""


def load_files(dir_path):
    files = []
    for entry in os.scandir(dir_path):
        if entry.is_file() and entry.name.endswith(".csv"):
            files.append(entry)

    return files


"""
Read files content and save it into a list of dictionary containing the dataframe and dataset name
"""


def read_files(files):
    main_data_frame = []
    FIRST_TIMESTAMP = 1730818413390259424

    for file in files:
        temp_data_frame = pd.read_csv(file.path)
        temp_data_frame = clean_data_frame(temp_data_frame)

        # normalize time
        temp_data_frame["time"] = (
            temp_data_frame["time"]
            .apply(lambda x: (x - FIRST_TIMESTAMP) / 1_000_000_000)
            .round(2)
        )

        # extract run_id from filename "measures<index>.csv"
        match = re.search(r"measures(\d+)\.csv", file.name)
        if match:
            run_id = int(match.group(1))
        else:
            run_id = -1  # fallback if regex fails

        temp_data_frame["run_id"] = run_id

        main_data_frame.append(
            {"data_frame": temp_data_frame, "dataset_name": file.name}
        )

    return main_data_frame


"""
Concat dataframes returns only one dataframe dictionary having concatenated all its passed dataframes
"""


def concat_dataframes(data_frame_dicts):
    main_data_frame = pd.DataFrame()

    for dfd in data_frame_dicts:
        main_data_frame = pd.concat(
            [main_data_frame, dfd["data_frame"]], ignore_index=True
        )

    return {"data_frame": main_data_frame, "dataset_name": "global"}


"""
Cleans rows where the dataframe begins to have null rows all the way to the end of the dataframe
"""


def clean_data_frame(data_frame):
    mask = (data_frame["latency"] == 0) & (data_frame["dispatch"] == 0)

    if mask.any():
        first_zero_index = mask.idxmax()
        if (
            data_frame["latency"].iloc[first_zero_index + 1] == 0
            and data_frame["dispatch"].iloc[first_zero_index + 1] == 0
        ):
            return data_frame.loc[: first_zero_index - 1]
        else:
            return data_frame
    else:
        return data_frame


"""
Remove outliers using percentile method, function is subject to change because the actual difference is minimal
therefore we need a more efficient approach
"""


def remove_outliers(data_frame: pd.DataFrame):
    columns = ["wiops", "latency", "dispatch"]
    cleaned_df = data_frame.copy()

    for col in columns:
        Q1 = cleaned_df[col].quantile(0.25)
        Q3 = cleaned_df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        cleaned_df = cleaned_df[
            (cleaned_df[col] >= lower_bound) & (cleaned_df[col] <= upper_bound)
        ].reset_index(drop=True)

    return cleaned_df


"""
Checks if for the specific columns and dataset name we already have saved into the record an efficient polynomial degree
and then returns it otherwise returns False
"""


def check_for_best_degree(feat_col, pred_col, dataset_name):
    file_path = "./outputs/best_degrees.json"

    if os.path.exists(file_path) and os.path.getsize(file_path) != 0:
        with open(file_path, "r") as f:
            data = json.load(f)

            for d in data:
                if (
                    d["dataset_name"] == dataset_name
                    and d["feat_col"] == feat_col
                    and d["pred_col"] == pred_col
                ):
                    return d

        return False

    else:
        fd = os.open(file_path, os.O_CREAT)
        os.close(fd)

        with open(file_path, "w") as f:
            json.dump([], f)


"""
Save into the record the most efficient polynomial degrees found
"""


def save_best_degree(best_degree_dict):
    file_path = "./outputs/best_degrees.json"

    with open(file_path, "r") as f:
        read_data = json.load(f)

    read_data.append(best_degree_dict)

    with open(file_path, "w") as f:
        json.dump(read_data, f, indent=4)


"""
It finds best polynomial degree by trying multiple values ranging from 1 to 9, it calculates the model score and gets the best
five, ultimately it chooses the smallest degree out of them based on a tolerance which should be bigger than the difference
of the best degree score and other smaller degree score 
"""


def find_best_pol_degree_2d(
    df_train, df_test, feat_col, pred_col, dataset_name, tolerance=0.02
):
    X_train = df_train[feat_col].values.reshape(-1, 1)
    y_train = df_train[pred_col].values

    X_test = df_test[feat_col].values.reshape(-1, 1)
    y_test = df_test[pred_col].values

    degree_score_pairs = []

    for degree in range(1, 10):
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        degree_score_pairs.append((degree, r2))

    best_degrees_sorted = sorted(degree_score_pairs, key=lambda x: x[1], reverse=True)[
        0:5
    ]
    best_degree = best_degrees_sorted[0]
    best_score = best_degree[1]

    for bds in best_degrees_sorted:
        if bds[0] < best_degree[0] and best_score - bds[1] <= tolerance:
            best_degree = bds

    best_degree_dict = {
        "feat_col": feat_col,
        "pred_col": pred_col,
        "dataset_name": dataset_name,
        "best_degree": best_degree[0],
        "r2score": round(best_degree[1], 3),
    }

    save_best_degree(best_degree_dict)

    return best_degree_dict


"""
Create and plot model
"""


def create_and_plot_simple_regression(
    df_train_dict, df_test_dict, columns, force_degrees_dicts=[]
):
    MAIN_DIR_PATH = f"outputs/plots/regressions/2d/"

    if not os.path.isdir(MAIN_DIR_PATH):
        os.mkdir(MAIN_DIR_PATH)

    time_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    SECONDARY_DIR_PATH = os.path.join(MAIN_DIR_PATH, time_now)
    os.mkdir(SECONDARY_DIR_PATH)

    df_test = df_test_dict["data_frame"]

    for feat_col, pred_col in itertools.combinations(columns, 2):
        FILE_PATH = os.path.join(SECONDARY_DIR_PATH, f"{feat_col}-{pred_col}")
        os.mkdir(FILE_PATH)

        X_test = df_test[feat_col].values.reshape(-1, 1)
        y_test = df_test[pred_col].values

        no_of_samples = len(df_train_dict)

        average_pol_degree = 0
        average_r2score = 0

        for dftrd in df_train_dict:
            df_train = dftrd["data_frame"]
            dataset_name = dftrd["dataset_name"]

            df_train = remove_outliers(df_train)

            if force_degrees_dicts:
                for fdd in force_degrees_dicts:
                    if feat_col == fdd["feat_col"] and pred_col == fdd["pred_col"]:
                        degree = fdd["degree"]
            else:
                degree = 0

            X_train = df_train[feat_col].values.reshape(-1, 1)
            y_train = df_train[pred_col].values

            model, r2score = create_2d_regression_model(
                df_train, df_test, feat_col, pred_col, dataset_name, degree
            ).values()

            degree = model.named_steps["polynomialfeatures"].degree
            coefs = model.named_steps["linearregression"].coef_
            intercept = model.named_steps["linearregression"].intercept_

            average_pol_degree += degree
            average_r2score += r2score

            X_line = np.linspace(X_train.min(), X_train.max(), 200).reshape(-1, 1)
            y_line = model.predict(X_line)

            plot_title = f"Polynomial regression (degree={degree})\nModel's coeficitients and intercept are:\n{coefs} - {intercept}"
            plot_label = "Train regression curve"

            scatter_data = [
                {
                    "X_scatter": X_test,
                    "y_scatter": y_test,
                    "color": "red",
                    "label": "Test data",
                },
                {
                    "X_scatter": X_train,
                    "y_scatter": y_train,
                    "color": "skyblue",
                    "label": "Train data",
                },
            ]

            filename = (
                "".join(dataset_name.split(".")[:-1])
                if dataset_name != "global"
                else dataset_name
            )

            plot_results(
                X_line,
                y_line,
                "blue",
                plot_title,
                feat_col.upper(),
                pred_col.upper(),
                plot_label,
                scatter_data,
                filename,
                FILE_PATH,
            )

        average_pol_degree /= no_of_samples
        average_r2score /= no_of_samples

        if filename == "global":
            print(
                f"Polynomial degree and r2score of global model for {feat_col.upper()} and {pred_col.lower()} are: {round(average_pol_degree)} and {round(average_r2score, 3)}"
            )
        else:
            print(
                f"Average polynomial degree and r2score across all datasets for {feat_col.upper()} and {pred_col.lower()} are: {round(average_pol_degree)} and {round(average_r2score, 3)}"
            )


"""
Create a two dimensional regression model
"""


def create_2d_regression_model(
    df_train, df_test, feat_col, pred_col, dataset_name, degree=0
):
    X_train = df_train[feat_col].values.reshape(-1, 1)
    y_train = df_train[pred_col].values

    X_test = df_test[feat_col].values.reshape(-1, 1)
    y_test = df_test[pred_col].values

    r2score = 0

    if degree == 0:
        best_degree_dict = check_for_best_degree(feat_col, pred_col, dataset_name)
        if best_degree_dict:
            degree = best_degree_dict["best_degree"]
        else:
            best_degree_dict = find_best_pol_degree_2d(
                df_train, df_test, feat_col, pred_col, dataset_name
            )
            degree = best_degree_dict["best_degree"]
            r2score = best_degree_dict["r2score"]

    model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression())

    model.fit(X_train, y_train)

    if r2score == 0:
        y_pred = model.predict(X_test)
        r2score = r2_score(y_test, y_pred)

    return {"model": model, "r2score": r2score}


"""
Plots the regression curve or the timeseries
"""


def plot_results(
    X_line,
    y_line,
    color,
    title,
    xlabel,
    ylabel,
    plot_label,
    scatter_data,
    filename,
    file_path,
):
    plt.plot(X_line, y_line, color=color, label=plot_label)

    if scatter_data:
        for sd in scatter_data:
            X_scatter = sd["X_scatter"]
            y_scatter = sd["y_scatter"]
            plt.scatter(X_scatter, y_scatter, color=sd["color"], label=sd["label"])

    plt.title(title)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()

    save_figure(plt.gcf(), filename, file_path)
    plt.clf()


def create_and_plot_3d_regression_model(df_train_dicts, feat_cols, pred_col, degree=2):
    MAIN_DIR_PATH = f"./outputs/plots/regressions/3d/"
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


"""
Find the most efficient number of lags
"""


def find_best_no_lags(global_df_dict, cols, target_col):
    global_df = global_df_dict["data_frame"].copy()
    best_score = 0
    best_n = 0
    average_score = 0

    for n in range(3, 21):
        feat_cols = cols.copy()
        for lag in range(1, n):
            latency_lag = f"latency_lag{lag}"
            global_df[latency_lag] = global_df.groupby("run_id")["latency"].shift(lag)
            feat_cols.append(latency_lag)

        global_df = remove_outliers(global_df.dropna())

        X = global_df[feat_cols]
        y = global_df[target_col]

        # Train/test split by time
        train_mask = global_df["time"] < global_df["time"].quantile(0.7)
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[~train_mask], y[~train_mask]
        time_test = global_df.loc[~train_mask, "time"]

        # Train XGBoost model
        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        average_score += r2

        if r2 > best_score:
            best_score = r2
            best_n = n

    result_dict = {"cols": cols, "no_lags": best_n, "score": round(best_score, 3)}

    save_best_no_lags(result_dict)
    return result_dict

    # print(f"Best score is: {best_score:.4f} for number of lags {best_n}")
    # print(f"Mean of the score is: {average_score / 18:.4f} for columns: {cols}")


"""
Get the saved number of lags for creating the most efficient model
"""


def get_best_no_lags(cols):
    FILE_PATH = "outputs/best_no_lags.json"

    # Ensure file exists with empty list
    if not os.path.isfile(FILE_PATH) or os.path.getsize(FILE_PATH) == 0:
        with open(FILE_PATH, "w") as f:
            json.dump([], f)

    with open(FILE_PATH, "r") as f:
        data = json.load(f)

    # Check if entry exists
    for entry in data:
        if set(entry["cols"]) == set(cols):
            return entry

    return None


"""
Save the found number of lags
"""


def save_best_no_lags(new_entry):
    FILE_PATH = "outputs/best_no_lags.json"

    with open(FILE_PATH, "r") as f:
        data = json.load(f)

    data.append(new_entry)

    # Save updated data
    with open(FILE_PATH, "w") as f:
        json.dump(data, f, indent=4)


"""
Trains and plots a timeseries gradient boosted decision tree (GBDT) regressor
"""


def create_and_plot_xgb_model(global_df_dict, cols, target_col, TIMESTAMP_DIR):
    SUB_DIR = "-".join(cols)
    DIR_PATH = os.path.join(TIMESTAMP_DIR, SUB_DIR)

    os.mkdir(DIR_PATH)

    global_df = global_df_dict["data_frame"].copy()
    best_score = 0
    best_n = 0
    average_score = 0

    best_no_lags_dict = get_best_no_lags(cols)

    if best_no_lags_dict is None:
        best_no_lags_dict = find_best_no_lags(global_df_dict, cols, target_col)

    no_lags = best_no_lags_dict["no_lags"]

    feat_cols = cols.copy()

    for lag in range(1, no_lags):
        latency_lag = f"latency_lag{lag}"
        global_df[latency_lag] = global_df.groupby("run_id")["latency"].shift(lag)
        feat_cols.append(latency_lag)

    global_df = remove_outliers(global_df.dropna())

    X = global_df[feat_cols]
    y = global_df[target_col]

    # Train/test split by time
    train_mask = global_df["time"] < global_df["time"].quantile(0.7)
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[~train_mask], y[~train_mask]
    time_test = global_df.loc[~train_mask, "time"]

    # Train XGBoost model
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Saving the model
    save_model(model, os.path.basename(TIMESTAMP_DIR), file_name=SUB_DIR)

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # # Time series plot
    plt.figure(figsize=(12, 6))
    plt.title(f"r2score: {r2}")
    plt.plot(time_test, y_test.values, label="Actual Latency", color="blue")
    plt.plot(time_test, y_pred, label="Predicted Latency", color="red", linestyle="--")
    plt.xlabel("Time (ms)")
    plt.ylabel("Latency")
    plt.title("Actual vs Predicted Latency (Test Set)")
    plt.legend()

    save_figure(plt.gcf(), "model_plot", DIR_PATH)
    plt.clf()

    # # Feature importance plot
    importance = model.feature_importances_
    plt.figure(figsize=(8, 5))
    plt.barh(feat_cols, importance, color="green")
    plt.xlabel("Importance")
    plt.title("Feature Importances")

    save_figure(plt.gcf(), "importance_plot", DIR_PATH)
    plt.clf()


"""
Save the model
"""


def save_model(model, timestamp, file_name):
    DIR_PATH = os.path.join("outputs/models/xgb", timestamp)
    if not os.path.isdir(DIR_PATH):
        os.mkdir(DIR_PATH)

    FILE_PATH = os.path.join(DIR_PATH, f"{file_name}.pkl")

    with open(FILE_PATH, "wb") as f:
        pickle.dump(model, f)


"""
Save figure
"""


def save_figure(fig, file_name, file_path):
    file_path = os.path.join(file_path, file_name)
    fig.savefig(file_path, dpi=300)
