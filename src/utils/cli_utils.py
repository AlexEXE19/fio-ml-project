import os
import itertools
from datetime import datetime
import pandas as pd

from src.utils.pred_plot_utils import (
    load_files,
    read_files,
    concat_dataframes,
    plot_results,
    create_and_plot_simple_regression,
    create_and_plot_3d_regression_model,
    create_and_plot_xgb_model,
)

DATASETS_DIR_PATH = "./datasets/1.openloop/"


# --------------------
# Helpers
# --------------------


def make_timestamp_dir(base_path: str) -> str:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join(base_path, ts)
    os.makedirs(path, exist_ok=True)
    return path


def load_context():
    files = load_files(DATASETS_DIR_PATH)
    no_of_files = len(files)

    cols = pd.read_csv(
        os.path.join(DATASETS_DIR_PATH, "measures0.csv")
    ).columns.tolist()

    # ensure latency is last
    if len(cols) >= 2:
        cols[-2], cols[-1] = cols[-1], cols[-2]

    return files, no_of_files, cols


# --------------------
# CLI
# --------------------


def welcome() -> str:
    banner = """
#####################################################
  CLI TOOL FOR CREATING MODELS AND PLOTTING RESULTS
#####################################################
"""
    print(banner)

    print(
        "Note: All plots and models will be saved in subfolders inside:\n"
        "'./outputs/plots' and './outputs/models'.\n"
        "Each run is timestamped.\n"
    )

    return input(
        "Choose an option:\n"
        "1. Predict and plot a regression model\n"
        "2. Plot timeseries for each column\n"
        "3. Predict and plot an XGB model\n"
        "q. Quit\n\n"
        "Your choice: "
    )


# --------------------
# Commands
# --------------------


def execute_xgb_model(files, all_df_cols):
    global_df_dict = concat_dataframes(read_files(files))

    target_col = "latency"
    feature_cols = [c for c in all_df_cols if c != target_col]

    out_dir = make_timestamp_dir("outputs/plots/xgb")

    create_and_plot_xgb_model(global_df_dict, feature_cols, target_col, out_dir)
    create_and_plot_xgb_model(global_df_dict, ["time", "wiops"], target_col, out_dir)
    create_and_plot_xgb_model(global_df_dict, ["time", "dispatch"], target_col, out_dir)


def parse_file_indexes(raw_input, no_of_files):
    if not raw_input:
        return list(range(no_of_files))

    if len(raw_input) == 1 and "-" in raw_input[0]:
        try:
            start, end = map(int, raw_input[0].split("-"))
            if start > end or start < 0 or end >= no_of_files:
                raise ValueError
            return list(range(start, end + 1))
        except ValueError:
            return list(range(no_of_files))

    try:
        indexes = [int(i) for i in raw_input]
        if any(i < 0 or i >= no_of_files for i in indexes):
            raise ValueError
        return indexes
    except ValueError:
        return list(range(no_of_files))


def get_regression_setting(files, no_of_files, all_df_cols):
    raw_indexes = input(
        "Select dataset indexes (e.g. 0 1 2 or 0-6). Press enter for all: "
    ).split()

    file_indexes = parse_file_indexes(raw_indexes, no_of_files)

    train_files = [files[i] for i in file_indexes]
    test_files = (
        train_files
        if len(file_indexes) == no_of_files
        else [f for i, f in enumerate(files) if i not in file_indexes]
    )

    print("\nTraining files:")
    for f in train_files:
        print(f.name)

    print("\nTesting files:")
    for f in test_files:
        print(f.name)

    df_train_dicts = read_files(train_files)
    df_test_concat = concat_dataframes(read_files(test_files))
    df_train_global = [concat_dataframes(df_train_dicts)]

    reg_type = input("\n[Settings] Select regression type (1 = 2D, 2 = 3D): ")

    print("\nAvailable columns:")
    for i, col in enumerate(all_df_cols):
        print(f"{i}. {col}")

    if reg_type == "1":
        run_2d_regression(df_train_dicts, df_train_global, df_test_concat, all_df_cols)
    elif reg_type == "2":
        run_3d_regression(df_train_dicts, df_train_global, all_df_cols)
    else:
        print("Invalid regression type.")


def run_2d_regression(df_train_dicts, df_train_global, df_test_concat, all_df_cols):
    print(
        "[Settings] Choose column indexes (space separated).\nNote: For invalid input it will select all columns."
    )
    raw_cols = input("Press enter to select all: ").split()

    print("\n[Settings] For the polynomial degree:")
    print("Enter a number to fix the degree.")
    print("Press Enter to auto-calculate the best R2 score (or load from JSON).")

    try:
        chosen_columns = (
            [all_df_cols[int(i)] for i in raw_cols] if raw_cols else all_df_cols
        )
    except (ValueError, IndexError):
        chosen_columns = all_df_cols

    force_degrees = []
    for feat, pred in itertools.combinations(chosen_columns, 2):
        try:
            degree = int(input(f"Polynomial degree for {feat}-{pred}: "))
        except ValueError:
            degree = 0
        force_degrees.append({"feat_col": feat, "pred_col": pred, "degree": degree})

    create_and_plot_simple_regression(
        df_train_dicts, df_test_concat, chosen_columns, force_degrees
    )
    create_and_plot_simple_regression(
        df_train_global, df_test_concat, chosen_columns, force_degrees
    )


def run_3d_regression(df_train_dicts, df_train_global, all_df_cols):
    feat_idx = input("Feature column indexes (2 values): ").split()
    pred_idx = input("Prediction column index: ").split()

    try:
        feat_cols = [all_df_cols[int(i)] for i in feat_idx]
        pred_col = all_df_cols[int(pred_idx[0])]
        if len(feat_cols) != 2:
            raise ValueError
    except (ValueError, IndexError):
        feat_cols = ["wiops", "dispatch"]
        pred_col = "latency"

    create_and_plot_3d_regression_model(df_train_dicts, feat_cols, pred_col)
    create_and_plot_3d_regression_model(df_train_global, feat_cols, pred_col)


def plot_timeseries(files, all_df_cols):
    out_dir = make_timestamp_dir("outputs/plots/timeseries")

    dataframe_dicts = read_files(files)
    global_df = concat_dataframes(dataframe_dicts)["data_frame"]

    for col in all_df_cols[1:]:
        col_dir = os.path.join(out_dir, col)
        os.makedirs(col_dir, exist_ok=True)

        for dfd in dataframe_dicts:
            plot_results(
                X=dfd["data_frame"]["time"],
                y=dfd["data_frame"][col],
                title=f"Time series of {col}\n{dfd['dataset_name']}",
                xlabel="time",
                ylabel=col,
                path=col_dir,
                name=f"{col}_timeseries",
                scatter=None,
            )

        plot_results(
            X=global_df["time"],
            y=global_df[col],
            title=f"Time series of {col}\nGLOBAL",
            xlabel="time",
            ylabel=col,
            path=col_dir,
            name=f"{col}_timeseries_global",
            scatter=None,
        )


# --------------------
# Dispatcher
# --------------------


def run_cli():
    files, no_of_files, all_df_cols = load_context()

    actions = {
        "1": lambda: get_regression_setting(files, no_of_files, all_df_cols),
        "2": lambda: plot_timeseries(files, all_df_cols),
        "3": lambda: execute_xgb_model(files, all_df_cols),
    }

    while True:
        choice = welcome().lower()
        if choice == "q":
            break

        action = actions.get(choice)
        if action:
            action()
        else:
            print("Invalid choice. Try again.")
