import pandas as pd
from src.utils.pred_plot_utils import *
import itertools
import numpy as np
from datetime import datetime

DATASETS_DIR_PATH = "./datasets/1.openloop/"

files = load_files(DATASETS_DIR_PATH)
no_of_files = len(files)

"""
Fetch all column names from the dataset
"""
all_df_cols = pd.read_csv(DATASETS_DIR_PATH + "measures0.csv").columns.tolist()

# Swaping the last 2 elements in order to ensure "latency" stays the last one so
# it would be always the target column in case no target column is passed by user

temp = all_df_cols[-2]
all_df_cols[-2] = all_df_cols[-1]
all_df_cols[-1] = temp

print(all_df_cols)

"""
Provides basic information to the user of the functionalities of the CLI.
"""


def welcome():
    banner = """
    #####################################################\n
      CLI TOOL FOR CREATING MODELS AND PLOTTING RESULTS\n
    #####################################################\n
    """
    print(banner)

    print(
        "\nNote: All plots and models will be saved in separate subfolders inside the following directories:\n"
        "'./outputs/plots' and './outputs/models'.\n"
        "Each subfolder is named with the date and time, and contains the corresponding saved files.\n"
    )

    choice = input(
        "This tool has the following functionalities, select the index for each:\n"
        "1. Predict and plot a regression model based on your settings.\n"
        "2. Plot timeseries for each column.\n"
        "3. Predict and plot a xgb model.\n"
        "Or press q to exit the interface.\n\n"
        "Your choice is: "
    )

    return choice


"""
Prepares the enivronment for creating xgb models
"""


def execute_xgb_model():
    files = load_files(DATASETS_DIR_PATH)
    global_df_dict = concat_dataframes(read_files(files))

    target_col = "latency"
    cols = list(filter(lambda col: col != target_col, all_df_cols))

    MAIN_PATH_DIR = "outputs/plots/xgb/"

    time_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    TIMESTAMP_DIR = os.path.join(MAIN_PATH_DIR, time_now)
    os.mkdir(TIMESTAMP_DIR)

    create_and_plot_xgb_model(global_df_dict, cols, target_col, TIMESTAMP_DIR)
    create_and_plot_xgb_model(
        global_df_dict, ["time", "wiops"], target_col, TIMESTAMP_DIR
    )
    create_and_plot_xgb_model(
        global_df_dict, ["time", "dispatch"], target_col, TIMESTAMP_DIR
    )


"""
Prepares the enivronment for creating regression models
"""


def get_regression_setting():
    file_indexes = input(
        "Select by typing the indexes of the files from the datasets directory with space in between,\n"
        "choose a range of indexes (eg. 0-6) or just press enter to select all the files for both training and testing:\n"
    ).split()

    if len(file_indexes) == 1 and len(file_indexes[0].split("-")) == 2:
        limits = [int(n) for n in file_indexes[0].split("-")]
        file_indexes = range(limits[0], limits[1] + 1)
    elif len(file_indexes) > 1:
        for fi in file_indexes:
            if fi not in range(no_of_files):
                print("Invalid input, I consider all files.")
                file_indexes = range(no_of_files)
    else:
        print("Invalid input, I consider all files.")
        file_indexes = range(no_of_files)

    file_indexes = [int(i) for i in file_indexes]

    chosen_files_train = []
    chosen_files_test = []

    if len(file_indexes) == no_of_files:
        for i in range(no_of_files):
            chosen_files_train.append(files[i])
        chosen_files_test = chosen_files_train.copy()
    else:
        for i in range(no_of_files):
            if i in file_indexes:
                chosen_files_train.append(files[i])
            else:
                chosen_files_test.append(files[i])

    print("\nYou have selected the following files for training:")
    for file in chosen_files_train:
        print(file.name)
    print("\nYou have selected the following files for testing:")
    for file in chosen_files_test:
        print(file.name)

    df_train_dicts = read_files(chosen_files_train)
    df_train_global_dict = [concat_dataframes(df_train_dicts)]
    df_test_dicts = read_files(chosen_files_test)
    df_test_concat = concat_dataframes(df_test_dicts)

    print(
        "\nProcceding to make a model for both each file dataset and global in order to compare\n"
        "its coeficitents and intercept."
    )

    print(
        "Would you like the regression to be shown in 2D (one independent variable, easy to plot as a line) "
        "or in 3D (two independent variables, shown as a plane)?\n"
    )

    print("Select the index of your choice:\n1. 2D\n2. 3D\n")
    reg_type_choice = input("Your choice is: ")

    print(
        f"Inside 1.openloop folder we have {no_of_files} datasets, each having the following columns:"
    )
    for i, col in enumerate(all_df_cols):
        print(f"{i}. {col}")    

    if reg_type_choice == "1":
        chosen_columns_indexes = input(
            "\nChoose at least two columns by typing the indexes of each otherwise\n"
            "just press enter to select all of them: "
        ).split()
        print(
            "The software will take the selected columns and run regressions on every possible pair of them."
        )
        chosen_columns = []
        if chosen_columns_indexes and len(chosen_columns_indexes) > 1:
            for i in chosen_columns_indexes:
                chosen_columns.append(all_df_cols[int(i)])
        else:
            chosen_columns = all_df_cols

        print(
            "\nSelect polynomial degrees by typing them for each combination of columns,\n"
            "press enter to skip it so the software will automatically choose an efficient one."
        )

        print(
            "\nThe optimal degree is selected from the `best_degrees.json` file.\n"
            "It was determined by testing multiple degrees and choosing the one\n"
            "with the highest R² score."
        )

        print(
            "\nNote: A higher polynomial degree allows the model to capture more complex patterns\n"
            "and fit curved lines. However, using too high a degree can cause overfitting,\n"
            "so choose carefully."
        )


        force_degrees_dicts = []

        for feat_col, pred_col in itertools.combinations(chosen_columns, 2):
            try:
                degree = int(
                    input(
                        f"Polynomial degree for {feat_col.upper()}-{pred_col.upper()}: "
                    )
                )
            except ValueError:
                degree = 0
            force_degrees_dicts.append(
                {"feat_col": feat_col, "pred_col": pred_col, "degree": degree}
            )

        create_and_plot_simple_regression(
            df_train_dicts, df_test_concat, chosen_columns, force_degrees_dicts
        )

        create_and_plot_simple_regression(
            df_train_global_dict, df_test_concat, chosen_columns, force_degrees_dicts
        )

    elif reg_type_choice == "2":
        print(
            "\nChoose two columns as features and one as predicted column.\nType the indexes for feature columns "
            "and then the predicted one\n"
        )
        feature_columns_indexes = input("Feature columns: ").split()
        predicted_column_index = input("Prediction column: ").split()

        if feature_columns_indexes and len(feature_columns_indexes) == 2:
            chosen_feat_cols = []
            for i in feature_columns_indexes:
                chosen_feat_cols.append(all_df_cols[int(i)])
        else:
            print(
                "Invalid input.\nThe system will automatically choose a set of feature columns and a prediction one.\n"
            )
            chosen_feat_cols = ["wiops", "dispatch"]
            chosen_pred_col = "latency"

        if predicted_column_index and len(predicted_column_index) == 1:
            chosen_pred_col = all_df_cols[int(predicted_column_index)]
        else:
            print(
                "Invalid input.\nThe system will automatically choose a set of feature columns and a prediction one.\n"
            )
            chosen_feat_cols = ["wiops", "dispatch"]
            chosen_pred_col = "latency"

        print("\nBelow, the average polynomial degree and r2score for each combination of columns across all dataset files.\n")

        create_and_plot_3d_regression_model(
            df_train_dicts, chosen_feat_cols, chosen_pred_col
        )
        create_and_plot_3d_regression_model(
            df_train_global_dict, chosen_feat_cols, chosen_pred_col
        )


"""
Directly plots the timeseries for each column and each dataset passed
"""


def plot_timeseries():
    # Create the directory where the timeseries will be saved with current date and time
    time_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    MAIN_DIR_PATH = os.path.join("outputs/plots/timeseries/", time_now)
    os.mkdir(MAIN_DIR_PATH)

    # Fetch the dictionaires containing all the 1.openloop datasets
    dataframe_dicts = read_files(files)
    # Fetch the global dictionary containing all the dataframes concatenated
    global_dataframe_dict = concat_dataframes(read_files(files))

    global_dataframe = global_dataframe_dict["data_frame"]
    global_dataset_name = "global"

    # Select all the columns except the first which is the time
    columns = all_df_cols[1:]

    for col in columns:
        SECONDARY_DIR_PATH = os.path.join(MAIN_DIR_PATH, col)
        os.mkdir(SECONDARY_DIR_PATH)

        y_label = col

        for dfd in dataframe_dicts:
            dataframe = dfd["data_frame"]
            dataset_name = dfd["dataset_name"]
            title = f"Time series of {col}\n{dataset_name}"

            time_df = dataframe["time"]
            X_label = "time"

            y_df = dataframe[col]

            plot_results(
                time_df,
                y_df,
                "red",
                title,
                X_label,
                y_label,
                f"{col} evolution in time",
                [],
                dataset_name.split(".")[0],
                SECONDARY_DIR_PATH,
            )

        plot_results(
            global_dataframe["time"],
            global_dataframe[col],
            "red",
            title,
            X_label,
            y_label,
            f"{col} evolution in time",
            [],
            global_dataset_name,
            SECONDARY_DIR_PATH,
        )
