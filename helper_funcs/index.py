import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline

import matplotlib.pyplot as plt

def read_files(file_indexes, concat):
    if concat:
        main_data_frame = pd.DataFrame()
    else:
        main_data_frame = []
    first_timestamp = 1730818413390259424

    for fi in file_indexes:
        temp_data_frame = pd.read_csv(f"datasets/1.openloop/measures{fi}.csv")
        temp_data_frame = clean_data_frame(temp_data_frame)
        temp_data_frame["time"] = temp_data_frame["time"].apply(
        lambda x: (x - first_timestamp) / 1_000_000_000
        ).round(2)
        if concat:
            main_data_frame = pd.concat([main_data_frame, temp_data_frame], ignore_index=True)
        else:
            main_data_frame.append(temp_data_frame)
    return main_data_frame

def clean_data_frame(data_frame):
    mask = (data_frame["latency"] == 0) & (data_frame["dispatch"] == 0) 

    if mask.any():
        first_zero_index = mask.idxmax() 
        if data_frame["latency"].iloc[first_zero_index + 1] == 0 and  data_frame["dispatch"].iloc[first_zero_index + 1] == 0:
            return data_frame.loc[:first_zero_index-1]
        else:
            return data_frame
    else:
        return data_frame
    
def remove_outliers(data_frame):
    columns = ["wiops", "latency", "dispatch"]

    for col in columns:
        Q1 = data_frame[col].quantile(0.25)
        Q3 = data_frame[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = data_frame[(data_frame[col] < lower_bound) | (data_frame[col] > upper_bound)]
        # num_outliers = len(outliers)

        # if num_outliers > 0:
        #     print(f"For the column {col}, {num_outliers} have been removed!")


        cleaned_df = data_frame[(data_frame[col] >= lower_bound) & (data_frame[col] <= upper_bound)].reset_index(drop=True)
    return cleaned_df

def find_best_pol_degree_2d(df_train, df_test, feat_col, pred_col):
    X = df_train[feat_col].values.reshape(-1, 1)
    y = df_train[pred_col].values

    degree_score_pairs = []

    for dg in range(1, 30):
        poly = PolynomialFeatures(degree=dg)
        X_poly = poly.fit_transform(X)

        model = LinearRegression()
        model.fit(X_poly, y)

        r2 = r2_score(y, model.predict(X_poly))
        degree_score_pairs.append((dg, r2))

    best_degrees = sorted(degree_score_pairs, key=lambda x: x[1], reverse=True)

    return test_model(df_train, df_test, feat_col, pred_col, best_degrees)

def test_model(df_train, df_test, feat_col, pred_col, best_degrees):
    X_train = df_train[[feat_col]].values
    y_train = df_train[pred_col].values

    X_test = df_test[[feat_col]].values
    y_test = df_test[pred_col].values  

    test_scores = []

    for degree, _ in best_degrees:
        poly = PolynomialFeatures(degree=degree)
        X_train_poly = poly.fit_transform(X_train)

        model = LinearRegression()
        model.fit(X_train_poly, y_train)

        X_test_poly = poly.transform(X_test)
        y_pred = model.predict(X_test_poly)

        score = r2_score(y_test, y_pred)
        test_scores.append((degree, score))

    return max(test_scores, key=lambda x: x[1])

def plot_2d_regression(df_train, df_test, column_dicts):
    plt_rows = len(df_train)
    plt_cols = 3

    fig, axes = plt.subplots(plt_rows, plt_cols, figsize=(8*plt_rows, 6*plt_cols))
    
    axes = axes.flatten()
    for i, df in enumerate(df_train):
        df = remove_outliers(df)

        for j, cd in enumerate(column_dicts):
            feat_col = cd["ft_col"]
            pred_col = cd["prd_col"]

            tuple = find_best_pol_degree_2d(df, df, feat_col, pred_col)
            degree = tuple[0]
            r2score = tuple[1]

            # degree = cd["dgr"]

            X_train = df[[feat_col]].values
            y_train = df[pred_col].values

            X_train = df[[feat_col]].values
            y_train = df[pred_col].values

            poly = PolynomialFeatures(degree=degree)
            X_train_poly = poly.fit_transform(X_train)

            model = LinearRegression()
            model.fit(X_train_poly, y_train)

            X_line = np.linspace(X_train.min(), X_train.max(), 200).reshape(-1, 1)
            X_line_poly = poly.transform(X_line)
            y_line = model.predict(X_line_poly)

            ax = axes[i * plt_cols + j]
            ax.plot(X_line, y_line, color='blue', label='Train Regression Curve')

            ax.scatter(X_train, y_train, color='red', label='Test Data')

            if df_test is not None:
                X_test = df_test[[feat_col]].values
                y_test = df_test[pred_col].values

                ax.scatter(X_test, y_test, color='red', label='Test Data')

            if i * plt_cols + j in range(3):
                ax.set_xlabel(feat_col.upper())
                ax.set_ylabel(pred_col.upper())
                ax.legend()
                
            ax.set_title(f'Polynomial Regression (degree={degree})\n{feat_col.upper()} → {pred_col.upper()}')
    plt.show()   

def find_best_pol_degree_3d(df_train, df_test, feat_cols, pred_col):
    X_train = df_train[feat_cols].values
    y_train = df_train[pred_col].values

    X_test = df_test[feat_cols].values
    y_test = df_test[pred_col].values

    degree_score_pairs = []

    for degree in range(1, 30):
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)

        degree_score_pairs.append((degree, r2))

    best_degrees = sorted(degree_score_pairs, key=lambda x: x[1], reverse=True)

    return best_degrees[0]  

def plot_3d_regression(df_train, feat_cols, pred_col, degree, pred_point=None):
    X = df_train[feat_cols].values
    y = df_train[pred_col].values

    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X, y)

    f1_range = np.linspace(df_train[feat_cols[0]].min(), df_train[feat_cols[0]].max(), 30)
    f2_range = np.linspace(df_train[feat_cols[1]].min(), df_train[feat_cols[1]].max(), 30)
    F1, F2 = np.meshgrid(f1_range, f2_range)
    X_grid = np.c_[F1.ravel(), F2.ravel()]
    Z = model.predict(X_grid).reshape(F1.shape)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(df_train[feat_cols[0]], df_train[feat_cols[1]], df_train[pred_col],
               color='blue', label='Actual', alpha=0.7)

    ax.plot_surface(F1, F2, Z, color='red', alpha=0.4)

    if pred_point:
        pred_input = np.array([pred_point])
        pred_val = model.predict(pred_input)[0]
        ax.scatter(pred_point[0], pred_point[1], pred_val,
                   color='green', s=100, label='Prediction', edgecolors='black')
        print(f"Predicted {pred_col} at {feat_cols}={pred_point} is {pred_val:.2f}")

    ax.set_xlabel(feat_cols[0])
    ax.set_ylabel(feat_cols[1])
    ax.set_zlabel(pred_col)
    ax.set_title(f'3D Polynomial Regression (degree={degree})')

    plt.legend()
    plt.tight_layout()
    plt.show()
