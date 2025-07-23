import itertools

import helper_funcs as hpf

# Read and clean data
df_train = hpf.read_files(list(range(10)), concat=False)
# df_train = hpf.remove_outliers(train_data_frame)

# test_data_frame = hpf.read_files(list(range(0, 9)))
# df_test = hpf.remove_outliers(test_data_frame)

# print(df.tail())

columns = ["wiops", "dispatch", "latency"]
column_dicts = []
output = ""

for feat_col, pred_col in itertools.combinations(columns, 2):
    # degree_r2_tuple = hpf.find_best_pol_degree_2d(df_train, df_train, feat_col, pred_col)
    # degree = degree_r2_tuple[0]

    column_dicts.append({
        "ft_col": feat_col,
        "prd_col": pred_col,
        "dgr": 5
    })

    # output += (
    # 110 * "#" + "\n"
    # + f"For the model where the feature is: {feat_col.upper()} and the predicted column is: {pred_col.upper()}, got the following insights:\n"
    # + f"The model did best when the polynomial degree was: {degree} with a R2 score obtained against the test data of: {round(degree_r2_tuple[1], 4)}\n"
    # + 110 * "#" + "\n\n"
    # )

    # print(output)

    # with open("reports/2d-regression.txt", "w") as file:
    #     file.write(output)


column_dicts[2]["dgr"] = 3

hpf.plot_2d_regression(df_train, None, column_dicts)
print(f"SIZE OF COL DICTS IS: {len(column_dicts)}")


