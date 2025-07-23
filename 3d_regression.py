import helper_funcs as hpf

train_data_frame = hpf.read_files(list(range(0, 7)))
df_train = hpf.remove_outliers(train_data_frame)

test_data_frame = hpf.read_files(list(range(7, 10)))
df_test = hpf.remove_outliers(test_data_frame)

best_degree, best_r2 = hpf.find_best_pol_degree_3d(df_train, df_test, ["wiops", "latency"], "dispatch")
print(f"Best polynomial degree: {best_degree} with R² = {best_r2:.4f}")

hpf.plot_3d_regression(df_train, ["wiops", "latency"], "dispatch", degree=best_degree, pred_point=(50000, 0.5))


