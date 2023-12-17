import pandas as pd

def is_reverse(key):
    if key in ["$time$","$RMSE$"]:
        return False
    return True

def get_dataset(rows, columns):
    if columns == 66:
        return "downsampled"
    if rows == 21782:
        return "original"
    return "truncated"

df = pd.read_csv("results/final/results.csv")
df.drop(inplace=True, columns=["selected_features"])
time = {}
r2 = {}
rmse = {}

dataset = None
data_time = {}
data_r2 = {}
data_rmse = {}
algorithm = None

for index, row in df.iterrows():
    dataset = get_dataset(row["rows"], row["columns"])

    if dataset not in data_time:
        data_time[dataset] = {}
        data_r2[dataset] = {}
        data_rmse[dataset] = {}

    t = str(row["target_size"])

    if t not in data_time[dataset]:
        data_time[dataset][t] = {}
        data_r2[dataset][t] = {}
        data_rmse[dataset][t] = {}

    algorithm = row["algorithm"]

    data_time[dataset][t][algorithm] = row["time"]
    data_r2[dataset][t][algorithm] = row["r2_test"]
    data_rmse[dataset][t][algorithm] = row["rmse_test"]

data = {
    "$time$" : data_time,
    "$R^2$" : data_r2,
    "$RMSE$" : data_rmse,
}

print(r"\begin{tabular}")
print(r"{|l|l|r|r|r|r|r|}\hline Metric & Dataset & t  & MI & SFS & LASSO & FSDR \\ \hline")

for metric, metric_data in data.items():
    print(r"\multirow{15}{*}{" + metric + r"} ")
    for index, (dataset, dataset_data) in enumerate(metric_data.items()):
        print("\t" + r" & \multirow{5}{*}{"+dataset+"}")
        first = True
        for t,t_data in dataset_data.items():
            if not first:
                print(" &\t ", end="")
            else:
                print(" \t ", end="")
            first = False
            print(f" & {t} ", end="")
            keys = list(t_data.keys())
            custom_order = ['mi', 'sfs', 'lasso', 'fsdr']
            sorted_keys = sorted(keys, key=lambda x: custom_order.index(x))
            vals = []
            for key in sorted_keys:
                value = t_data[key]
                vals.append(value)
            vals = sorted(vals, reverse=is_reverse(metric))
            for key in sorted_keys:
                value = t_data[key]
                formatted_number = "{:.2f}".format(value)
                if value == vals[0]:
                    print(r" & \textbf{"+formatted_number+r"} ", end="")
                elif value == vals[1]:
                    print(r" & \textcolor{blue}{" + formatted_number + r"} ", end="")
                else:
                    print(f" & {formatted_number} ", end="")
            print(r"\\")
        if index != len(metric_data)-1:
            print(r"\cline{2-7}")
        else:
            print("\hline")
print("")
print(r"\end{tabular}")