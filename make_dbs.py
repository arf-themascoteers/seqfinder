import pandas as pd
import spec_utils
import pywt
import matplotlib.pyplot as plt


def make_4200():
    df = pd.read_csv("data/dataset.csv")
    bands = spec_utils.get_wavelengths()
    df = df[bands+["oc"]]
    all_columns = [str(i) for i in range(len(bands))] + ["oc"]
    df.columns = all_columns
    size = len(df)
    df.to_csv(f"data/dataset_4200_{size}.csv", index=False)
    df = df.sample(frac=0.04)
    size = len(df)
    df.to_csv(f"data/dataset_4200_{size}.csv", index=False)


def make_525():
    df = pd.read_csv("data/dataset.csv")
    bands = spec_utils.get_wavelengths()
    df = df[bands+["oc"]]
    all_columns = [str(i) for i in range(len(bands))] + ["oc"]
    df.columns = all_columns

    df2 = pd.DataFrame(columns=[str(i) for i in range(525)]+["oc"])
    for index, row in df.iterrows():
        signal = row.iloc[0:-1]
        short_signal,_,_,_ = pywt.wavedec(signal, 'db1', level=3)
        df2.loc[len(df2)] = list(short_signal) + [row.iloc[-1]]
    df2.to_csv(f"data/dataset_525_{len(df2)}.csv", index=False)
    df2 = df2.sample(frac=0.04)
    size = len(df2)
    df2.to_csv(f"data/dataset_525_{size}.csv", index=False)

def make_66():
    df = pd.read_csv("data/dataset.csv")
    bands = spec_utils.get_wavelengths()
    df = df[bands+["oc"]]
    all_columns = [str(i) for i in range(len(bands))] + ["oc"]
    df.columns = all_columns

    df2 = pd.DataFrame(columns=[str(i) for i in range(66)]+["oc"])
    for index, row in df.iterrows():
        signal = row.iloc[0:-1]
        short_signal,_,_,_,_,_,_ = pywt.wavedec(signal, 'db1', level=6)
        df2.loc[len(df2)] = list(short_signal) + [row.iloc[-1]]
    df2.to_csv(f"data/dataset_66_{len(df2)}.csv", index=False)

def test():
    df = pd.read_csv("data/dataset.csv")
    bands = spec_utils.get_wavelengths()
    s1 = df[bands].iloc[12,:]
    plt.plot(list(range(len(s1))), s1)
    plt.show()
    df2 = pd.read_csv("data/dataset_66_21782.csv")
    s2 = df2.iloc[12,0:66]
    plt.plot(list(range(len(s2))), s2)
    plt.show()
    print(s1.shape)
    print(s2.shape)


if __name__ == "__main__":
    make_4200()
    make_525()
    make_66()
