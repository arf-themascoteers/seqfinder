import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
from PIL import Image
matplotlib.use("TkAgg")
import my_utils

num_bands = 0
num_seq = 0
line = None
points = []
df = None
signal = None
title = None
colour = my_utils.get_colours()
ani = None
original_size= None
line = None


def update(frame):
    title.set_text(f'Epoch {frame + 1}, R^2')
    bands = get_bands(frame + 1)
    for i in range(num_bands):
        for j in range(num_seq):
            band_value = bands[num_seq*i + j]
            signal_value = signal[band_value]
            points[num_seq*i + j].set_offsets([band_value, signal_value])
    return points,


def get_bands(row):
    bands = []
    for i in range(1, num_bands+1):
        for j in range(1, num_seq+1):
            bands.append(df.iloc[row][f"band_{i}_{j}"])
    return bands


def count_bands():
    band_no = []
    seq_no = []
    for item in df.columns:
        if item.startswith("band_"):
            parts = item.split("_")
            band_no.append(parts[1])
            seq_no.append(parts[2])
    return len(set(band_no)), len(set(seq_no))


def draw_init():
    global title, ani, line
    fig = plt.figure(figsize=(8, 4))
    line = plt.plot(signal)
    bands = get_bands(0)
    for i in range(num_bands):
        for j in range(num_seq):
            band_value = bands[num_seq*i + j]
            signal_value = signal[band_value]
            point = plt.scatter(band_value,signal_value, color=colour[i],s=50)
            points.append(point)
    plt.tight_layout()
    title = plt.title('Epoch 0')
    plt.subplots_adjust(top=0.85)
    ani = FuncAnimation(fig, update, frames=1499, interval=100, repeat=False)


def animate(file, save=False):
    global df, signal, title, line, num_bands, num_seq, original_size
    df = pd.read_csv(file)
    num_bands, num_seq = count_bands()
    original_size = df.iloc[0]["original_size"]
    signal = get_sample_signal()
    draw_init()
    if save:
        ani.save('animation.gif', writer='pillow', fps=10)
    else:
        plt.show()
    print("done")


def get_sample_signal():
    file = "dataset_4200_871.csv"
    if original_size == 525:
        file = "dataset_525_871.csv"
    elif original_size == 66:
        file = "dataset_66_21782.csv"
    file = f"../data/{file}"
    band_df = pd.read_csv("../data/dataset_4200_871.csv")
    return band_df.iloc[99][0:-1]


if __name__ == "__main__":
    animate("../results/fsdr-20-1702933718238.csv", False)

