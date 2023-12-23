import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
from PIL import Image
matplotlib.use("TkAgg")
import os

num_bands = 0
line = None
points = []
res = None
signal = None
title = None

def update(frame):
    title.set_text(f'Epoch {frame + 1}')
    bands = get_bands(frame + 1)
    track = 0
    for index, val in enumerate(bands):
        val = int(val)
        for i in range(5):
            new_val = int(val + (i * 0.005 * 525))
            points[track].set_offsets([new_val, signal[new_val]])
            track = track + 1
    return points,

def get_bands(row):
    bands = []
    for i in range(num_bands):
        bands.append(res.iloc[row][f"band_{i+1}"])
    return bands


def count_bands():
    return sum(1 for item in res.columns if item.startswith("band_"))


def animate(file):
    global res, signal, title, line, num_bands
    res = pd.read_csv(file)
    num_bands = count_bands()
    colour = [
        '#0000FF', '#FFA500', '#008000', '#FF0000', '#800080',
        '#A52A2A', '#FFC0CB', '#808080', '#808000', '#00FFFF',
        '#00008B', '#FF8C00', '#006400', '#8B0000', '#800080',
        '#A52A2A', '#FFC0CB', '#808080', '#808000', '#008080'
    ]

    df = pd.read_csv("../data/dataset_525_871.csv")
    signal = df.iloc[0].to_numpy()
    signal = signal[0:-1]
    fig = plt.figure(figsize=(8, 4))
    line = plt.plot(signal)

    scale = 0.005
    bands = get_bands(0)
    for index, val in enumerate(bands):
        val = int(val)
        for i in range(5):
            new_val = int(val + (i*0.005 * 525))
            point = plt.scatter(new_val,signal[new_val],label=f"Band {index+1}", color=colour[index],s=10)
            points.append(point)

    #plt.legend()
    plt.tight_layout()
    title = plt.title('Epoch 0')


    #ani = FuncAnimation(fig, update, frames=len(res)-1, interval=100)
    plt.subplots_adjust(top=0.85)
    ani = FuncAnimation(fig, update, frames=1500, interval=100, repeat=False)
    plt.show()
    #ani.save('animation.gif', writer='pillow', fps=10)
    print("done")

if __name__ == "__main__":
    animate("../results/fsdr-True-20-1703287355725959.csv")
