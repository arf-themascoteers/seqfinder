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
    print(f"Frame {frame}")
    title.set_text(f'Epoch {frame + 1}')
    bands = get_bands(frame + 1)
    track = 0
    for index, val in enumerate(bands):
        val = int(val)
        for i in range(5):
            new_val = int(val + (i * 0.005 * 4200))
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
        '#e06c48', '#5dce87', '#a96a7a', '#df816f', '#77a0bf',
        '#b27bc7', '#5e9a85', '#bf7b77', '#aec573', '#b2977a',
        '#9b7db8', '#a970c1', '#8672c4', '#6d9d9d', '#c092c0',
        '#b1907a', '#9466a3', '#a58fb4', '#6a86bf', '#8eb085',
        '#c89b96', '#a68da6', '#92798e', '#c28d7b', '#877b9b',
        '#b5b479', '#8b719c', '#af7f79', '#a77e59', '#827eb8',
        '#7d8db7', '#877394', '#8e8e84', '#a77d72', '#b08093',
        '#9980b6', '#8179a7', '#bf9f6e', '#8a9e9c', '#8e7cbb',
        '#887c94', '#c08b8b', '#ad9a7a', '#a880a0', '#8d8787',
        '#8c9a9d', '#bfa16e', '#bda075', '#9372bf', '#b5a27e',
        '#9377a7', '#988eb7', '#bf996c', '#9c7d89', '#8e8c92',
        '#907f9b', '#9c8587', '#8d9da4', '#997e95', '#bca36c',
        '#98898d', '#c69f7a', '#9b8b9a', '#b3837d', '#c180a0',
        '#857b91', '#b89386', '#b38876', '#b77a95', '#7d91a7',
        '#b2b876', '#9c88a2', '#b58a7a', '#9e778f', '#7a8cb1',
        '#9b7b82', '#988b91', '#b7817b', '#9d7f8f', '#a77290',
        '#c79380', '#92788f', '#af8482', '#a68a99', '#87748f',
        '#bbad71', '#a780aa', '#9b8491', '#8c89a1', '#8d7f9b',
        '#c8957e', '#bf8c93', '#a6868d', '#b8a477', '#a57c9d',
        '#c4a77d', '#b6959b', '#c48d87', '#a57c97', '#b4958a'
    ]



    df = pd.read_csv("../data/dataset_4200_871.csv")
    signal = df.iloc[0].to_numpy()
    signal = signal[0:-1]
    fig = plt.figure(figsize=(8, 4))
    line = plt.plot(signal)

    scale = 0.005
    bands = get_bands(0)
    for index, val in enumerate(bands):
        val = int(val)
        for i in range(5):
            new_val = int(val + (i*0.005 * 4200))
            point = plt.scatter(new_val,signal[new_val],label=f"Band {index+1}", color=colour[index],s=50)
            points.append(point)

    #plt.legend()
    plt.tight_layout()
    title = plt.title('Epoch 0')


    #ani = FuncAnimation(fig, update, frames=len(res)-1, interval=100)
    plt.subplots_adjust(top=0.85)
    ani = FuncAnimation(fig, update, frames=1499, interval=100, repeat=False)
    plt.show()
    #ani.save('animation.gif', writer='pillow', fps=10)
    print("done")

if __name__ == "__main__":
    animate("../results/fsdr-True-30-1703297907837274.csv")
