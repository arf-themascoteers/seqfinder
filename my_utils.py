from sklearn.linear_model import LinearRegression
import torch

def get_internal_model():
    return LinearRegression()


def inverse_sigmoid_torch(x):
    return -torch.log(1.0 / x - 1.0)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_colours():
    return [
        '#FF5733', '#70DB93', '#FF6699', '#FF9966', '#66B2FF',
        '#CC99FF', '#66CC99', '#FF6666', '#CCFF66', '#FFCC66',
        '#9966FF', '#FF66CC', '#9999FF', '#66CCCC', '#FF66FF',
        '#FF9966', '#CC66FF', '#FF9999', '#66A6FF', '#99CC66',
        '#FFCC99', '#FF99CC', '#CC99CC', '#FFCC66', '#BB99FF',
        '#CCCC66', '#9975CC', '#FF9966', '#FFA359', '#7F87B8',
        '#99A6CC', '#A7A7A4', '#FF9966', '#FF6699', '#D279A6',
        '#8C7A9E', '#FF9966', '#FFD966', '#FFCCCC', '#9975CC',
        '#66CC99', '#FFA359', '#CC9966', '#FFCC66', '#CC66CC',
        '#FF9966', '#FF6699', '#8F6F8F', '#D279A6', '#8B7767',
        '#FF9966', '#FF6699', '#E8A89E', '#C9948F', '#A27677',
        '#9966FF', '#FF66CC', '#66A6FF', '#FF9966', '#FFCC99',
        '#66CCCC', '#FF66FF', '#FF6699', '#FFCC99', '#8C7A9E',
        '#8F6F8F', '#FF9966', '#FF6699', '#C9948F', '#66CCCC',
        '#FF6666', '#FFCCCC', '#FF9966', '#CC9966', '#FF6666',
        '#FFCCCC', '#FF9966', '#66CCCC', '#FF9966', '#FFCCCC',
        '#FF6666', '#FFCCCC', '#FF9966', '#FF6699', '#FFCCCC',
        '#FF9966', '#66CCCC', '#FF6666', '#FFCCCC', '#FF9966',
        '#CC9966', '#FF6666', '#FF9966', '#CC99CC', '#FF9966',
        '#FFCCCC', '#CC9966', '#FFCCCC', '#FF9966', '#FFCCCC'
    ]
