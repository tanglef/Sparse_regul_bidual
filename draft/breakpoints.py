"""
Section 3 plot with breakpoints
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.join(path, "..")) # to find utils from draft
from utils import save_fig
sns.set()


def G1(x, m, alpha):
    return m * np.abs(x - alpha)

def G2(x, m, beta):
    return 1 - m * np.abs(x - beta)


def G(x, m, n, split):
    alpha = - n / m
    beta = (1 - n) / m
    if split:
        return .5 * G1(x, m, alpha) + .5 * G2(x, m, beta)
    else:
        if x <= alpha:
            return 0
        elif x >= beta:
            return 1
        else:
            return m * x + n


if __name__ == "__main__":
    m, n = 1, -1
    alpha, beta = - n / m, (1 - n) / m 
    x = np.linspace(-2, 4, 100)

    Gx = [G(x_i, m, n, split=False) for x_i in x]
    Gx_split = [G(x_i, m, n, split=True) for x_i in x]

    plt.figure()
    plt.plot(x, Gx, label="G", color='k', linestyle=':', linewidth=3)
    plt.plot(x, Gx_split, label="G1 + G2", color="red")
    plt.plot(x, G1(x, m, alpha), label="G1", color='blue')
    plt.plot(x, G2(x, m, beta), label="G2", color='green')
    plt.legend()
    plt.title("Decomposition of a linear piecewise function with two breakpoints")
    plt.tight_layout()
    plt.savefig(save_fig(path, "decomposition", "pdf"))
    plt.show()