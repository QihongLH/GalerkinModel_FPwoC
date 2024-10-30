import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt

def plotConfig():

    mpl.rcParams['lines.linewidth'] = 1.3
    mpl.rcParams['lines.color'] = 'b'
    mpl.rcParams['lines.markersize'] = 7
    mpl.rcParams['axes.linewidth'] = 1.3
    mpl.rcParams['axes.facecolor'] = '#e6e6e6'
    mpl.rcParams['axes.edgecolor'] = 'black'
    mpl.rcParams['axes.labelcolor'] = 'black'
    mpl.rcParams['xtick.color'] = 'black'
    mpl.rcParams['ytick.color'] = 'black'
    mpl.rcParams['axes.titlecolor'] = 'black'
    mpl.rcParams['legend.facecolor'] = 'white'
    mpl.rcParams['legend.edgecolor'] = 'black'
    mpl.rcParams['legend.loc'] = 'upper left'
    mpl.rcParams['legend.fontsize'] = 12
    mpl.rcParams['axes.titlecolor'] = 'black'
    mpl.rcParams['font.family'] = 'Latex'
    mpl.rcParams['font.size'] = 11
    mpl.rcParams.update({"text.usetex": True, "font.family": "Times New Roman"})

def plot_body(ax, flag_flow):

    if flag_flow == 'FP':

        R = 0.5

        xF, yF = -3 / 2 * np.cos(30 * np.pi / 180), 0
        xB, yB = 0, -3 / 4
        xT, yT = 0, 3 / 4

        circle1 = plt.Circle((xF, yF), R, color='k')
        circle2 = plt.Circle((xB, yB), R, color='k')
        circle3 = plt.Circle((xT, yT), R, color='k')

        ax.add_patch(circle1)
        ax.add_patch(circle2)
        ax.add_patch(circle3)