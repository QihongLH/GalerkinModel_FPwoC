import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.animation as animation

# LOCAL FUNCTIONS
from modules.plots.plt_config import plot_body

def plot_video_snps_2x2(grid, D, path_out, titles=[], limits=[-0.5, 0.5], make_axis_visible=[1, 1], show_colorbar=1,
                   flag_flow='FP'):
    X = grid['X']
    Y = grid['Y']
    B = grid['B']

    m = np.shape(X)[0]
    n = np.shape(X)[1]
    nt = np.shape(D[0])[1]
    p = 4
    nrows, ncols = 2, 2

    M = np.zeros((m, n, nt, p))
    for i in range(p):
        M[:, :, :, i] = np.reshape(D[i], (m, n, nt), order='F')

    cticks = np.linspace(limits[0], limits[1], 3)
    clevels = limits

    fig, ax = plt.subplots(nrows, ncols, layout='tight')

    plt.tight_layout()

    c = 0
    for i in range(nrows):
        for j in range(ncols):
            cp0 = ax[i, j].pcolormesh(X, Y, M[:, :, 0, c].reshape(m, n), cmap='jet', vmin=clevels[0], vmax=clevels[1])
            cp00 = ax[i, j].contourf(X, Y, B, colors='k')

            c += 1

    def animate(it):
        c = 0
        for i in range(nrows):
            for j in range(ncols):
                ax[i, j].cla()

                cp0 = ax[i, j].pcolormesh(X, Y, M[:, :, it, c].reshape(m, n), cmap='jet', vmin=clevels[0], vmax=clevels[1])
                cp00 = ax[i, j].contourf(X, Y, B, colors='k')

                if titles:
                    ax[i, j].set_title(titles[c])
                ax[i, j].axis('scaled')
                ax[i, j].set_xlim([np.min(X), np.max(X)])
                ax[i, j].set_ylim([np.min(Y), np.max(Y)])

                if (j == (ncols - 1)) and show_colorbar:
                    divider = make_axes_locatable(ax[i, j])
                    cax = divider.append_axes('right', size='5%', pad=0.05)
                    cbar = fig.colorbar(cp0, ax=ax[i, j], ticks=cticks, cax=cax)

                if j == 0:
                    ax[i, j].set_ylabel('$y/D$')
                else:
                    ax[i, j].set_yticks([])
                if i == (nrows - 1):
                    ax[i, j].set_xlabel('$x/D$')
                else:
                    ax[i, j].set_xticks([])

                if not make_axis_visible[0]:
                    ax[i, j].set_xticks([])
                    ax[i, j].set_xlabel('')
                if not make_axis_visible[1]:
                    ax[i, j].set_yticks([])
                    ax[i, j].set_ylabel('')

                plot_body(ax[i, j], flag_flow)
                c += 1

        plt.tight_layout()

    anim = animation.FuncAnimation(fig, animate, frames=nt, interval=200)
    writergif = animation.PillowWriter(fps=2)
    anim.save(path_out, writer=writergif)