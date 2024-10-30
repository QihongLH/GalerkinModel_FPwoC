# PACKAGES
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from scipy.interpolate import griddata

# LOCAL FUNCTIONS
from modules.plots.plt_config import plot_body

def plot_snp_1x1(grid, U, D_LIC=[], title=[], limits = [-0.5, 0.5], make_axis_visible = [1, 1], show_colorbar = 1, flag_flow = 'FP', flag_lic=0):

    X = grid['X']
    Y = grid['Y']
    B = grid['B']

    m = np.shape(X)[0]
    n = np.shape(X)[1]

    M = np.reshape(U, (m,n), order='F') # Take u velocity

    cticks = np.linspace(limits[0], limits[1], 3)
    clevels = limits

    if flag_lic:
        import lic

        # Kernel & texture
        size = 500
        points = np.array((X.flatten(), Y.flatten())).T
        X0, Y0 = np.meshgrid(np.linspace(X.min(), X.max(), size), np.linspace(Y.min(), Y.max(), size))

        k = 2
        M_LIC = np.zeros((size, size, k))
        for i in range(k):
            Z = np.reshape(D_LIC[((n * m) * i):((n * m) * (i + 1))], (m, n), order='F')
            M_LIC[:, :, i] = griddata(points, Z.flatten(), (X0, Y0))

        LIC = lic.lic(M_LIC[:, :, 0].T, M_LIC[:, :, 1].T, length=30)

    fig, ax = plt.subplots(1, 1, layout='tight')

    plt.tight_layout()

    cp0 = ax.pcolormesh(X, Y, M, cmap='jet', vmin=clevels[0], vmax=clevels[1])
    if flag_lic:
        cp00 = ax.pcolormesh(X0, Y0, LIC.T, cmap='gray', alpha=0.75)
    cp00 = ax.contourf(X, Y, B, colors='k')

    if title:
        ax.set_title(title)
    ax.axis('scaled')
    ax.set_xlabel('$x/D$')
    ax.set_ylabel('$y/D$')
    ax.set_xlim([np.min(X), np.max(X)])
    ax.set_ylim([np.min(Y), np.max(Y)])

    if show_colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = fig.colorbar(cp0, ax=ax, ticks=cticks, cax=cax)

    if not make_axis_visible[0]:
        ax.set_xticks([])
        ax.set_xlabel('')
    if not make_axis_visible[1]:
        ax.set_yticks([])
        ax.set_ylabel('')

    plot_body(ax, flag_flow)
    plt.tight_layout()
    plt.show()

def plot_snps_2x2(grid, U, D_LIC=[], titles=[], limits=[-0.5, 0.5], make_axis_visible=[1, 1], show_colorbar=1,
                 flag_flow='FP', flag_lic=0):

    X = grid['X']
    Y = grid['Y']
    B = grid['B']

    m = np.shape(X)[0]
    n = np.shape(X)[1]
    p = 4
    nrows, ncols = 2, 2

    M = np.zeros((m,n,p))
    for j in range(p):
        M[:,:,j] = np.reshape(U[j], (m, n), order='F')  # Take u velocity

    cticks = np.linspace(limits[0], limits[1], 3)
    clevels = limits

    if flag_lic:
        import lic

        # Kernel & texture
        size = 500
        points = np.array((X.flatten(), Y.flatten())).T
        X0, Y0 = np.meshgrid(np.linspace(X.min(), X.max(), size), np.linspace(Y.min(), Y.max(), size))

        k = 2
        M_LIC = np.zeros((size, size, k, p))
        LIC = np.zeros((size, size, p))
        for j in range(p):
            for i in range(k):
                Z = np.reshape(D_LIC[p][((n * m) * i):((n * m) * (i + 1))], (m, n), order='F')
                M_LIC[:, :, i, j] = griddata(points, Z.flatten(), (X0, Y0))

            LIC[:, :, j] = lic.lic(M_LIC[:, :, 0, j].T, M_LIC[:, :, 1, j].T, length=30)

    fig, ax = plt.subplots(nrows, ncols, layout='tight')

    plt.tight_layout()

    c = 0
    for i in range(nrows):
        for j in range(ncols):
            cp0 = ax[i, j].pcolormesh(X, Y, M[:, :, c], cmap='jet', vmin=clevels[0], vmax=clevels[1])
            if flag_lic:
                cp00 = ax[i, j].pcolormesh(X0, Y0, LIC[:, :, c].T, cmap='gray', alpha=0.75)
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
    plt.show()