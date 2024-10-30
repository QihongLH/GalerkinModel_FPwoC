import numpy as np
import matplotlib.pyplot as plt

def plot_energy_POD(Sigma):

    cum_energy = np.cumsum(Sigma**2) / np.sum(Sigma**2)
    energy = Sigma**2 / np.sum(Sigma**2)

    nr = len(energy)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5), subplot_kw=dict(box_aspect=1))

    ax[0].semilogx(np.arange(1, nr + 1), energy)
    ax[0].set_title('Energy')
    ax[0].set_xlabel('$n_r$')
    ax[0].grid()

    ax[1].semilogx(np.arange(1, nr + 1), cum_energy)
    ax[1].set_title('Cumulative energy')
    ax[1].set_xlabel('$n_r$')
    ax[1].axis([1, nr, 0, 1])

    plt.tight_layout()
    plt.show()

def plot_psi(t, t_IC, X_true, X_GP, X_interp, X_IC, nr, nf):

    if nr > 5:
        nc = 5
        nr = int(np.ceil(nr/5))
    else:
        nc = nr
        nr = 1

    fig, ax = plt.subplots(nr, nc, subplot_kw=dict(box_aspect=1))

    if nr != 1 and nc != 1:
        for i in range(nr):
            for j in range(nc):
                ax[i, j].locator_params(axis='both', nbins=3)

                ax[i, j].plot(t, X_true[:,i * nc + j], 'k-', label='True')
                ax[i, j].plot(t, X_interp[:,i * nc + j], 'g--', label='Interp')
                ax[i, j].plot(t, X_GP[:, i * nc + j], 'b--', label='GP')
                ax[i, j].plot(t_IC, X_IC[:, i * nc + j], 'ro', label='ICs')

                if i == nr-1:
                    ax[i, j].set_xlabel(r'$t/\tau$')
                    ax[i, j].set_xticks([t[0], t[nf]])
                else:
                    ax[i, j].set_xticks([])

                ax[i, j].set_ylabel(r'$a_{'+str(i * nc + j + 1)+'}$')

                ax[i, j].set_xlim([t[0], t[nf]])

    elif nr == 1 and nc != 1:
        for j in range(nc):
            ax[j].locator_params(axis='both', nbins=3)

            ax[j].plot(t, X_true[:, j], 'r-', label='True')
            ax[j].plot(t, X_interp[:, j], 'g--', label='Interp')
            ax[j].plot(t, X_GP[:, j], 'b--', label='GP')
            ax[j].plot(t_IC, X_IC[:, j], 'ro', label='IC')

            ax[j].set_ylabel(r'$a_{'+str(j+1)+'}$')
            ax[j].set_xlabel(r'$t/\tau$')

            ax[j].set_xlim([t[0], t[nf]])
            ax[j].set_xticks([t[0], t[nf]])

    else:
        ax.locator_params(axis='both', nbins=3)

        ax.plot(t, X_true[:,0], 'r-', label='True')
        ax.plot(t, X_interp[:, 0], 'g--', label='Interp')
        ax.plot(t, X_GP[:, 0], 'b--', label='GP')
        ax.plot(t_IC, X_IC[:, 0], 'ro', label='IC')

        ax.set_xlabel(r'$t/\tau$')
        ax.set_ylabel(r'$a_{1}$')

        ax.set_xlim([t[0], t[nf]])
        ax.set_xticks([t[0], t[nf]])

    if nr != 1 and nc != 1:
        ax[-1, -1].legend(labelcolor='black')
    elif nr == 1 and nc != 1:
        ax[-1].legend(labelcolor='black')
    else:
        ax.legend(labelcolor='black')

    plt.tight_layout()
    plt.show()