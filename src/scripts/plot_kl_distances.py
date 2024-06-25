import pandas as pd
import numpy as np
import paths
from data_cache import DataCache
import matplotlib.pyplot as plt

KL_DATA = 'https://raw.githubusercontent.com/COMPAS-Surrogate/paper_data/main/kl_distance_ci_data.csv'


def plot_kl_data():
    data = pd.read_csv(DataCache(KL_DATA).fpath)
    x, y, yu, yl = data.npts, data.medians, data.upper_ci_95, data.lower_ci_95
    x, y, yu, yl = np.array(x.tolist()), y.tolist(), yu.tolist(), yl.tolist()
    fig, axs = plt.subplots(1, 1)
    axs.plot(x, y, label='Median')
    axs.fill_between(x.tolist(), yl, yu, alpha=0.3, label=r'$95\%$ C.I.')
    axs.set_ylabel('JS divergence')
    axs.set_xlabel('Number of GP training Points')
    leg = axs.legend()
    leg.legend_handles[0].set_linewidth(2.0)
    axs.set_xlim(min(data.npts), 900)
    axs.set_ylim(bottom=0)
    plt.tight_layout()
    fig.savefig(paths.figures / "kl_distances.pdf", bbox_inches="tight", dpi=300)


if __name__ == '__main__':
    plot_kl_data()