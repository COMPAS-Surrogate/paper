import json
import numpy as np
import matplotlib.pyplot as plt
import paths
from data_cache import DataCache
from collections import namedtuple
from typing import List
from scipy.stats import binom
from matplotlib.ticker import FixedLocator

PP_DATA_URL = 'https://raw.githubusercontent.com/COMPAS-Surrogate/paper_data/main/pp_data.json'

PP_DATA = namedtuple(
    'pp_data',
    ['p_value', 'pp', 'label']
)

CONFIDENCE_INTERVAL = [0.68, 0.95, 0.997]
CI_ALPHA = [0.1] * len(CONFIDENCE_INTERVAL)


def load_pp_data() -> List[PP_DATA]:
    with open(DataCache(PP_DATA_URL).fpath, 'r') as f:
        loaded_data = json.load(f)
    data = [PP_DATA(**d) for d in loaded_data]
    return data


def plot_pp_data():
    data = load_pp_data()
    x = np.linspace(0, 1, len(data[0].pp))

    fig, ax = plt.subplots(1, 1)

    for d in data:
        ax.plot(x, d.pp, label=d.label)
    _add_ci_bounds(ax, x, n_posteriors=100)

    ax.set_xlabel('Credible Interval (C.I.)')
    ax.set_ylabel('Fraction of Samples in C.I.')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    leg = ax.legend(
        loc='upper left', fontsize='small', markerscale=2.5,
        labelspacing=0.25,handlelength=1,
        handletextpad=0.33
    )
    for legobj in leg.legend_handles[:4]:
        legobj.set_linewidth(2.0)
    ax.set_aspect('equal')
    ax.yaxis.set_major_locator(FixedLocator([0, 0.5, 1]))
    ax.xaxis.set_major_locator(FixedLocator([0.5, 1]))

    fig.savefig(paths.figures / "pp_plot.pdf", bbox_inches="tight", dpi=300)


def _add_ci_bounds(ax, x, n_posteriors):
    for i, (ci, alpha) in enumerate(zip(CONFIDENCE_INTERVAL, CI_ALPHA)):
        edge_of_bound = (1. - ci) / 2.
        lower = binom.ppf(1 - edge_of_bound, n_posteriors, x) / n_posteriors
        upper = binom.ppf(edge_of_bound, n_posteriors, x) / n_posteriors
        lower[0], lower[-1] = 0, 1
        upper[0], upper[-1] = 0, 1

        label = None if i > 0 else "1,2,3-$\sigma$ CI"
        ax.fill_between(
            x, lower, upper, alpha=alpha, color='k', lw=0, label=label
        )


if __name__ == '__main__':
    plot_pp_data()
