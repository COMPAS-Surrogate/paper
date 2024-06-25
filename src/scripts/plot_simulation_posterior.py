import numpy as np
from pygtc import plotGTC
from typing import List
import matplotlib.pyplot as plt
from data_cache import DataCache
import paths

POSTERIOR_URL = 'https://raw.githubusercontent.com/COMPAS-Surrogate/paper_data/main/simulation_posteriors.npz'

LATEX = dict(
    mu_z=r"$\mu_z$",
    sigma_0=r"$\sigma_0$",
    aSF=r"$a_{\rm SF}$",
    dSF=r"$d_{\rm SF}$",
)


class PosteriorDatasets:
    def __init__(self, posteriors: List[np.ndarray], truths: np.ndarray, params: List[str], labels: List[str]):
        self.chains = posteriors
        self.truths = truths
        self.paramNames = [LATEX[p] for p in params]
        self.chainLabels = labels
        self.n_post, self.n_samp, self.n_dim = (len(posteriors), *posteriors[0].shape)
        assert len(truths) == self.n_dim
        assert len(labels) == self.n_post

    @property
    def posterior_matrix(self):
        return np.array(self.chains)

    @property
    def paramRanges(self):
        ranges = np.zeros((self.n_dim, 2))
        for d in range(self.n_dim):
            p = self.posterior_matrix[:, :, d].flatten()
            ranges[d] = np.array([min(p), max(p)])
        return ranges.tolist()

    @classmethod
    def from_npz(cls, fname):
        data = np.load(fname)
        posteriors = [p for p in data["posteriors"]]
        return cls(posteriors, data["truths"], data["params"], data["labels"])

    @classmethod
    def from_cache(cls):
        return cls.from_npz(DataCache(POSTERIOR_URL).fpath)

    def __dict__(self):
        return {
            "chains": self.chains[0:2],
            "chainLabels": self.chainLabels[0:2],
            "truths": self.truths,
            "paramNames": self.paramNames,
            "paramRanges": self.paramRanges,
            "reference": self.chains[2],
        }

if __name__ == '__main__':
    data = PosteriorDatasets.from_cache()
    fig = plotGTC(
        **data.__dict__(),
        filledPlots=True,
        figureSize="MNRAS_page",
        truthLabels="Truth",
        legendMarker="All",
        confLevels=(.6065, .1353, .0111),
        nContourLevels=3,
        reference_label="Reference"
    )
    # turn off all minor ticks
    for ax in fig.axes:
        ax.tick_params(which='minor', length=0)
    fig.savefig(paths.figures / "simulation_posteriors.pdf", bbox_inches="tight", dpi=300)
