from compas_python_utils.cosmic_integration.binned_cosmic_integrator.cosmological_model import CosmologicalModel
from compas_python_utils.cosmic_integration.binned_cosmic_integrator.detection_matrix import DetectionMatrix
from collections  import namedtuple
import paths
import os
import numpy as np
from typing import List
import matplotlib.pyplot as plt
from ogc4_interface.plotting.cmap_generator import CMAP
from scipy.ndimage import gaussian_filter
from data_cache import DataCache
ROOT = "https://github.com/COMPAS-Surrogate/paper_data/raw/main/detection_matrix"


DATA_URLS = [
    f"{ROOT}/mczgrid_h5out_512M_aSF_0.0100_dSF_4.7000_mu_z_-0.2300_sigma_z_0.0000.h5",
    f"{ROOT}/mczgrid_h5out_512M_aSF_0.0100_dSF_6.0000_mu_z_-0.2300_sigma_0_3.0000.h5",
    f"{ROOT}/mczgrid_h5out_512M_aSF_0.0100_dSF_3.0000_mu_z_-0.0100_sigma_0_0.2500.h5"
]


MCZ_MODEL = namedtuple("mcz", ["matrix", "z", "sfr", "dPdlogZ"])


redshift_range = [0, 10.0]
logZ_range = [-12.0, 0] # metallicity
mc_bins = np.linspace(3, 40, 50)
redshift_bins = np.linspace(0, 1, 100)







def load_data()->List[MCZ_MODEL]:
    mcz_data = []
    for url in DATA_URLS:
        fname = DataCache(url).fpath
        det_matrx = DetectionMatrix.from_h5(fname)
        cosmo = CosmologicalModel(**det_matrx.cosmological_parameters)
        mcz_data.append(MCZ_MODEL(det_matrx.rate_matrix, cosmo.redshift, cosmo.sfr, cosmo.dPdlogZ))
    return mcz_data

def plot_mcz_grid(
        colors=["tab:green", "tab:orange", "tab:blue"],
        cmaps=["Greens", "Oranges", "Blues"]
):
    data = load_data()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].set_xlabel(r"$z$")
    axes[0].set_ylabel(r"sSFR []")
    axes[0].set_xlim(*redshift_range)
    axes[1].set_xlabel(r"$z$")
    axes[1].set_ylabel(r"$\log Z$")
    axes[2].set_xlabel(r"$z$")
    axes[2].set_ylabel(r"$\mathcal{M}_{\rm src}\, [M_{\odot}]$")

    for i, (mcz, c) in enumerate(zip(data, colors)):
        axes[0].plot(mcz.z, mcz.sfr, label=f"Model {i}", color=c, lw=2)

        axes[0].legend()


    n_z, n_logZ = data[0].dPdlogZ.T.shape

    MET_GRID, ZLARGE_GRID = np.meshgrid(
        np.linspace(logZ_range[0], logZ_range[1], n_logZ),
        np.linspace(redshift_range[0], redshift_range[1], n_z)
    )
    Z_GRID, MC_GRID = np.meshgrid(redshift_bins, mc_bins)
    for i, (mcz, cmap, c) in enumerate(zip(data, cmaps, colors)):
        _add_contour(
            axes[1], ZLARGE_GRID, MET_GRID, mcz.dPdlogZ.T, cmap, label=f"Model {i}", fill=True,
            levels= [0.75, 0.9,  0.95, 0.997]
        )
        _add_contour(
            axes[2], Z_GRID, MC_GRID, mcz.matrix, cmap, label="", smooth=1.2, alpha=0.6, lines=True, color=c,
            levels= [ 0.9, 0.925, 0.95, 0.997, 0.999]
        )

    plt.savefig(os.path.join(paths.figures, "mcz_grid.pdf"), bbox_inches="tight", dpi=300)


def _add_contour(ax, XX,YY,ZZ, cmap, label=None, fill=True, smooth=0, alpha=0.5, levels= [0.5, 0.68, 0.75, 0.9, 0.925, 0.95, 0.997], lines=False, color='k'):
    ZZ = ZZ / np.max(ZZ)
    levels = np.array([np.percentile(ZZ, l * 100) for l in levels])

    if smooth:
        ZZ = gaussian_filter(ZZ, smooth)

    if fill:
        ax.contourf(XX, YY, ZZ, label=label, levels=levels, alpha=alpha, extend='max',
                     cmap=cmap)
    if lines:
        ax.contour(XX, YY, ZZ, label=label, levels=[levels[-5]],  colors=color, linewidths=1)


if __name__ == "__main__":
    plot_mcz_grid()