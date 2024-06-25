from ogc4_interface.population_mcz import PopulationMcZ
import paths

if __name__ == '__main__':
    pop = PopulationMcZ.load()
    fig, axes = pop.plot_event_mcz_estimates()
    # remve axes minor ticks along y
    for ax in fig.get_axes():
        ax.tick_params(axis='y', which='minor', length=0)
    fig.savefig(paths.figures / "ogc4_events.pdf", bbox_inches="tight", dpi=70)