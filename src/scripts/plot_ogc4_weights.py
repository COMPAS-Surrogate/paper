from ogc4_interface.population_mcz import PopulationMcZ
import paths

if __name__ == '__main__':
    pop = PopulationMcZ.load()
    pop = pop.filter_events(threshold=0.95)
    fig = pop.plot_weights().get_figure()
    fig.savefig(paths.figures / "ogc4_weights.pdf", bbox_inches="tight", dpi=300)