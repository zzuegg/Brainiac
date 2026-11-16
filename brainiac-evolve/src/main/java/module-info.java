/**
 * Neuroevolution: searches for network designs (wiring, neuron counts, per-neuron
 * activations, hyperparameters) with a genetic algorithm, scoring each candidate by
 * training it briefly.
 */
module io.github.zzuegg.brainiac.evolve {
    requires transitive io.github.zzuegg.brainiac.train;
    exports io.github.zzuegg.brainiac.evolve;
}
