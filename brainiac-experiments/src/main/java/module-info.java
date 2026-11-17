/**
 * Experimental research ideas built on Brainiac. Each package collects one theme; everything
 * here is untested research code whose API may change at any time.
 */
module io.github.zzuegg.brainiac.experiments {
    requires transitive io.github.zzuegg.brainiac.nn;
    requires transitive io.github.zzuegg.brainiac.train;
    requires transitive io.github.zzuegg.brainiac.data;
    requires transitive io.github.zzuegg.brainiac.rl;
    requires transitive io.github.zzuegg.brainiac.evolve;
    requires transitive io.github.zzuegg.brainiac.llm;
    exports io.github.zzuegg.brainiac.experiments.agents;
    exports io.github.zzuegg.brainiac.experiments.attention;
    exports io.github.zzuegg.brainiac.experiments.context;
    exports io.github.zzuegg.brainiac.experiments.evolution;
    exports io.github.zzuegg.brainiac.experiments.exotic;
    exports io.github.zzuegg.brainiac.experiments.generative;
    exports io.github.zzuegg.brainiac.experiments.interp;
    exports io.github.zzuegg.brainiac.experiments.learning;
    exports io.github.zzuegg.brainiac.experiments.memory;
    exports io.github.zzuegg.brainiac.experiments.optim;
    exports io.github.zzuegg.brainiac.experiments.reasoning;
    exports io.github.zzuegg.brainiac.experiments.sequence;
}
