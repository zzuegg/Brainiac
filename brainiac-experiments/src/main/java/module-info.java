/**
 * Experimental research ideas built on Brainiac, in two layers. The {@code experimental.*}
 * packages hold reusable building blocks written as core code (each mirrors the core package it
 * would move to); the {@code experiments.*} packages hold the experiments themselves, one theme
 * per package. Everything here is untested research code whose API may change at any time.
 */
module io.github.zzuegg.brainiac.experiments {
    requires transitive io.github.zzuegg.brainiac.nn;
    requires transitive io.github.zzuegg.brainiac.train;
    requires transitive io.github.zzuegg.brainiac.data;
    requires transitive io.github.zzuegg.brainiac.rl;
    requires transitive io.github.zzuegg.brainiac.evolve;
    requires transitive io.github.zzuegg.brainiac.llm;
    // reusable building blocks, mirroring the core packages they would move to
    exports io.github.zzuegg.brainiac.experimental.autograd;
    exports io.github.zzuegg.brainiac.experimental.data;
    exports io.github.zzuegg.brainiac.experimental.evolve;
    exports io.github.zzuegg.brainiac.experimental.graph;
    exports io.github.zzuegg.brainiac.experimental.llm;
    exports io.github.zzuegg.brainiac.experimental.ml;
    exports io.github.zzuegg.brainiac.experimental.nn;
    exports io.github.zzuegg.brainiac.experimental.nn.diffusion;
    exports io.github.zzuegg.brainiac.experimental.recsys;
    exports io.github.zzuegg.brainiac.experimental.rl;
    exports io.github.zzuegg.brainiac.experimental.sciml;
    exports io.github.zzuegg.brainiac.experimental.tensor;
    exports io.github.zzuegg.brainiac.experimental.train;
    // the experiments: toy tasks, comparisons and demos built from those blocks
    exports io.github.zzuegg.brainiac.experiments.agents;
    exports io.github.zzuegg.brainiac.experiments.attention;
    exports io.github.zzuegg.brainiac.experiments.audio;
    exports io.github.zzuegg.brainiac.experiments.classic;
    exports io.github.zzuegg.brainiac.experiments.context;
    exports io.github.zzuegg.brainiac.experiments.control;
    exports io.github.zzuegg.brainiac.experiments.data;
    exports io.github.zzuegg.brainiac.experiments.evolution;
    exports io.github.zzuegg.brainiac.experiments.exotic;
    exports io.github.zzuegg.brainiac.experiments.generative;
    exports io.github.zzuegg.brainiac.experiments.graphs;
    exports io.github.zzuegg.brainiac.experiments.interp;
    exports io.github.zzuegg.brainiac.experiments.language;
    exports io.github.zzuegg.brainiac.experiments.learning;
    exports io.github.zzuegg.brainiac.experiments.memory;
    exports io.github.zzuegg.brainiac.experiments.multiagent;
    exports io.github.zzuegg.brainiac.experiments.multimodal;
    exports io.github.zzuegg.brainiac.experiments.numerics;
    exports io.github.zzuegg.brainiac.experiments.optim;
    exports io.github.zzuegg.brainiac.experiments.probabilistic;
    exports io.github.zzuegg.brainiac.experiments.reasoning;
    exports io.github.zzuegg.brainiac.experiments.recsys;
    exports io.github.zzuegg.brainiac.experiments.safety;
    exports io.github.zzuegg.brainiac.experiments.sciml;
    exports io.github.zzuegg.brainiac.experiments.selfsupervised;
    exports io.github.zzuegg.brainiac.experiments.sequence;
    exports io.github.zzuegg.brainiac.experiments.synthesis;
    exports io.github.zzuegg.brainiac.experiments.techniques;
    exports io.github.zzuegg.brainiac.experiments.timeseries;
    exports io.github.zzuegg.brainiac.experiments.vision;
}
