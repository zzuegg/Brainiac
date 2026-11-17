/**
 * Brainiac: a neural network library written from scratch. Requiring this module gives
 * access to the whole public API, with the CPU backend included.
 */
module io.github.zzuegg.brainiac {
    requires transitive io.github.zzuegg.brainiac.tensor;
    requires transitive io.github.zzuegg.brainiac.autograd;
    requires transitive io.github.zzuegg.brainiac.nn;
    requires transitive io.github.zzuegg.brainiac.train;
    requires transitive io.github.zzuegg.brainiac.data;
    requires transitive io.github.zzuegg.brainiac.io;
    requires transitive io.github.zzuegg.brainiac.llm;
    requires transitive io.github.zzuegg.brainiac.audio;
    requires transitive io.github.zzuegg.brainiac.rl;
    requires transitive io.github.zzuegg.brainiac.evolve;
    requires io.github.zzuegg.brainiac.backend.cpu;

    exports io.github.zzuegg.brainiac;
}
