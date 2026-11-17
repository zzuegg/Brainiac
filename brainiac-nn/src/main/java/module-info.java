/**
 * Neural network building blocks: layers, activations, losses, initializers and models.
 */
module io.github.zzuegg.brainiac.nn {
    requires transitive io.github.zzuegg.brainiac.tensor;
    requires transitive io.github.zzuegg.brainiac.autograd;
    exports io.github.zzuegg.brainiac.nn;
    exports io.github.zzuegg.brainiac.nn.diffusion;
}
