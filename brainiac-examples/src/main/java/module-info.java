/**
 * Runnable examples showing how to use Brainiac.
 */
module io.github.zzuegg.brainiac.examples {
    requires transitive io.github.zzuegg.brainiac;
    requires io.github.zzuegg.brainiac.backend.opencl;
    requires io.github.zzuegg.brainiac.backend.vulkan;
    requires io.github.zzuegg.brainiac.evolve;
    exports io.github.zzuegg.brainiac.examples;
}
