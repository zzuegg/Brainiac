/**
 * The default backend: runs all tensor operations on the CPU in plain Java.
 */
module io.github.zzuegg.brainiac.backend.cpu {
    requires io.github.zzuegg.brainiac.tensor;

    provides io.github.zzuegg.brainiac.tensor.spi.Backend
            with io.github.zzuegg.brainiac.backend.cpu.CpuBackend;
}
