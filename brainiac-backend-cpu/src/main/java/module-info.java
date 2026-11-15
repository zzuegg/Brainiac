/**
 * The default backend: runs all tensor operations on the CPU in plain Java.
 *
 * <p>When the {@code jdk.incubator.vector} module is added to the JVM (for example with
 * {@code --add-modules jdk.incubator.vector}), matrix multiplication uses SIMD
 * instructions through the Vector API.
 */
module io.github.zzuegg.brainiac.backend.cpu {
    requires io.github.zzuegg.brainiac.tensor;
    requires static jdk.incubator.vector;

    provides io.github.zzuegg.brainiac.tensor.spi.Backend
            with io.github.zzuegg.brainiac.backend.cpu.CpuBackend;
}
