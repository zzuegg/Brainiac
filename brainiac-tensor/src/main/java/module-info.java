/**
 * Shapes, float32 tensors and the math operations on them.
 *
 * <p>Tensors are handles; their data lives in storage owned by a {@link
 * io.github.zzuegg.brainiac.tensor.spi.Backend}. Backends (CPU now, GPU later) are
 * discovered with {@link java.util.ServiceLoader}.
 */
module io.github.zzuegg.brainiac.tensor {
    exports io.github.zzuegg.brainiac.tensor;
    exports io.github.zzuegg.brainiac.tensor.spi;

    requires java.management;
    requires static jdk.management;

    uses io.github.zzuegg.brainiac.tensor.spi.Backend;
}
