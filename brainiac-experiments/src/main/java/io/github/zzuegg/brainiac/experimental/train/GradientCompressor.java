package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;

/**
 * A simulated collective for data-parallel training with compressed gradients: every worker
 * hands in its gradient for one parameter, the compressor encodes them, exchanges the
 * encodings (counting the bytes every worker sends and receives) and returns the aggregate that
 * every worker applies. Implementations keep per-worker state such as error-feedback residuals.
 * {@link PowerSgd}, {@link SparseGradientCompression} and {@link QuantizedGradients} implement
 * it; {@link #exact()} is the uncompressed float32 all-reduce baseline.
 */
public interface GradientCompressor {

    /**
     * Aggregates one parameter's gradients over the workers.
     *
     * @param parameter the parameter (the key for any per-parameter state)
     * @param workerGradients one gradient per worker, same shape, in a fixed worker order
     * @return the gradient every worker applies
     */
    Tensor aggregate(Var parameter, List<Tensor> workerGradients);

    /** @return the bytes sent and received by all workers so far */
    long bytes();

    /**
     * The uncompressed baseline: a float32 ring all-reduce in which every worker sends and
     * receives about the full gradient once, {@code 2 * 4} bytes per value.
     *
     * @return the compressor
     */
    static GradientCompressor exact() {
        return new GradientCompressor() {
            private long bytes;

            @Override
            public Tensor aggregate(Var parameter, List<Tensor> workerGradients) {
                bytes += 8L * workerGradients.get(0).size() * workerGradients.size();
                return mean(workerGradients);
            }

            @Override
            public long bytes() {
                return bytes;
            }

            @Override
            public String toString() {
                return "fp32 all-reduce";
            }
        };
    }

    /**
     * The mean of some tensors.
     *
     * @param tensors the tensors, same shape
     * @return their mean
     */
    static Tensor mean(List<Tensor> tensors) {
        Tensor sum = tensors.get(0);
        for (int i = 1; i < tensors.size(); i++) {
            sum = sum.add(tensors.get(i));
        }
        return sum.div(tensors.size());
    }
}
