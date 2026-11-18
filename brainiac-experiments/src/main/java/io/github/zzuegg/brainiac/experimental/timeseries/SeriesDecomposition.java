package io.github.zzuegg.brainiac.experimental.timeseries;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.Map;
import java.util.SplittableRandom;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Series decomposition, the block Autoformer (Wu et al., NeurIPS 2021,
 * <a href="https://arxiv.org/abs/2106.13008">arXiv:2106.13008</a>) builds into every layer:
 * the trend is a moving average over time (edges replicate-padded), the seasonal part is the
 * remainder. As a {@link Layer} it returns the seasonal part of {@code [batch, time, features]}
 * (the inner decomposition blocks of Autoformer and FEDformer drop the trend);
 * {@link #decompose} returns both.
 */
public final class SeriesDecomposition implements Layer {

    private final int kernel;
    private final Map<Integer, Tensor> matrices = new ConcurrentHashMap<>();

    /**
     * Creates the block.
     *
     * @param kernel the moving-average width (odd)
     */
    public SeriesDecomposition(int kernel) {
        this.kernel = kernel;
    }

    /**
     * @param kernel the moving-average width
     * @return a spec for model builders
     */
    public static LayerSpec spec(int kernel) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return input;
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return new SeriesDecomposition(kernel);
            }
        };
    }

    /**
     * The {@code [time, time]} matrix {@code A} with {@code trend = A x} along time.
     *
     * @param length the series length
     * @param kernel the moving-average width
     * @return the averaging matrix (row = output step)
     */
    public static Tensor averagingMatrix(int length, int kernel) {
        float[] m = new float[length * length];
        int half = kernel / 2;
        for (int t = 0; t < length; t++) {
            for (int k = -half; k <= half; k++) {
                int src = Math.max(0, Math.min(length - 1, t + k));
                m[t * length + src] += 1f / (2 * half + 1);
            }
        }
        return Tensor.of(Shape.of(length, length), m);
    }

    /**
     * Splits a series.
     *
     * @param x {@code [batch, time, features]}
     * @return {seasonal, trend}, both of the input shape
     */
    public Var[] decompose(Var x) {
        int t = x.shape().dim(1);
        Tensor at = matrices.computeIfAbsent(t, len -> averagingMatrix(len, kernel).transpose());
        Var trend = x.permute(0, 2, 1).matmul(Var.of(at)).permute(0, 2, 1);
        return new Var[] {x.sub(trend), trend};
    }

    @Override
    public Var forward(Var input) {
        return decompose(input)[0];
    }

    @Override
    public String toString() {
        return "SeriesDecomposition(" + kernel + ")";
    }
}
