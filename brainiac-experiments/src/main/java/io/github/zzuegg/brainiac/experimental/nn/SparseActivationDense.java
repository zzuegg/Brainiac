package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.Arrays;
import java.util.List;

/**
 * Training-free activation sparsity (TEAL, CATS) for a dense layer: inputs whose magnitude
 * is below a calibrated threshold are set to zero, and at decode time the rows of the weight
 * matrix belonging to those inputs are skipped entirely. Hidden states of LLMs are
 * concentrated around zero, so 40–50% of the inputs can be dropped with little loss in
 * quality; single-token decoding is memory-bound, and every skipped row is weight memory
 * that is never read, which turns sparsity into wall-clock speed-up on CPUs.
 *
 * <p>The threshold for a target sparsity is the matching quantile of {@code |x|} over a
 * calibration set ({@link #threshold}). Applied to the input of every projection this is
 * TEAL; applied only to the input of the FFN's down projection (the gated activations) it
 * is CATS.
 *
 * <p>{@link #forward} computes the masked layer with autograd ops (for evaluating the
 * quality); {@link #decode} is the sparse matrix-vector fast path, which reads only the
 * weight rows of the kept inputs. It is a drop-in for the {@link Dense} it wraps and shares
 * its weights.
 *
 * <p>Papers: Liu et al., "Training-Free Activation Sparsity in Large Language Models" (TEAL),
 * <a href="https://arxiv.org/abs/2408.14690">arXiv:2408.14690</a>; Lee et al., "CATS:
 * Contextually-Aware Thresholding for Sparsity in Large Language Models",
 * <a href="https://arxiv.org/abs/2404.08763">arXiv:2404.08763</a>.
 */
public final class SparseActivationDense implements Layer {

    private final Var weight;
    private final Var bias;
    private final Activation activation;
    private final int inputs;
    private final int units;
    private float threshold;
    private float[] rows;
    private float[] biasValues;
    private long kept;
    private long seen;

    /**
     * Wraps a dense layer, sharing its weights.
     *
     * @param dense the layer
     * @param threshold inputs with {@code |x| <= threshold} are dropped; 0 keeps all
     */
    public SparseActivationDense(Dense dense, float threshold) {
        this.weight = dense.weight();
        this.bias = dense.bias();
        this.activation = dense.activation();
        this.inputs = weight.shape().dim(0);
        this.units = weight.shape().dim(1);
        this.threshold = threshold;
    }

    /**
     * Wraps a dense layer with the threshold that drops a target share of its inputs.
     *
     * @param dense the layer
     * @param sparsity the share of inputs to drop, in {@code [0, 1)}
     * @param calibration typical inputs {@code [..., inputs]}
     * @return the sparse layer
     */
    public static SparseActivationDense calibrated(Dense dense, float sparsity, Tensor calibration) {
        return new SparseActivationDense(dense, threshold(calibration, sparsity));
    }

    /**
     * The {@code sparsity}-quantile of {@code |x|}: the threshold that drops that share of
     * the given inputs.
     *
     * @param inputs calibration inputs
     * @param sparsity the share to drop, in {@code [0, 1)}
     * @return the threshold, 0 for no sparsity
     */
    public static float threshold(Tensor inputs, float sparsity) {
        if (sparsity <= 0) {
            return 0;
        }
        float[] a = inputs.abs().toArray();
        Arrays.sort(a);
        return a[Math.min(a.length - 1, (int) Math.floor(sparsity * a.length))];
    }

    @Override
    public Var forward(Var input) {
        Var x = input;
        if (threshold > 0) {
            Tensor keep = input.value().abs().greater(Tensor.full(Shape.of(), threshold, input.value().device()));
            float[] k = keep.toArray();
            for (float v : k) {
                kept += v > 0 ? 1 : 0;
            }
            seen += k.length;
            x = input.mask(keep);
        }
        return activation.forward(x.matmul(weight).add(bias));
    }

    /**
     * The sparse decode path for one input vector: accumulates {@code x_i W[i, :]} over the
     * kept inputs only, then adds the bias and applies the activation. Uses a snapshot of the
     * weights taken on the first call (see {@link #refresh()}).
     *
     * @param x one input row
     * @return the output row
     */
    public float[] decode(float[] x) {
        if (rows == null) {
            refresh();
        }
        float[] y = biasValues.clone();
        int k = 0;
        for (int i = 0; i < inputs; i++) {
            float v = x[i];
            if (Math.abs(v) <= threshold) {
                continue;
            }
            k++;
            int base = i * units;
            for (int o = 0; o < units; o++) {
                y[o] += v * rows[base + o];
            }
        }
        kept += k;
        seen += inputs;
        if (activation != Activation.IDENTITY) {
            y = activation.forward(Var.of(Tensor.of(Shape.of(1, units), y))).value().toArray();
        }
        return y;
    }

    /** Re-reads the weights for {@link #decode}, after they changed. */
    public void refresh() {
        rows = weight.value().toArray();
        biasValues = bias.value().toArray();
    }

    /** @return the threshold */
    public float threshold() {
        return threshold;
    }

    /** @param t the new threshold; 0 keeps all inputs */
    public void threshold(float t) {
        this.threshold = t;
    }

    /** @return the share of inputs kept since the last {@link #resetCounts()}, 1 if none seen */
    public double density() {
        return seen == 0 ? 1 : kept / (double) seen;
    }

    /** Resets the kept-input counters. */
    public void resetCounts() {
        kept = 0;
        seen = 0;
    }

    @Override
    public List<Var> parameters() {
        return List.of(weight, bias);
    }

    @Override
    public int outputSize(int inputSize) {
        return units;
    }

    @Override
    public String toString() {
        return "SparseActivationDense(" + inputs + " -> " + units + ", threshold=" + threshold + ")";
    }
}
