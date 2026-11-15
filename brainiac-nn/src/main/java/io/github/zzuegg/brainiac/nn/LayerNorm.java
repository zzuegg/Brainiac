package io.github.zzuegg.brainiac.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;

/**
 * Normalizes each sample over its last dimensions to zero mean and unit variance, then
 * scales and shifts by learned values: the normalization of transformers. With
 * {@code rms} it is RMSNorm, which only divides by the root mean square.
 *
 * <p>Unlike batch normalization it does not depend on the batch, so it behaves the same
 * while training and predicting.
 */
public final class LayerNorm implements Layer {

    private final int[] normalized;
    private final float epsilon;
    private final boolean rms;
    private final Var gamma;
    private final Var beta;

    /**
     * Creates a layer normalization over the trailing dimensions {@code normalized}.
     *
     * @param normalized the shape of the trailing dimensions to normalize over, such as {@code [features]}
     * @param epsilon added to the variance for stability
     * @param scale whether to learn a scale per value
     * @param shift whether to learn a shift per value (ignored for RMSNorm)
     * @param rms {@code true} for RMSNorm
     */
    public LayerNorm(int[] normalized, float epsilon, boolean scale, boolean shift, boolean rms) {
        if (normalized.length == 0) {
            throw new IllegalArgumentException("give the shape of the dimensions to normalize");
        }
        if (!(epsilon > 0)) {
            throw new IllegalArgumentException("epsilon must be positive, got " + epsilon);
        }
        Shape s = Shape.of(normalized);
        this.normalized = normalized.clone();
        this.epsilon = epsilon;
        this.rms = rms;
        this.gamma = scale ? Var.param(Tensor.ones(s)) : null;
        this.beta = shift && !rms ? Var.param(Tensor.zeros(s)) : null;
    }

    /**
     * Layer normalization over the last dimension, with learned scale and shift.
     *
     * @param features the size of the last dimension
     * @return the layer
     */
    public static LayerNorm of(int features) {
        return new LayerNorm(new int[] {features}, 1e-5f, true, true, false);
    }

    /**
     * RMSNorm over the last dimension, with a learned scale.
     *
     * @param features the size of the last dimension
     * @return the layer
     */
    public static LayerNorm rms(int features) {
        return new LayerNorm(new int[] {features}, 1e-6f, true, false, true);
    }

    @Override
    public Var forward(Var input) {
        Shape s = input.shape();
        int k = normalized.length;
        if (s.rank() <= k) {
            throw new IllegalArgumentException(this + " needs a batch dimension in front of " + Shape.of(normalized)
                    + ", got " + s);
        }
        for (int i = 0; i < k; i++) {
            if (s.dim(s.rank() - k + i) != normalized[i]) {
                throw new IllegalArgumentException(this + " expects trailing dimensions " + Shape.of(normalized)
                        + ", got " + s);
            }
        }
        long width = Shape.of(normalized).size();
        int rows = Math.toIntExact(s.size() / width);
        Var x = input.reshape(rows, Math.toIntExact(width));
        Var y;
        if (rms) {
            y = x.div(x.square().mean(1, true).add(epsilon).sqrt());
        } else {
            Var centered = x.sub(x.mean(1, true));
            y = centered.div(centered.square().mean(1, true).add(epsilon).sqrt());
        }
        y = y.reshape(s.toArray());
        if (gamma != null) {
            y = y.mul(gamma);
        }
        if (beta != null) {
            y = y.add(beta);
        }
        return y;
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>();
        if (gamma != null) {
            p.add(gamma);
        }
        if (beta != null) {
            p.add(beta);
        }
        return p;
    }

    @Override
    public Shape outputShape(Shape input) {
        return input;
    }

    @Override
    public String toString() {
        return (rms ? "RMSNorm" : "LayerNorm") + Shape.of(normalized);
    }

    static Config write(LayerNorm l) {
        return Config.builder().put("normalized", l.normalized).put("epsilon", l.epsilon)
                .put("scale", l.gamma != null).put("shift", l.beta != null).put("rms", l.rms).build();
    }

    static LayerNorm read(Config c) {
        return new LayerNorm(c.getInts("normalized"), c.getFloat("epsilon"), c.getBoolean("scale"),
                c.getBoolean("shift"), c.getBoolean("rms"));
    }
}
