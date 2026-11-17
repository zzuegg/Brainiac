package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.random.RandomGenerator;

/**
 * The Hyena operator: an attention replacement built from long convolutions and gating.
 * The input is projected (and passed through a short depthwise convolution) into a value
 * {@code v} and {@code order} gates {@code x_1..x_N}; then
 *
 * <pre>
 * z = v;   z = x_i * (h_i conv z)   for i = 1..N;   y = z W_out
 * </pre>
 *
 * <p>where each {@code h_i} is a causal filter as long as the sequence, one per channel. The
 * filters are implicit: a small network maps a positional encoding of {@code t} to the
 * filter value, multiplied by an exponentially decaying window, so the parameter count does
 * not grow with the length. The convolutions run through the FFT in {@code O(L log L)},
 * which is what lets Hyena reach very long contexts; the gates make the operator
 * data-dependent, like attention.
 *
 * <p>Input and output are {@code [batch, steps, features]}; the operator is causal.
 *
 * <p>Paper: Poli et al., "Hyena Hierarchy: Towards Larger Convolutional Language Models",
 * <a href="https://arxiv.org/abs/2302.10866">arXiv:2302.10866</a>.
 */
public final class HyenaOperator implements Layer {

    private static final int BANDS = 4;

    private final int features;
    private final int order;
    private final Var inProj;
    private final Var shortConv;
    private final Var outProj;
    private final List<Var[]> filters = new ArrayList<>();
    private final Var decay;

    /**
     * Creates the operator.
     *
     * @param features the width
     * @param order the number of gated long convolutions, usually 2
     * @param filterHidden the hidden width of the implicit filter network
     * @param random the random source
     */
    public HyenaOperator(int features, int order, int filterHidden, RandomGenerator random) {
        this.features = features;
        this.order = order;
        this.inProj = MixerOps.weight(random, features, (order + 1) * features);
        this.shortConv = MixerOps.uniform(random, 0.5f, 3, (order + 1) * features);
        this.outProj = MixerOps.weight(random, features, features);
        int pe = 1 + 2 * BANDS;
        for (int i = 0; i < order; i++) {
            filters.add(new Var[] {MixerOps.weight(random, pe, filterHidden), MixerOps.uniform(random, 1f, filterHidden),
                MixerOps.weight(random, filterHidden, features)});
        }
        float[] rates = new float[features];
        for (int c = 0; c < features; c++) {
            // decay rates spread so that some channels see far and some near
            rates[c] = (float) Math.log(0.05 + 1.5 * c / Math.max(1, features - 1));
        }
        this.decay = Var.param(Tensor.of(Shape.of(features), rates));
    }

    /**
     * Describes the operator for {@code Model.sequential().add(...)}.
     *
     * @param order the number of gated long convolutions
     * @param filterHidden the hidden width of the filter network
     * @return the spec
     */
    public static LayerSpec of(int order, int filterHidden) {
        return MixerOps.sameShape((in, r) -> new HyenaOperator(in.dim(-1), order, filterHidden, r));
    }

    /** Positional encoding [L, 1 + 2 * BANDS]: t / L and sines and cosines. */
    private static Tensor encoding(int length) {
        int pe = 1 + 2 * BANDS;
        float[] e = new float[length * pe];
        for (int t = 0; t < length; t++) {
            e[t * pe] = t / (float) length;
            for (int k = 0; k < BANDS; k++) {
                double w = 2 * Math.PI * (k + 1) * t / length;
                e[t * pe + 1 + 2 * k] = (float) Math.sin(w);
                e[t * pe + 2 + 2 * k] = (float) Math.cos(w);
            }
        }
        return Tensor.of(Shape.of(length, pe), e);
    }

    /**
     * Returns the implicit filter of one order for a length.
     *
     * @param index which of the {@code order} filters
     * @param length the filter length
     * @return {@code [features, length]}
     */
    public Var filter(int index, int length) {
        Var[] f = filters.get(index);
        Var h = Var.of(encoding(length)).matmul(f[0]).mul(f[1]).sin().matmul(f[2]);   // [L, d]
        float[] steps = new float[length];
        for (int t = 0; t < length; t++) {
            steps[t] = t;
        }
        Var window = decay.exp().reshape(1, features).mul(Tensor.of(Shape.of(length, 1), steps)).neg().exp();
        return h.mul(window).transpose();
    }

    @Override
    public Var forward(Var input) {
        MixerOps.requireSequence(input, features, "HyenaOperator");
        int t = input.shape().dim(1);
        Var proj = MixerOps.causalDepthwise(input.matmul(inProj), shortConv);
        Var z = proj.slice(2, 0, features).permute(0, 2, 1);   // [b, d, L]
        for (int i = 0; i < order; i++) {
            Var gate = proj.slice(2, (i + 1) * features, (i + 2) * features).permute(0, 2, 1);
            z = gate.mul(MixerOps.causalConvolution(z, filter(i, t)));
        }
        return z.permute(0, 2, 1).matmul(outProj);
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(inProj, shortConv, outProj, decay));
        filters.forEach(f -> p.addAll(List.of(f)));
        return p;
    }

    @Override
    public String toString() {
        return "HyenaOperator(" + features + ", order=" + order + ")";
    }
}
