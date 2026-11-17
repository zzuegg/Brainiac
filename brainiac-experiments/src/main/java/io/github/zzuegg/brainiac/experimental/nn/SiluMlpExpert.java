package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.random.RandomGenerator;

/**
 * A two-layer SiLU feed-forward network {@code silu(x W1 + b1) W2} over the last axis: the
 * dense FFN of a transformer block and the expert of the mixtures of experts in this
 * package ({@link FineGrainedMoe}, {@link ExpertChoiceMoe}), so a trained dense FFN can be
 * {@linkplain SparseUpcycling upcycled} into experts.
 */
public final class SiluMlpExpert implements Layer {

    private final Var w1;
    private final Var b1;
    private final Var w2;

    /**
     * Creates the network with Xavier-uniform weights and zero bias.
     *
     * @param in the input width
     * @param hidden the hidden width
     * @param out the output width
     * @param random the random source
     */
    public SiluMlpExpert(int in, int hidden, int out, RandomGenerator random) {
        this(MixerOps.weight(random, in, hidden).value(), Tensor.zeros(Shape.of(hidden)),
                MixerOps.weight(random, hidden, out).value());
    }

    /**
     * Creates the network from given weights (copied into fresh parameters).
     *
     * @param w1 {@code [in, hidden]}
     * @param b1 {@code [hidden]}
     * @param w2 {@code [hidden, out]}
     */
    public SiluMlpExpert(Tensor w1, Tensor b1, Tensor w2) {
        this.w1 = Var.param(Tensor.of(w1.shape(), w1.toArray()));
        this.b1 = Var.param(Tensor.of(b1.shape(), b1.toArray()));
        this.w2 = Var.param(Tensor.of(w2.shape(), w2.toArray()));
    }

    /**
     * Describes a width-keeping FFN for {@code Model.sequential().add(...)}.
     *
     * @param hiddenFactor the hidden width as a multiple of the input width
     * @return the spec
     */
    public static LayerSpec spec(float hiddenFactor) {
        return MixerOps.sameShape((in, r) -> {
            int d = in.dim(-1);
            return new SiluMlpExpert(d, Math.max(1, Math.round(d * hiddenFactor)), d, r);
        });
    }

    @Override
    public Var forward(Var input) {
        return Activation.SILU.forward(input.matmul(w1).add(b1)).matmul(w2);
    }

    /** @return the first weight {@code [in, hidden]} */
    public Var w1() {
        return w1;
    }

    /** @return the hidden bias {@code [hidden]} */
    public Var b1() {
        return b1;
    }

    /** @return the second weight {@code [hidden, out]} */
    public Var w2() {
        return w2;
    }

    @Override
    public List<Var> parameters() {
        return List.of(w1, b1, w2);
    }

    @Override
    public int outputSize(int inputSize) {
        return w2.shape().dim(1);
    }

    @Override
    public String toString() {
        return "SiluMlpExpert(" + w1.shape().dim(0) + " -> " + w1.shape().dim(1) + " -> " + w2.shape().dim(1) + ")";
    }
}
