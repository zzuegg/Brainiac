package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;

import java.util.List;
import java.util.random.RandomGenerator;

/**
 * A Canon layer: a short causal depthwise convolution along the steps with a residual
 * connection, {@code y_t = x_t + sum_{j < k} w_j * x_{t-j}} (per channel). Inserted before
 * attention (Canon-A) and before the MLP (Canon-C), it moves information horizontally
 * between neighbouring tokens for almost no compute or parameters, so a single attention
 * layer can do what otherwise needs a previous-token head in an earlier layer (induction,
 * associative recall, multi-hop reasoning). It also gives NoPE models local order.
 *
 * <p>Input and output are {@code [batch, steps, features]}; see
 * {@link ResidualMixerBlock.Spec#canon(int)} to put them in a block.
 *
 * <p>Paper: Allen-Zhu, "Physics of Language Models: Part 4.1, Architecture Design and the
 * Magic of Canon Layers" (2025).
 */
public final class CanonLayer implements Layer {

    private final int features;
    private final Var weight;

    /**
     * Creates the layer.
     *
     * @param features the width
     * @param kernel the kernel length, 4 in the paper
     * @param random the random source
     */
    public CanonLayer(int features, int kernel, RandomGenerator random) {
        if (kernel < 1) {
            throw new IllegalArgumentException("kernel must be at least 1, got " + kernel);
        }
        this.features = features;
        this.weight = MixerOps.uniform(random, 1f / kernel, kernel, features);
    }

    /**
     * Describes the layer for {@code Model.sequential().add(...)}.
     *
     * @param kernel the kernel length
     * @return the spec
     */
    public static LayerSpec of(int kernel) {
        return MixerOps.sameShape((in, r) -> new CanonLayer(in.dim(-1), kernel, r));
    }

    @Override
    public Var forward(Var input) {
        MixerOps.requireSequence(input, features, "CanonLayer");
        return input.add(MixerOps.causalDepthwise(input, weight));
    }

    /** @return the kernel {@code [kernel, features]}; row {@code j} weighs the step {@code j} back */
    public Var weight() {
        return weight;
    }

    @Override
    public List<Var> parameters() {
        return List.of(weight);
    }

    @Override
    public String toString() {
        return "CanonLayer(" + features + ", kernel=" + weight.shape().dim(0) + ")";
    }
}
