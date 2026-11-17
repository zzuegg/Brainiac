package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.random.RandomGenerator;

/**
 * Stochastic depth ("drop path"): while training, drops a residual branch's whole output
 * for randomly chosen samples and scales the kept ones by {@code 1 / (1 - rate)}, so each
 * sample trains a randomly thinner network; at inference every branch runs. Put it on the
 * branch, before the residual add: {@code x + dropPath(f(x))}. The standard regularizer of
 * ViT, DeiT, Swin and ConvNeXt, usually with a rate that grows linearly with depth
 * ({@link #linearRate}).
 *
 * <p>Paper: Huang et al., "Deep Networks with Stochastic Depth" (2016),
 * <a href="https://arxiv.org/abs/1603.09382">arXiv:1603.09382</a>.
 */
public final class DropPath implements Layer {

    private final float rate;
    private final RandomGenerator random;

    /**
     * Creates the layer.
     *
     * @param rate the probability of dropping a sample's branch, in {@code [0, 1)}
     * @param random the random source
     */
    public DropPath(float rate, RandomGenerator random) {
        if (!(rate >= 0 && rate < 1)) {
            throw new IllegalArgumentException("rate must be in [0, 1), got " + rate);
        }
        this.rate = rate;
        this.random = random;
    }

    /**
     * Describes the layer for {@code Model.sequential().add(...)}.
     *
     * @param rate the drop probability
     * @return the spec
     */
    public static LayerSpec spec(float rate) {
        return TechniqueOps.sameShape((in, r) -> new DropPath(rate, r.split()));
    }

    /**
     * Drops whole samples of a branch output.
     *
     * @param branch the branch output, {@code [batch, ...]}
     * @param rate the drop probability
     * @param training whether to drop (at inference this returns {@code branch})
     * @param random the random source
     * @return the branch with dropped samples zeroed and kept ones rescaled
     */
    public static Var apply(Var branch, float rate, boolean training, RandomGenerator random) {
        if (!training || rate == 0) {
            return branch;
        }
        int[] dims = branch.shape().toArray();
        int batch = dims[0];
        float[] keep = new float[batch];
        for (int i = 0; i < batch; i++) {
            keep[i] = random.nextFloat() < rate ? 0f : 1f / (1 - rate);
        }
        int[] maskDims = new int[dims.length];
        java.util.Arrays.fill(maskDims, 1);
        maskDims[0] = batch;
        return branch.mul(Tensor.of(Shape.of(maskDims), keep, branch.value().device()));
    }

    /**
     * The usual depth schedule: block {@code i} of {@code blocks} drops with
     * {@code maxRate * i / (blocks - 1)}, so early blocks almost always run.
     *
     * @param block the block index, from 0
     * @param blocks the number of blocks
     * @param maxRate the rate of the last block
     * @return the rate of this block
     */
    public static float linearRate(int block, int blocks, float maxRate) {
        return blocks <= 1 ? maxRate : maxRate * block / (blocks - 1);
    }

    @Override
    public Var forward(Var input) {
        return input;
    }

    @Override
    public Var forward(Var input, boolean training) {
        return apply(input, rate, training, random);
    }

    @Override
    public String toString() {
        return "DropPath(" + rate + ")";
    }
}
