package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.Arrays;
import java.util.List;
import java.util.function.Predicate;

/**
 * Adam-mini (Zhang et al. 2024, <a href="https://arxiv.org/abs/2406.16793">arXiv:2406.16793</a>):
 * Adam with one second-moment scalar per <em>block</em> of parameters instead of one per
 * parameter. The Hessian of a network is close to block-diagonal (one block per output
 * neuron, per attention head), and within a block a single learning rate does as well as
 * Adam's per-entry rates, so {@code v} shrinks from {@code n} values to one per block and the
 * optimizer state from {@code 2n} to about {@code n}: about half of AdamW's memory at the same
 * loss.
 *
 * <p>Blocks here: for a weight matrix {@code [in, out]} (the layout of the core
 * {@code Dense}), each group of {@code blockSize} output units (1 means one block per
 * neuron, {@code headSize} gives one per attention head); for any other parameter, the
 * whole tensor. {@code v} is the mean of {@code g^2} over the block. Parameters the
 * {@code dense} predicate selects (the paper keeps embeddings and the output layer dense)
 * get Adam's full second moment.
 */
public final class AdamMini extends SlotOptimizer {

    private final float beta1;
    private final float beta2;
    private final float epsilon;
    private final float weightDecay;
    private final int blockSize;
    private final Predicate<Var> dense;

    /**
     * @param learningRate the step size, as for AdamW
     * @param beta1 the first-moment decay
     * @param beta2 the second-moment decay
     * @param epsilon added to the denominator
     * @param weightDecay the decoupled weight decay
     * @param blockSize output units per block of a matrix, at least 1
     * @param dense which parameters keep Adam's full second moment
     */
    public AdamMini(float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int blockSize,
            Predicate<Var> dense) {
        super(learningRate, "m", "v");
        if (blockSize < 1 || beta1 < 0 || beta1 >= 1 || beta2 < 0 || beta2 >= 1 || weightDecay < 0) {
            throw new IllegalArgumentException("bad Adam-mini settings");
        }
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
        this.weightDecay = weightDecay;
        this.blockSize = blockSize;
        this.dense = dense;
    }

    /**
     * @param learningRate the step size
     * @param weightDecay the decoupled weight decay
     * @return Adam-mini with one block per output neuron, AdamW's usual betas
     */
    public static AdamMini of(float learningRate, float weightDecay) {
        return new AdamMini(learningRate, 0.9f, 0.999f, 1e-8f, weightDecay, 1, p -> false);
    }

    /** The mean of {@code g^2} over each block, in the block-reduced shape {@code v} keeps. */
    private Tensor blockMeanSquare(Var p, Tensor g) {
        Tensor sq = g.square();
        if (dense.test(p)) {
            return sq;
        }
        if (g.rank() == 2 && g.shape().dim(0) > 1 && g.shape().dim(1) % blockSize == 0) {
            int out = g.shape().dim(1);
            Tensor perUnit = sq.mean(0, true);
            return blockSize == 1 ? perUnit : perUnit.reshape(out / blockSize, blockSize).mean(1, true)
                    .reshape(1, out / blockSize);
        }
        return sq.mean().reshape(1);
    }

    /** Spreads a block-reduced {@code v} back over the parameter's shape. */
    private Tensor expand(Tensor v, Tensor g) {
        if (v.shape().equals(g.shape())) {
            return v;
        }
        if (v.rank() == 2 && blockSize > 1) {
            int blocks = v.shape().dim(1);
            Tensor perUnit = v.reshape(blocks, 1).expand(Shape.of(blocks, blockSize))
                    .reshape(1, blocks * blockSize);
            return perUnit.expand(g.shape());
        }
        return v.rank() == 1 ? v.reshape(onesLike(g.rank())).expand(g.shape()) : v.expand(g.shape());
    }

    private static int[] onesLike(int rank) {
        int[] dims = new int[Math.max(rank, 1)];
        Arrays.fill(dims, 1);
        return dims;
    }

    @Override
    protected Tensor[] update(Var p, Tensor g, Tensor[] s, int t) {
        Tensor m = s[0] == null ? g.mul(1 - beta1) : s[0].mul(beta1).add(g.mul(1 - beta1));
        Tensor blockSq = blockMeanSquare(p, g);
        Tensor v = s[1] == null ? blockSq.mul(1 - beta2) : s[1].mul(beta2).add(blockSq.mul(1 - beta2));
        float c1 = 1 - (float) Math.pow(beta1, t);
        float c2 = 1 - (float) Math.pow(beta2, t);
        Tensor denom = expand(v.div(c2).sqrt().add(epsilon), g);
        float lr = learningRate();
        Tensor w = weightDecay > 0 ? p.value().mul(1 - lr * weightDecay) : p.value();
        p.assign(w.sub(m.div(c1).div(denom).mul(lr)));
        return new Tensor[] {m, v};
    }

    /**
     * The bytes an optimizer keeps as state for these parameters, from its
     * {@link Optimizer#state} (the step counter excluded): works for core and experimental
     * optimizers alike, including packed 8-bit state.
     *
     * @param optimizer the optimizer, after at least one step
     * @param parameters its parameters
     * @return the state's size in bytes
     */
    public static long stateBytes(Optimizer optimizer, List<Var> parameters) {
        long bytes = 0;
        for (var e : optimizer.state(parameters).entrySet()) {
            if (!e.getKey().equals("step")) {
                bytes += e.getValue().bytes();
            }
        }
        return bytes;
    }

    @Override
    public String toString() {
        return "Adam-mini(lr=" + learningRate() + ", blockSize=" + blockSize + ")";
    }
}
