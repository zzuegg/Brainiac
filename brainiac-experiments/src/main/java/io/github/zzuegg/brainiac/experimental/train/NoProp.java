package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * NoProp (Li, Teh and Pascanu 2025, <a href="https://arxiv.org/abs/2503.24322">arXiv:2503.24322</a>),
 * the discrete-time variant: a stack of {@code T} blocks, each trained on its own to
 * <em>denoise a label embedding</em>, as in a diffusion model whose "image" is the label.
 * Block {@code t} sees the input {@code x} and a noisy label {@code z_{t-1} = sqrt(a_{t-1})
 * u_y + sqrt(1 - a_{t-1}) eps} and predicts the clean embedding {@code u_y}, with a
 * squared-error loss. No gradient flows between blocks, so they train fully in parallel,
 * with neither a forward nor a backward pass through the whole stack.
 *
 * <p>Inference starts from {@code z_0 = 0} (the prior's mean) and runs the blocks in
 * order, each turning its prediction into the next, less noisy {@code z_t} with a
 * deterministic DDIM step; the last block's prediction is the output. Label embeddings
 * are one-hot vectors; the signal level {@code a_t} rises linearly from 0 to 1.
 */
public final class NoProp implements LocalLearner {

    private final int classes;
    private final int blocks;
    private final List<Var[]> params = new ArrayList<>();   // per block: w1, b1, w2, b2
    private final float[] alphaBar;
    private final Optimizer optimizer;
    private final SplittableRandom random;

    /**
     * @param inputs features per sample
     * @param classes the number of classes
     * @param blocks the number of denoising blocks {@code T}, at least 1
     * @param hidden the hidden width of every block
     * @param optimizer the optimizer for all blocks
     * @param seed the random seed
     */
    public NoProp(int inputs, int classes, int blocks, int hidden, Optimizer optimizer, long seed) {
        if (blocks < 1) {
            throw new IllegalArgumentException("need at least one block");
        }
        this.classes = classes;
        this.blocks = blocks;
        this.optimizer = optimizer;
        this.random = new SplittableRandom(seed);
        SplittableRandom r = new SplittableRandom(seed + 1);
        int in = inputs + classes;
        for (int t = 0; t < blocks; t++) {
            params.add(new Var[] {
                Var.param(Tensor.normal(Shape.of(in, hidden), 0, (float) Math.sqrt(2.0 / in), r)),
                Var.param(Tensor.zeros(Shape.of(1, hidden))),
                Var.param(Tensor.normal(Shape.of(hidden, classes), 0, (float) Math.sqrt(1.0 / hidden), r)),
                Var.param(Tensor.zeros(Shape.of(1, classes)))});
        }
        alphaBar = new float[blocks + 1];
        for (int t = 0; t <= blocks; t++) {
            alphaBar[t] = (float) t / blocks;
        }
    }

    /** @return every block's parameters */
    public List<Var> parameters() {
        List<Var> out = new ArrayList<>();
        for (Var[] p : params) {
            out.addAll(List.of(p));
        }
        return out;
    }

    /** @return the signal level {@code a_t} of every step {@code t = 0..T} */
    public float[] signalLevels() {
        return alphaBar.clone();
    }

    private Var block(int t, Tensor x, Tensor z) {
        Var[] p = params.get(t);
        Var h = Var.of(Tensor.concat(1, x, z)).matmul(p[0]).add(p[1]).relu();
        return h.matmul(p[2]).add(p[3]);
    }

    @Override
    public float trainBatch(Tensor inputs, Tensor labels) {
        int batch = inputs.shape().dim(0);
        Tensor u = LearningMath.oneHot(labels, classes);
        Var total = null;
        for (int t = 1; t <= blocks; t++) {
            float a = alphaBar[t - 1];
            Tensor eps = Tensor.normal(Shape.of(batch, classes), 0, 1, random);
            Tensor z = u.mul((float) Math.sqrt(a)).add(eps.mul((float) Math.sqrt(1 - a)));
            Var loss = block(t - 1, inputs, z).sub(u).square().sum(1, false).mean();
            total = total == null ? loss : total.add(loss);
        }
        total.backward();
        float value = total.value().item() / blocks;
        optimizer.step(parameters());
        return value;
    }

    @Override
    public Tensor predict(Tensor inputs) {
        int batch = inputs.shape().dim(0);
        Tensor z = Tensor.zeros(Shape.of(batch, classes));
        Tensor uHat = z;
        for (int t = 1; t <= blocks; t++) {
            uHat = block(t - 1, inputs, z).value();
            float aPrev = alphaBar[t - 1];
            float a = alphaBar[t];
            Tensor epsHat = aPrev >= 1 ? Tensor.zeros(z.shape())
                    : z.sub(uHat.mul((float) Math.sqrt(aPrev))).div((float) Math.sqrt(1 - aPrev));
            z = uHat.mul((float) Math.sqrt(a)).add(epsHat.mul((float) Math.sqrt(Math.max(0, 1 - a))));
        }
        return uHat;
    }
}
