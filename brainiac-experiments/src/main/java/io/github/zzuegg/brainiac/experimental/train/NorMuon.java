package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * NorMuon (Li et al. 2025, "NorMuon: Making Muon more efficient and scalable",
 * <a href="https://arxiv.org/abs/2510.05491">arXiv:2510.05491</a>): Muon followed by a
 * neuron-wise Adam-style normalisation. Orthogonalisation makes the update well
 * conditioned as a matrix, but its per-neuron norms still vary widely, so a few neurons
 * take most of the step. NorMuon keeps one second moment per output neuron, the running
 * mean of the squared orthogonalised update over that neuron's weights, divides each
 * neuron's update by its root, and rescales the whole update back to its Frobenius norm
 * before normalisation (or, with {@code matchAdamW}, to AdamW's typical RMS of 0.2, so
 * AdamW's learning rate can be reused). Memory: one vector per matrix on top of Muon.
 *
 * <p>Weights here are {@code [in, out]} ({@code x W}), so a neuron is a column.
 */
public final class NorMuon extends SlotOptimizer {

    private final float weightDecay;
    private final float momentum;
    private final float beta2;
    private final Orthogonalizer orthogonalizer;
    private final boolean matchAdamW;

    /**
     * @param learningRate the step size, as for Muon (or AdamW's with {@code matchAdamW})
     * @param weightDecay the decoupled weight decay
     * @param momentum the momentum, 0.95
     * @param beta2 the decay of the neuron-wise second moment, 0.95
     * @param orthogonalizer how the momentum is orthogonalised
     * @param matchAdamW whether to scale updates to RMS 0.2 instead of Muon's size
     */
    public NorMuon(float learningRate, float weightDecay, float momentum, float beta2, Orthogonalizer orthogonalizer,
            boolean matchAdamW) {
        super(learningRate, "momentum", "neuronSecondMoment");
        if (weightDecay < 0 || momentum < 0 || momentum >= 1 || beta2 < 0 || beta2 >= 1) {
            throw new IllegalArgumentException("bad NorMuon settings");
        }
        this.weightDecay = weightDecay;
        this.momentum = momentum;
        this.beta2 = beta2;
        this.orthogonalizer = orthogonalizer;
        this.matchAdamW = matchAdamW;
    }

    /**
     * NorMuon at Muon's step size with Polar Express (5 iterations) and no weight decay.
     *
     * @param learningRate the step size
     * @return the optimizer
     */
    public static NorMuon of(float learningRate) {
        return new NorMuon(learningRate, 0, 0.95f, 0.95f, PolarExpress.of(5), false);
    }

    @Override
    protected Tensor[] update(Var p, Tensor g, Tensor[] s, int t) {
        if (g.rank() != 2) {
            throw new IllegalArgumentException("NorMuon updates matrices only, got " + g.shape());
        }
        Tensor[] m = OrthogonalMuon.nesterov(g, s[0], momentum);
        Tensor o = orthogonalizer.orthogonalize(m[1]);
        Tensor sq = o.square().mean(0, true);   // [1, out]: per neuron
        Tensor v = s[1] == null ? sq.mul(1 - beta2) : s[1].mul(beta2).add(sq.mul(1 - beta2));
        Tensor vHat = v.div(1 - (float) Math.pow(beta2, t));
        Tensor n = o.div(vHat.sqrt().add(1e-8f));
        Tensor target = matchAdamW
                ? Tensor.scalar((float) (0.2 * Math.sqrt((double) g.size())))
                : o.square().sum().sqrt().mul(OrthogonalMuon.shapeScale(g));
        Tensor u = n.mul(target.div(n.square().sum().sqrt().add(1e-12f)));
        OrthogonalMuon.apply(p, u, learningRate(), weightDecay, false);
        return new Tensor[] {m[0], v};
    }

    /**
     * The spread of per-neuron update norms of a matrix update: the ratio of the largest
     * to the median column norm.
     *
     * @param update an update {@code [in, out]}
     * @return max / median column norm
     */
    public static float neuronSpread(Tensor update) {
        float[] norms = update.square().sum(0, false).sqrt().toArray();
        java.util.Arrays.sort(norms);
        float median = norms[norms.length / 2];
        return norms[norms.length - 1] / Math.max(median, 1e-12f);
    }

    @Override
    public String toString() {
        return "NorMuon(lr=" + learningRate() + ")";
    }
}
