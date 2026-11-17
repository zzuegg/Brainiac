package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * Muon with adaptive step sizes, in three modes:
 *
 * <ul>
 *   <li>{@link Mode#NAMO} (noise-adaptive Muon, <a href="https://arxiv.org/abs/2602.17080">arXiv:2602.17080</a>):
 *       the orthogonalised momentum is scaled by an Adam-style signal-to-noise step size,
 *       {@code ||m|| / sqrt(v)} with {@code v} the running mean of {@code ||g||^2}: about
 *       1 when gradients agree from step to step, small when they are mostly noise, so the
 *       step shrinks by itself late in training or at small batch sizes.</li>
 *   <li>{@link Mode#NAMO_D}: the same ratio per output neuron (column), a diagonal
 *       step size that also evens out neurons, like {@link NorMuon} but driven by noise
 *       rather than by the size of the orthogonalised update.</li>
 *   <li>{@link Mode#MALT} (Muon in an adaptively preconditioned space,
 *       <a href="https://arxiv.org/abs/2608.05088">arXiv:2608.05088</a>): Adafactor-style
 *       row and column second moments {@code r} and {@code c} define a diagonal
 *       preconditioner; the momentum is orthogonalised in the preconditioned space,
 *       {@code D_r^-1/2 orth(D_r^-1/2 m D_c^-1/2) D_c^-1/2}, so rows and columns with large
 *       gradients take smaller steps, and the result is rescaled to Muon's Frobenius
 *       size.</li>
 * </ul>
 * Weights are {@code [in, out]}; matrices only.
 */
public final class AdaptiveMuon extends SlotOptimizer {

    /** Which adaptive rule is used. */
    public enum Mode {
        /** A noise-adaptive scalar step size per matrix. */
        NAMO,
        /** A noise-adaptive step size per output neuron. */
        NAMO_D,
        /** Orthogonalisation in a row/column diagonally preconditioned space. */
        MALT
    }

    private final Mode mode;
    private final float weightDecay;
    private final float momentum;
    private final float beta2;
    private final Orthogonalizer orthogonalizer;

    /**
     * @param mode the rule
     * @param learningRate the step size, as for Muon
     * @param weightDecay the decoupled weight decay
     * @param momentum the momentum, 0.95
     * @param beta2 the decay of the second moments, 0.99
     * @param orthogonalizer how the momentum is orthogonalised
     */
    public AdaptiveMuon(Mode mode, float learningRate, float weightDecay, float momentum, float beta2,
            Orthogonalizer orthogonalizer) {
        super(learningRate, "momentum", "rowMoment", "columnMoment");
        if (weightDecay < 0 || momentum < 0 || momentum >= 1 || beta2 < 0 || beta2 >= 1) {
            throw new IllegalArgumentException("bad AdaptiveMuon settings");
        }
        this.mode = mode;
        this.weightDecay = weightDecay;
        this.momentum = momentum;
        this.beta2 = beta2;
        this.orthogonalizer = orthogonalizer;
    }

    /**
     * @param mode the rule
     * @param learningRate the step size
     * @return the optimizer with Polar Express (5 iterations), no weight decay
     */
    public static AdaptiveMuon of(Mode mode, float learningRate) {
        return new AdaptiveMuon(mode, learningRate, 0, 0.95f, 0.99f, PolarExpress.of(5));
    }

    private Tensor ema(Tensor old, Tensor now) {
        return old == null ? now.mul(1 - beta2) : old.mul(beta2).add(now.mul(1 - beta2));
    }

    @Override
    protected Tensor[] update(Var p, Tensor g, Tensor[] s, int t) {
        if (g.rank() != 2) {
            throw new IllegalArgumentException("AdaptiveMuon updates matrices only, got " + g.shape());
        }
        Tensor[] m = OrthogonalMuon.nesterov(g, s[0], momentum);
        float c1 = 1 - (float) Math.pow(momentum, t);
        float c2 = 1 - (float) Math.pow(beta2, t);
        float scale = OrthogonalMuon.shapeScale(g);
        Tensor u;
        Tensor rows = null;
        Tensor cols = null;
        switch (mode) {
            case NAMO -> {
                cols = ema(s[2], g.square().sum().reshape(1, 1));
                Tensor ratio = m[0].div(c1).square().sum().sqrt().div(cols.div(c2).sqrt().add(1e-12f));
                u = orthogonalizer.orthogonalize(m[1]).mul(ratio.minimum(Tensor.scalar(1f))).mul(scale);
            }
            case NAMO_D -> {
                cols = ema(s[2], g.square().sum(0, true));
                Tensor ratio = m[0].div(c1).square().sum(0, true).sqrt().div(cols.div(c2).sqrt().add(1e-12f));
                u = orthogonalizer.orthogonalize(m[1]).mul(ratio.minimum(Tensor.scalar(1f))).mul(scale);
            }
            default -> {
                rows = ema(s[1], g.square().mean(1, true));
                cols = ema(s[2], g.square().mean(0, true));
                Tensor dr = rows.div(rows.mean()).sqrt().add(1e-6f);   // [in, 1], mean 1
                Tensor dc = cols.div(cols.mean()).sqrt().add(1e-6f);   // [1, out]
                Tensor o = orthogonalizer.orthogonalize(m[1].div(dr).div(dc)).div(dr).div(dc);
                float target = (float) Math.sqrt(Math.min(g.shape().dim(0), g.shape().dim(1))) * scale;
                u = o.mul(Tensor.scalar(target).div(o.square().sum().sqrt().add(1e-12f)));
            }
        }
        OrthogonalMuon.apply(p, u, learningRate(), weightDecay, false);
        return new Tensor[] {m[0], rows, cols};
    }

    @Override
    public String toString() {
        return "AdaptiveMuon(" + mode + ", lr=" + learningRate() + ")";
    }
}
