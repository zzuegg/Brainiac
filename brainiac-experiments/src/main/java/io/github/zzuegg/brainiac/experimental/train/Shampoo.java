package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Linalg;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * Shampoo (Gupta, Koren and Singer 2018, <a href="https://arxiv.org/abs/1802.09568">arXiv:1802.09568</a>):
 * a Kronecker-factored full-matrix preconditioner. For a weight matrix with gradient
 * {@code G} it accumulates {@code L += G G^T} and {@code R += G^T G} and steps along
 * {@code L^{-1/4} G R^{-1/4}}, which captures correlations between rows and between
 * columns at the memory cost of two small matrices. The inverse roots are recomputed every
 * {@code updateEvery} steps. As in distributed Shampoo (Shi et al. 2023,
 * <a href="https://arxiv.org/abs/2309.06497">arXiv:2309.06497</a>) the step's size is
 * grafted from diagonal AdaGrad, so learning rates transfer; vectors and matrices with a
 * side above {@code maxDim} use AdaGrad alone.
 */
public final class Shampoo extends SlotOptimizer {

    private final float momentum;
    private final int updateEvery;
    private final int maxDim;
    private final float epsilon;

    /**
     * @param learningRate the step size, as for AdaGrad, for example 0.05
     * @param momentum the momentum on the preconditioned step, for example 0.9
     * @param updateEvery steps between recomputing the inverse roots
     * @param maxDim the largest side preconditioned with a full matrix
     */
    public Shampoo(float learningRate, float momentum, int updateEvery, int maxDim) {
        super(learningRate, "L", "R", "Linv", "Rinv", "diag", "momentum");
        this.momentum = momentum;
        this.updateEvery = updateEvery;
        this.maxDim = maxDim;
        this.epsilon = 1e-6f;
    }

    @Override
    protected Tensor[] update(Var p, Tensor g, Tensor[] s, int t) {
        Tensor diag = s[4] == null ? g.square() : s[4].add(g.square());
        Tensor graft = g.div(diag.sqrt().add(1e-8f));
        Tensor dir = graft;
        boolean matrix = g.rank() == 2 && g.shape().dim(0) <= maxDim && g.shape().dim(1) <= maxDim;
        if (matrix) {
            Tensor gg = g.matmul(g.transpose());
            Tensor ggT = g.transpose().matmul(g);
            s[0] = s[0] == null ? gg : s[0].add(gg);
            s[1] = s[1] == null ? ggT : s[1].add(ggT);
            if (s[2] == null || (t - 1) % updateEvery == 0) {
                s[2] = inverseFourthRoot(s[0]);
                s[3] = inverseFourthRoot(s[1]);
            }
            Tensor pre = s[2].matmul(g).matmul(s[3]);
            float pn = LearningMath.norm(pre);
            dir = pn > 0 ? pre.mul(LearningMath.norm(graft) / pn) : graft;
        }
        Tensor m = s[5] == null || momentum == 0 ? dir : s[5].mul(momentum).add(dir);
        p.assign(p.value().sub(m.mul(learningRate())));
        return new Tensor[] {s[0], s[1], s[2], s[3], diag, momentum == 0 ? null : m};
    }

    /** {@code (A + epsilon I)^(-1/4)} of a positive semi-definite statistic. */
    private Tensor inverseFourthRoot(Tensor a) {
        Tensor damped = a.add(Linalg.identity(a.shape().dim(0), a.dtype(), a.device()).mul(epsilon));
        return Linalg.symmetricPower(damped, -0.25, epsilon);
    }

    @Override
    public String toString() {
        return "Shampoo(lr=" + learningRate() + ")";
    }
}
