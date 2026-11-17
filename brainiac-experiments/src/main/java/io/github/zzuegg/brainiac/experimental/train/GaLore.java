package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Linalg;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * GaLore (Zhao et al. 2024, <a href="https://arxiv.org/abs/2403.03507">arXiv:2403.03507</a>):
 * full-parameter training with Adam state of low rank. Every {@code updateEvery} steps the
 * gradient {@code G} ({@code [m, n]}) of each weight matrix gives a projector {@code P}, its
 * top-{@code rank} singular vectors on the shorter side. Adam runs on the projected
 * gradient {@code P^T G} ({@code [rank, n]}) and the step is projected back,
 * {@code w -= lr * scale * P adam(P^T G)}. Optimizer memory drops from {@code 2mn} to
 * about {@code 2 rank n + m rank}, yet every weight still changes, and the subspace moves
 * over time. Vectors and matrices whose shorter side is at most the rank use plain Adam.
 */
public final class GaLore extends SlotOptimizer {

    private final int rank;
    private final int updateEvery;
    private final float scale;
    private final float beta1 = 0.9f;
    private final float beta2 = 0.999f;
    private final float weightDecay;

    /**
     * @param learningRate the step size, as for Adam
     * @param rank the projection rank
     * @param updateEvery steps between projector updates, for example 200
     * @param scale the factor on the projected-back step, for example 0.25 (paper) or 1
     * @param weightDecay the decoupled weight decay
     */
    public GaLore(float learningRate, int rank, int updateEvery, float scale, float weightDecay) {
        super(learningRate, "m", "v", "projector");
        this.rank = rank;
        this.updateEvery = updateEvery;
        this.scale = scale;
        this.weightDecay = weightDecay;
    }

    /**
     * How many optimizer-state values a parameter of this shape needs, for comparison with
     * Adam's {@code 2 m n}.
     *
     * @param rows the parameter's rows
     * @param cols its columns
     * @return the state size
     */
    public long stateSize(int rows, int cols) {
        if (Math.min(rows, cols) <= rank) {
            return 2L * rows * cols;
        }
        boolean left = rows <= cols;
        long projected = (long) rank * (left ? cols : rows);
        return 2 * projected + (long) rank * (left ? rows : cols);
    }

    @Override
    protected Tensor[] update(Var p, Tensor g, Tensor[] s, int t) {
        boolean lowRank = g.rank() == 2 && Math.min(g.shape().dim(0), g.shape().dim(1)) > rank;
        boolean left = lowRank && g.shape().dim(0) <= g.shape().dim(1);
        Tensor proj = s[2];
        if (lowRank && (proj == null || (t - 1) % updateEvery == 0)) {
            Tensor gram = left ? g.matmul(g.transpose()) : g.transpose().matmul(g);
            int size = gram.shape().dim(0);
            proj = Linalg.eigh(gram)[1].slice(1, size - rank, size);   // [m, r] or [n, r], the leading eigenvectors
        }
        Tensor r = !lowRank ? g : left ? proj.transpose().matmul(g) : g.matmul(proj);
        Tensor m = s[0] == null ? r.mul(1 - beta1) : s[0].mul(beta1).add(r.mul(1 - beta1));
        Tensor v = s[1] == null ? r.square().mul(1 - beta2) : s[1].mul(beta2).add(r.square().mul(1 - beta2));
        Tensor n = m.div(1 - (float) Math.pow(beta1, t)).div(v.div(1 - (float) Math.pow(beta2, t)).sqrt().add(1e-8f));
        Tensor step = !lowRank ? n : (left ? proj.matmul(n) : n.matmul(proj.transpose())).mul(scale);
        Tensor w = weightDecay > 0 ? p.value().mul(1 - learningRate() * weightDecay) : p.value();
        p.assign(w.sub(step.mul(learningRate())));
        return new Tensor[] {m, v, lowRank ? proj : null};
    }

    @Override
    public String toString() {
        return "GaLore(lr=" + learningRate() + ", rank=" + rank + ")";
    }
}
