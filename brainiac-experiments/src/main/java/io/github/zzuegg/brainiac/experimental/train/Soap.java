package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.tensor.SymmetricEigen;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * SOAP (Vyas et al. 2024, <a href="https://arxiv.org/abs/2409.11321">arXiv:2409.11321</a>):
 * "Shampoo with Adam in the preconditioner's eigenbasis". For a weight matrix it keeps
 * Shampoo's factors as running averages, {@code L = b2 L + (1 - b2) G G^T} and likewise
 * {@code R}, and every {@code updateEvery} steps takes their eigenvectors {@code Q_L, Q_R}.
 * Adam then runs on the rotated gradient {@code Q_L^T G Q_R}, where the curvature is
 * closest to diagonal, and the step is rotated back. More stable than Shampoo with only one
 * extra hyperparameter (the update frequency). Vectors and matrices with a side above
 * {@code maxDim} use Adam.
 */
public final class Soap extends SlotOptimizer {

    private final float beta1;
    private final float beta2;
    private final float weightDecay;
    private final int updateEvery;
    private final int maxDim;

    /**
     * @param learningRate the step size, as for Adam, for example {@code 3e-3f}
     * @param beta1 momentum decay, 0.95
     * @param beta2 second-moment and factor decay, 0.95
     * @param weightDecay the decoupled weight decay
     * @param updateEvery steps between eigenbasis updates, for example 10
     * @param maxDim the largest side rotated
     */
    public Soap(float learningRate, float beta1, float beta2, float weightDecay, int updateEvery, int maxDim) {
        super(learningRate, "m", "v", "L", "R", "QL", "QR");
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.weightDecay = weightDecay;
        this.updateEvery = updateEvery;
        this.maxDim = maxDim;
    }

    /**
     * @param learningRate the step size
     * @return SOAP with {@code betas = (0.95, 0.95)}, eigenbasis update every 10 steps
     */
    public static Soap of(float learningRate) {
        return new Soap(learningRate, 0.95f, 0.95f, 0, 10, 512);
    }

    @Override
    protected Tensor[] update(Var p, Tensor g, Tensor[] s, int t) {
        boolean matrix = g.rank() == 2 && g.shape().dim(0) <= maxDim && g.shape().dim(1) <= maxDim;
        Tensor m = s[0] == null ? g.mul(1 - beta1) : s[0].mul(beta1).add(g.mul(1 - beta1));
        Tensor rotatedG = g;
        Tensor rotatedM = m;
        if (matrix) {
            Tensor gg = g.matmul(g.transpose()).mul(1 - beta2);
            Tensor ggT = g.transpose().matmul(g).mul(1 - beta2);
            s[2] = s[2] == null ? gg : s[2].mul(beta2).add(gg);
            s[3] = s[3] == null ? ggT : s[3].mul(beta2).add(ggT);
            if (s[4] == null || (t - 1) % updateEvery == 0) {
                s[4] = SymmetricEigen.of(s[2]).basis();
                s[5] = SymmetricEigen.of(s[3]).basis();
            }
            rotatedG = s[4].transpose().matmul(g).matmul(s[5]);
            rotatedM = s[4].transpose().matmul(m).matmul(s[5]);
        }
        Tensor v = s[1] == null ? rotatedG.square().mul(1 - beta2)
                : s[1].mul(beta2).add(rotatedG.square().mul(1 - beta2));
        float c1 = 1 - (float) Math.pow(beta1, t);
        float c2 = 1 - (float) Math.pow(beta2, t);
        Tensor step = rotatedM.div(c1).div(v.div(c2).sqrt().add(1e-8f));
        if (matrix) {
            step = s[4].matmul(step).matmul(s[5].transpose());
        }
        Tensor w = weightDecay > 0 ? p.value().mul(1 - learningRate() * weightDecay) : p.value();
        p.assign(w.sub(step.mul(learningRate())));
        return new Tensor[] {m, v, s[2], s[3], s[4], s[5]};
    }

    @Override
    public String toString() {
        return "SOAP(lr=" + learningRate() + ")";
    }
}
