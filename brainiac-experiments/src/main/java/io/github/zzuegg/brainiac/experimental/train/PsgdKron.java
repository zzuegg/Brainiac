package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Linalg;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.SplittableRandom;

/**
 * PSGD with a Kronecker-factored preconditioner (Li 2015, "Preconditioned Stochastic
 * Gradient Descent", <a href="https://arxiv.org/abs/1512.04202">arXiv:1512.04202</a>; Li
 * 2018, Lie-group preconditioners, <a href="https://arxiv.org/abs/1809.10232">arXiv:1809.10232</a>),
 * in the gradient-whitening form popularised as "PSGD Kron". For a weight matrix
 * {@code G [m, n]} the preconditioner is {@code P = (Q_L^T Q_L) ⊗ (Q_R^T Q_R)} with upper
 * triangular {@code Q_L [m, m]}, {@code Q_R [n, n]}. Unlike Shampoo it never inverts or
 * takes roots of a matrix: {@code Q} is <em>fitted online</em> by one step of gradient
 * descent on the Lie group of triangular matrices per iteration, using a random probe
 * {@code V}:
 *
 * <pre>
 *   A = Q_L G Q_R^T,   B = Q_L^{-T} V Q_R^{-1}
 *   Q_L -= lr_p / ||A A^T + B B^T|| * triu(A A^T - B B^T) Q_L      (Q_R likewise with A^T A, B^T B)
 * </pre>
 * whose fixed point satisfies {@code P E[g g^T] P = E[v v^T]}: {@code P} whitens the
 * gradient, like Adam's {@code 1/sqrt(v)} but with full (Kronecker) correlations. The
 * step is {@code w -= lr * P m} with {@code m} a momentum of the gradient, clipped to an
 * update RMS of 1.1 as in the reference implementation. Vectors, matrices with a side of 1
 * and sides above {@code maxDim} use a diagonal {@code Q}, fitted the same way.
 */
public final class PsgdKron extends SlotOptimizer {

    private final float beta;
    private final float preconditionerRate;
    private final float weightDecay;
    private final int maxDim;
    private final SplittableRandom random;

    /**
     * @param learningRate the step size, as for Adam, for example {@code 1e-3f}
     * @param beta the momentum, 0.9
     * @param preconditionerRate the step size of the preconditioner fit, 0.1
     * @param weightDecay the decoupled weight decay
     * @param maxDim the largest side given a full triangular factor
     * @param seed the seed of the random probes
     */
    public PsgdKron(float learningRate, float beta, float preconditionerRate, float weightDecay, int maxDim,
            long seed) {
        super(learningRate, "m", "QL", "QR");
        if (beta < 0 || beta >= 1 || !(preconditionerRate > 0) || weightDecay < 0 || maxDim < 1) {
            throw new IllegalArgumentException("bad PSGD settings");
        }
        this.beta = beta;
        this.preconditionerRate = preconditionerRate;
        this.weightDecay = weightDecay;
        this.maxDim = maxDim;
        this.random = new SplittableRandom(seed);
    }

    /**
     * @param learningRate the step size
     * @return PSGD Kron with momentum 0.9, preconditioner rate 0.1 and full factors up to 256
     */
    public static PsgdKron of(float learningRate) {
        return new PsgdKron(learningRate, 0.9f, 0.1f, 0, 256, 42);
    }

    private boolean kronecker(Tensor g) {
        return g.rank() == 2 && g.shape().dim(0) > 1 && g.shape().dim(1) > 1 && g.shape().dim(0) <= maxDim
                && g.shape().dim(1) <= maxDim;
    }

    @Override
    protected Tensor[] update(Var p, Tensor g, Tensor[] s, int t) {
        Tensor m = s[0] == null ? g.mul(1 - beta) : s[0].mul(beta).add(g.mul(1 - beta));
        Tensor mHat = m.div(1 - (float) Math.pow(beta, t));
        boolean kron = kronecker(g);
        Tensor ql = s[1];
        Tensor qr = s[2];
        if (ql == null) {
            // scale so that P m starts at about unit size: (mean g^4)^(-1/8) split over the factors
            double scale = Math.pow(Math.max(mHat.square().square().mean().item(), 1e-30), -1.0 / 8);
            if (kron) {
                int rows = g.shape().dim(0);
                int cols = g.shape().dim(1);
                float f = (float) Math.sqrt(scale);
                ql = Linalg.identity(rows, g.dtype(), g.device()).mul(f);
                qr = Linalg.identity(cols, g.dtype(), g.device()).mul(f);
            } else {
                ql = Tensor.full(g.shape(), (float) scale);
            }
        }
        Tensor v = Tensor.normal(g.shape(), 0, 1, random);
        Tensor pm;
        if (kron) {
            Tensor a = ql.matmul(mHat).matmul(qr.transpose());
            Tensor x = Linalg.solveTriangular(ql.transpose(), v, true);                     // Q_L^{-T} V
            Tensor b = Linalg.solveTriangular(qr.transpose(), x.transpose(), true).transpose();  // ... Q_R^{-1}
            ql = fit(ql, a.matmul(a.transpose()), b.matmul(b.transpose()));
            qr = fit(qr, a.transpose().matmul(a), b.transpose().matmul(b));
            pm = ql.transpose().matmul(ql).matmul(mHat).matmul(qr.transpose().matmul(qr));
        } else {
            Tensor a = ql.mul(mHat);
            Tensor b = v.div(ql);
            Tensor t1 = a.square();
            Tensor t2 = b.square();
            float bound = t1.add(t2).max().item();
            ql = ql.sub(t1.sub(t2).mul(ql).mul(preconditionerRate / Math.max(bound, 1e-30f)));
            pm = ql.square().mul(mHat);
        }
        Tensor rms = pm.square().mean().sqrt();
        pm = pm.mul(Tensor.full(Shape.of(), 1.1f).div(rms.maximum(Tensor.full(Shape.of(), 1.1f))));
        float lr = learningRate();
        Tensor w = weightDecay > 0 ? p.value().mul(1 - lr * weightDecay) : p.value();
        p.assign(w.sub(pm.mul(lr)));
        return new Tensor[] {m, ql, kron ? qr : null};
    }

    /** One Lie-group step of a triangular factor towards balancing {@code t1} and {@code t2}. */
    private Tensor fit(Tensor q, Tensor t1, Tensor t2) {
        int n = q.shape().dim(0);
        float bound = spectralLowerBound(t1.add(t2));
        Tensor triu = t1.sub(t2).mul(Tensor.upperTriangle(n, q.dtype(), q.device()));
        return q.sub(triu.matmul(q).mul(preconditionerRate / Math.max(bound, 1e-30f)));
    }

    /**
     * A lower bound of the spectral norm of a symmetric positive semi-definite matrix: a few
     * power iterations started from its largest row.
     *
     * @param a the matrix {@code [n, n]}
     * @return a lower bound of its largest eigenvalue
     */
    static float spectralLowerBound(Tensor a) {
        int n = a.shape().dim(0);
        float[] x = a.toArray();
        int best = 0;
        double bestNorm = -1;
        for (int i = 0; i < n; i++) {
            double s = 0;
            for (int j = 0; j < n; j++) {
                s += (double) x[i * n + j] * x[i * n + j];
            }
            if (s > bestNorm) {
                bestNorm = s;
                best = i;
            }
        }
        double[] u = new double[n];
        for (int j = 0; j < n; j++) {
            u[j] = x[best * n + j];
        }
        double lambda = 0;
        for (int it = 0; it < 4; it++) {
            double norm = 0;
            for (double d : u) {
                norm += d * d;
            }
            norm = Math.sqrt(norm);
            if (norm == 0) {
                return 0;
            }
            double[] next = new double[n];
            double rayleigh = 0;
            for (int i = 0; i < n; i++) {
                double s = 0;
                for (int j = 0; j < n; j++) {
                    s += x[i * n + j] * u[j] / norm;
                }
                next[i] = s;
                rayleigh += s * u[i] / norm;
            }
            lambda = Math.max(lambda, rayleigh);
            u = next;
        }
        return (float) lambda;
    }

    @Override
    public String toString() {
        return "PSGD-Kron(lr=" + learningRate() + ")";
    }
}
