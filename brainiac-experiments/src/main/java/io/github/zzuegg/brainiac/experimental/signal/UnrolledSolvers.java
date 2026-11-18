package io.github.zzuegg.brainiac.experimental.signal;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Sparse solvers unrolled into networks for a fixed measurement matrix {@code A}: a
 * {@link Layer} mapping measurements {@code [batch, m]} to estimates {@code [batch, n]}.
 *
 * <ul>
 *   <li>{@link Kind#LISTA}: {@code x_(k+1) = eta(W y + S x_k; theta_k)} with learned shared
 *       {@code W, S} and per-layer thresholds, initialised as ISTA (Gregor and LeCun, ICML
 *       2010).</li>
 *   <li>{@link Kind#LISTA_CP}: coupled weights,
 *       {@code x_(k+1) = eta(x_k + W^T (y - A x_k); theta_k)}, with one learned {@code W}.</li>
 *   <li>{@link Kind#ALISTA}: {@code W} is computed analytically from {@code A} (the closed-form
 *       Frobenius minimiser of {@code ||W^T A||} with {@code diag(W^T A) = 1}, standing in for
 *       ALISTA's coherence minimisation), and only a step size and a threshold per layer are
 *       learned (Liu et al., "ALISTA", ICLR 2019).</li>
 *   <li>{@link Kind#LAMP}: learned AMP, {@code x_(k+1) = eta(x_k + B z_k; theta_k ||z_k|| / sqrt(m))}
 *       with {@code z_k = y - A x_k + (||x_k||_0 / m) z_(k-1)}, learning {@code B} (initialised
 *       to {@code A^T}) and the per-layer threshold multipliers (Borgerding, Schniter and Rangan,
 *       IEEE TSP 2017, <a href="https://arxiv.org/abs/1612.01183">arXiv:1612.01183</a>).</li>
 * </ul>
 */
public final class UnrolledSolvers implements Layer {

    /** The unrolled architecture. */
    public enum Kind {
        /** LISTA with shared W and S. */
        LISTA,
        /** LISTA with coupled weights. */
        LISTA_CP,
        /** analytic weights, learned steps and thresholds. */
        ALISTA,
        /** learned AMP. */
        LAMP
    }

    private final Kind kind;
    private final int layers;
    private final int m;
    private final int n;
    private final Tensor a;
    private final Tensor at;
    private final Var w;
    private final Var s;
    private final Var theta;
    private final Var gamma;

    /**
     * Creates the solver initialised to its classical counterpart.
     *
     * @param kind the architecture
     * @param matrix the measurement matrix {@code [m][n]}
     * @param layers the number of layers
     * @param lambda the initial lasso penalty (thresholds start at {@code lambda / L}; LAMP
     *     starts its multipliers at 1.5)
     */
    public UnrolledSolvers(Kind kind, double[][] matrix, int layers, double lambda) {
        this.kind = kind;
        this.layers = layers;
        this.m = matrix.length;
        this.n = matrix[0].length;
        float[] af = new float[m * n];
        float[] atf = new float[n * m];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                af[i * n + j] = (float) matrix[i][j];
                atf[j * m + i] = (float) matrix[i][j];
            }
        }
        a = Tensor.of(Shape.of(m, n), af);
        at = Tensor.of(Shape.of(n, m), atf);
        double l = SparseRecovery.lipschitz(matrix);
        float[] thetas = new float[layers];
        float[] gammas = new float[layers];
        for (int k = 0; k < layers; k++) {
            thetas[k] = (float) (kind == Kind.LAMP ? 1.5 : lambda / l);
            gammas[k] = (float) (1 / l);
        }
        theta = Var.param(Tensor.of(Shape.of(layers), thetas));
        gamma = kind == Kind.ALISTA ? Var.param(Tensor.of(Shape.of(layers), gammas)) : null;
        switch (kind) {
            case LISTA -> {
                // W = A^T / L as [m, n] for y W; S = I - A^T A / L
                float[] wf = new float[m * n];
                for (int i = 0; i < m * n; i++) {
                    wf[i] = (float) (af[i] / l);
                }
                double[][] g = DenseMatrices.gram(matrix);
                float[] sf = new float[n * n];
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < n; j++) {
                        sf[i * n + j] = (float) ((i == j ? 1 : 0) - g[i][j] / l);
                    }
                }
                w = Var.param(Tensor.of(Shape.of(m, n), wf));
                s = Var.param(Tensor.of(Shape.of(n, n), sf));
            }
            case LISTA_CP -> {
                float[] wf = new float[m * n];
                for (int i = 0; i < m * n; i++) {
                    wf[i] = (float) (af[i] / l);
                }
                w = Var.param(Tensor.of(Shape.of(m, n), wf));
                s = null;
            }
            case ALISTA -> {
                w = Var.of(Tensor.of(Shape.of(m, n), toFloat(analyticWeights(matrix))));
                s = null;
            }
            default -> {
                w = Var.param(Tensor.of(Shape.of(m, n), af.clone()));
                s = null;
            }
        }
    }

    private static float[] toFloat(double[][] v) {
        float[] f = new float[v.length * v[0].length];
        for (int i = 0; i < v.length; i++) {
            for (int j = 0; j < v[0].length; j++) {
                f[i * v[0].length + j] = (float) v[i][j];
            }
        }
        return f;
    }

    /**
     * ALISTA-style analytic weights: column {@code i} is {@code (A A^T)^-1 a_i} scaled so that
     * {@code w_i^T a_i = 1}, the minimiser of {@code ||W^T A||_F} under that constraint.
     *
     * @param matrix {@code A}
     * @return {@code W [m][n]}
     */
    public static double[][] analyticWeights(double[][] matrix) {
        int m = matrix.length;
        int n = matrix[0].length;
        double[][] aat = DenseMatrices.gram(DenseMatrices.transpose(matrix));
        double[][] out = new double[m][n];
        for (int j = 0; j < n; j++) {
            double[] col = new double[m];
            for (int i = 0; i < m; i++) {
                col[i] = matrix[i][j];
            }
            double[] v = DenseMatrices.solveSpd(aat, col, 1e-8);
            double dot = 0;
            for (int i = 0; i < m; i++) {
                dot += v[i] * col[i];
            }
            for (int i = 0; i < m; i++) {
                out[i][j] = v[i] / dot;
            }
        }
        return out;
    }

    private static Var soft(Var u, Var t) {
        return u.sub(t).relu().sub(u.neg().sub(t).relu());
    }

    @Override
    public Var forward(Var y) {
        int batch = y.shape().dim(0);
        Var x = Var.of(Tensor.zeros(Shape.of(batch, n)));
        Var z = null;
        for (int k = 0; k < layers; k++) {
            Var th = theta.slice(0, k, k + 1);
            switch (kind) {
                case LISTA -> x = soft(y.matmul(w).add(x.matmul(s)), th);
                case LISTA_CP -> x = soft(x.add(y.sub(x.matmul(Var.of(at))).matmul(w)), th);
                case ALISTA -> x = soft(x.add(y.sub(x.matmul(Var.of(at))).matmul(w).mul(gamma.slice(0, k, k + 1))),
                        th.abs());
                default -> {
                    Var residual = y.sub(x.matmul(Var.of(at)));
                    if (z != null) {
                        float[] xv = x.value().toArray();
                        float[] onsager = new float[batch];
                        for (int b = 0; b < batch; b++) {
                            int active = 0;
                            for (int j = 0; j < n; j++) {
                                active += xv[b * n + j] != 0 ? 1 : 0;
                            }
                            onsager[b] = active / (float) m;
                        }
                        residual = residual.add(z.mul(Tensor.of(Shape.of(batch, 1), onsager)));
                    }
                    z = residual;
                    Var scale = z.square().sum(1, true).div((float) m).add(1e-12f).sqrt();
                    x = soft(x.add(z.matmul(w)), scale.mul(th));
                }
            }
        }
        return x;
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>();
        p.add(theta);
        if (gamma != null) {
            p.add(gamma);
        }
        if (kind != Kind.ALISTA) {
            p.add(w);
        }
        if (s != null) {
            p.add(s);
        }
        return p;
    }

    /** @return the number of learned scalars */
    public long learnedParameters() {
        long c = 0;
        for (Var v : parameters()) {
            c += v.value().size();
        }
        return c;
    }

    @Override
    public int outputSize(int inputSize) {
        return n;
    }

    /**
     * Trains on random {@code s}-sparse Gaussian vectors by Adam on the NMSE of the last layer.
     *
     * @param sparsity non-zeros per training vector
     * @param noise measurement noise standard deviation
     * @param steps gradient steps
     * @param batch vectors per step
     * @param learningRate the Adam step size
     * @param seed the seed
     * @return the final training NMSE
     */
    public double train(int sparsity, double noise, int steps, int batch, float learningRate, long seed) {
        Optimizer opt = Optimizer.adam(learningRate);
        SplittableRandom rnd = new SplittableRandom(seed);
        double last = 0;
        for (int step = 0; step < steps; step++) {
            Tensor[] data = batch(sparsity, noise, batch, rnd);
            Var est = forward(Var.of(data[0]));
            Var loss = est.sub(data[1]).square().sum().div(Var.of(data[1]).square().sum());
            loss.backward();
            last = loss.value().item();
            opt.step(parameters());
        }
        return last;
    }

    private Tensor[] batch(int sparsity, double noise, int size, SplittableRandom rnd) {
        float[] yv = new float[size * m];
        float[] xv = new float[size * n];
        double[][] am = new double[m][n];
        float[] af = a.toArray();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                am[i][j] = af[i * n + j];
            }
        }
        for (int b = 0; b < size; b++) {
            double[] x = SparseRecovery.sparseVector(n, sparsity, rnd);
            double[] y = DenseMatrices.multiply(am, x);
            for (int j = 0; j < n; j++) {
                xv[b * n + j] = (float) x[j];
            }
            for (int i = 0; i < m; i++) {
                yv[b * m + i] = (float) (y[i] + noise * io.github.zzuegg.brainiac.experimental.data.ChannelCodes.gaussian(rnd));
            }
        }
        return new Tensor[] {Tensor.of(Shape.of(size, m), yv), Tensor.of(Shape.of(size, n), xv)};
    }

    /**
     * The estimate for one measurement vector.
     *
     * @param y the measurements
     * @return the estimate
     */
    public double[] solve(double[] y) {
        float[] yf = new float[m];
        for (int i = 0; i < m; i++) {
            yf[i] = (float) y[i];
        }
        float[] x = Var.noGrad(() -> forward(Var.of(Tensor.of(Shape.of(1, m), yf)))).value().toArray();
        double[] out = new double[n];
        for (int j = 0; j < n; j++) {
            out[j] = x[j];
        }
        return out;
    }
}
