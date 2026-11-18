package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Flows;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Glow (Kingma and Dhariwal, <a href="https://arxiv.org/abs/1807.03039">arXiv:1807.03039</a>)
 * over vectors: each step is an <b>ActNorm</b> (a per-feature scale and shift initialised from
 * the first batch so activations start standardised), an <b>invertible 1x1 convolution</b>
 * (a learned square mixing matrix in LU form, {@code W = P L (U + diag(sign * exp(s)))}, whose
 * log-determinant is just {@code sum(s)}) and an <b>affine coupling</b> (core
 * {@link Flows.Coupling}). RealNVP only alternates fixed masks; the learned mixing lets every
 * coupling split along a direction that suits the data.
 *
 * <pre>{@code
 * GlowFlow glow = GlowFlow.of(2, 6, 64, 42);
 * Var nll = glow.logProb(Var.of(batch)).mean().neg();   // first call initialises ActNorm
 * }</pre>
 */
public final class GlowFlow implements DensityFlow {
    private final int dim;
    private final List<ActNorm> norms = new ArrayList<>();
    private final List<Mixing> mixings = new ArrayList<>();
    private final List<Flows.Coupling> couplings = new ArrayList<>();
    private final List<GenerativeOps.Mlp> nets = new ArrayList<>();

    private GlowFlow(int dim, int steps, int hidden, long seed) {
        if (dim < 2 || steps < 1) {
            throw new IllegalArgumentException("Glow needs at least 2 features and 1 step");
        }
        this.dim = dim;
        SplittableRandom random = new SplittableRandom(seed);
        Tensor[] masks = Flows.alternatingMasks(dim, 2);
        for (int s = 0; s < steps; s++) {
            norms.add(new ActNorm(dim));
            mixings.add(new Mixing(dim, random));
            GenerativeOps.Mlp net = new GenerativeOps.Mlp(random, true, dim, hidden, hidden, 2 * dim);
            nets.add(net);
            couplings.add(new Flows.Coupling(masks[s % 2], kept -> {
                Var st = net.apply(kept);
                return new Var[] {st.slice(1, 0, dim), st.slice(1, dim, 2 * dim)};
            }));
        }
    }

    /**
     * Creates the flow.
     *
     * @param dim the data dimension, at least 2
     * @param steps the number of ActNorm / mixing / coupling steps
     * @param hidden the hidden width of each coupling network
     * @param seed the initialization seed
     * @return the flow
     */
    public static GlowFlow of(int dim, int steps, int hidden, long seed) {
        return new GlowFlow(dim, steps, hidden, seed);
    }

    @Override
    public Flows.Result forward(Var x) {
        Var h = x;
        Var logDet = null;
        for (int s = 0; s < couplings.size(); s++) {
            Flows.Result a = norms.get(s).forward(h);
            Flows.Result m = mixings.get(s).forward(a.output());
            Flows.Result c = couplings.get(s).forward(m.output());
            h = c.output();
            Var ld = a.logDet().add(m.logDet()).add(c.logDet());
            logDet = logDet == null ? ld : logDet.add(ld);
        }
        return new Flows.Result(h, logDet);
    }

    @Override
    public Tensor inverse(Tensor z) {
        Var h = Var.of(z);
        for (int s = couplings.size() - 1; s >= 0; s--) {
            h = couplings.get(s).inverse(h).output();
            h = Var.of(mixings.get(s).inverse(h.value()));
            h = Var.of(norms.get(s).inverse(h.value()));
        }
        return h.value();
    }

    @Override
    public int dimension() {
        return dim;
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>();
        for (int s = 0; s < couplings.size(); s++) {
            p.add(norms.get(s).logScale);
            p.add(norms.get(s).bias);
            p.addAll(mixings.get(s).parameters());
            p.addAll(nets.get(s).parameters());
        }
        return p;
    }

    /** ActNorm: {@code y = (x + b) * exp(s)}, with data-dependent initialisation on first use. */
    private static final class ActNorm {
        final Var logScale;
        final Var bias;
        boolean initialized;

        ActNorm(int dim) {
            logScale = TechniqueOps.filled(0f, dim);
            bias = TechniqueOps.filled(0f, dim);
        }

        Flows.Result forward(Var x) {
            if (!initialized && x.shape().dim(0) > 1) {
                initialized = true;
                float[][] rows = GenerativeOps.rows(x.value());
                int d = rows[0].length;
                float[] b = new float[d];
                float[] s = new float[d];
                for (int k = 0; k < d; k++) {
                    double mean = 0;
                    for (float[] r : rows) {
                        mean += r[k];
                    }
                    mean /= rows.length;
                    double var = 0;
                    for (float[] r : rows) {
                        var += (r[k] - mean) * (r[k] - mean);
                    }
                    var /= rows.length;
                    b[k] = (float) -mean;
                    s[k] = (float) -Math.log(Math.sqrt(var) + 1e-6);
                }
                bias.assign(Tensor.of(Shape.of(d), b));
                logScale.assign(Tensor.of(Shape.of(d), s));
            }
            Var y = x.add(bias).mul(logScale.exp());
            int batch = x.shape().dim(0);
            Var ld = logScale.sum().reshape(1).expand(Shape.of(batch));
            return new Flows.Result(y, ld);
        }

        Tensor inverse(Tensor y) {
            return y.mul(logScale.value().neg().exp()).sub(bias.value());
        }
    }

    /** The invertible 1x1 convolution in LU form, starting from a random rotation. */
    private static final class Mixing {
        final int dim;
        final Tensor permutation;
        final Tensor lowerMask;
        final Tensor upperMask;
        final Tensor identity;
        final Tensor sign;
        final Var lower;
        final Var upper;
        final Var logS;

        Mixing(int dim, SplittableRandom random) {
            this.dim = dim;
            double[][] q = randomRotation(dim, random);
            double[][][] plu = lu(q);
            float[] p = new float[dim * dim];
            float[] l = new float[dim * dim];
            float[] u = new float[dim * dim];
            float[] lm = new float[dim * dim];
            float[] um = new float[dim * dim];
            float[] id = new float[dim * dim];
            float[] sg = new float[dim];
            float[] ls = new float[dim];
            for (int i = 0; i < dim; i++) {
                for (int j = 0; j < dim; j++) {
                    p[i * dim + j] = (float) plu[0][i][j];
                    if (j < i) {
                        l[i * dim + j] = (float) plu[1][i][j];
                        lm[i * dim + j] = 1;
                    } else if (j > i) {
                        u[i * dim + j] = (float) plu[2][i][j];
                        um[i * dim + j] = 1;
                    }
                }
                id[i * dim + i] = 1;
                sg[i] = plu[2][i][i] < 0 ? -1 : 1;
                ls[i] = (float) Math.log(Math.abs(plu[2][i][i]) + 1e-12);
            }
            permutation = Tensor.of(Shape.of(dim, dim), p);
            lowerMask = Tensor.of(Shape.of(dim, dim), lm);
            upperMask = Tensor.of(Shape.of(dim, dim), um);
            identity = Tensor.of(Shape.of(dim, dim), id);
            sign = Tensor.of(Shape.of(dim), sg);
            lower = Var.param(Tensor.of(Shape.of(dim, dim), l));
            upper = Var.param(Tensor.of(Shape.of(dim, dim), u));
            logS = Var.param(Tensor.of(Shape.of(dim), ls));
        }

        Var weight() {
            Var l = lower.mul(lowerMask).add(identity);
            Var diag = Var.of(identity).mul(logS.exp().mul(sign).reshape(1, dim));
            Var u = upper.mul(upperMask).add(diag);
            return Var.of(permutation).matmul(l).matmul(u);
        }

        Flows.Result forward(Var x) {
            int batch = x.shape().dim(0);
            return new Flows.Result(x.matmul(weight()), logS.sum().reshape(1).expand(Shape.of(batch)));
        }

        Tensor inverse(Tensor y) {
            float[] w = weight().value().toArray();
            double[][] a = new double[dim][dim];
            for (int i = 0; i < dim; i++) {
                for (int j = 0; j < dim; j++) {
                    a[i][j] = w[i * dim + j];
                }
            }
            double[][] inv = invert(a);
            float[] f = new float[dim * dim];
            for (int i = 0; i < dim; i++) {
                for (int j = 0; j < dim; j++) {
                    f[i * dim + j] = (float) inv[i][j];
                }
            }
            return y.matmul(Tensor.of(Shape.of(dim, dim), f));
        }

        List<Var> parameters() {
            return List.of(lower, upper, logS);
        }
    }

    private static double[][] randomRotation(int n, SplittableRandom random) {
        double[][] q = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                q[i][j] = random.nextGaussian();
            }
            for (int k = 0; k < i; k++) {
                double dot = 0;
                for (int j = 0; j < n; j++) {
                    dot += q[i][j] * q[k][j];
                }
                for (int j = 0; j < n; j++) {
                    q[i][j] -= dot * q[k][j];
                }
            }
            double norm = 0;
            for (int j = 0; j < n; j++) {
                norm += q[i][j] * q[i][j];
            }
            norm = Math.sqrt(norm);
            for (int j = 0; j < n; j++) {
                q[i][j] /= norm;
            }
        }
        return q;
    }

    /** {@code A = P L U} with partial pivoting; returns {@code {P, L, U}}. */
    private static double[][][] lu(double[][] a) {
        int n = a.length;
        double[][] u = new double[n][];
        for (int i = 0; i < n; i++) {
            u[i] = a[i].clone();
        }
        double[][] l = new double[n][n];
        int[] perm = new int[n];
        for (int i = 0; i < n; i++) {
            perm[i] = i;
        }
        for (int k = 0; k < n; k++) {
            int pivot = k;
            for (int i = k + 1; i < n; i++) {
                if (Math.abs(u[i][k]) > Math.abs(u[pivot][k])) {
                    pivot = i;
                }
            }
            double[] tmp = u[k];
            u[k] = u[pivot];
            u[pivot] = tmp;
            double[] tl = l[k];
            l[k] = l[pivot];
            l[pivot] = tl;
            int tp = perm[k];
            perm[k] = perm[pivot];
            perm[pivot] = tp;
            for (int i = k + 1; i < n; i++) {
                double f = u[i][k] / u[k][k];
                l[i][k] = f;
                for (int j = k; j < n; j++) {
                    u[i][j] -= f * u[k][j];
                }
            }
        }
        for (int i = 0; i < n; i++) {
            l[i][i] = 1;
        }
        // rows of P L U are rows perm of A, so A = P^T-applied: A[perm[i]] = (LU)[i]
        double[][] p = new double[n][n];
        for (int i = 0; i < n; i++) {
            p[perm[i]][i] = 1;
        }
        return new double[][][] {p, l, u};
    }

    private static double[][] invert(double[][] a) {
        int n = a.length;
        double[][] m = new double[n][2 * n];
        for (int i = 0; i < n; i++) {
            System.arraycopy(a[i], 0, m[i], 0, n);
            m[i][n + i] = 1;
        }
        for (int k = 0; k < n; k++) {
            int pivot = k;
            for (int i = k + 1; i < n; i++) {
                if (Math.abs(m[i][k]) > Math.abs(m[pivot][k])) {
                    pivot = i;
                }
            }
            double[] t = m[k];
            m[k] = m[pivot];
            m[pivot] = t;
            double d = m[k][k];
            for (int j = 0; j < 2 * n; j++) {
                m[k][j] /= d;
            }
            for (int i = 0; i < n; i++) {
                if (i != k) {
                    double f = m[i][k];
                    for (int j = 0; j < 2 * n; j++) {
                        m[i][j] -= f * m[k][j];
                    }
                }
            }
        }
        double[][] inv = new double[n][n];
        for (int i = 0; i < n; i++) {
            System.arraycopy(m[i], n, inv[i], 0, n);
        }
        return inv;
    }
}
