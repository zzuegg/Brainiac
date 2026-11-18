package io.github.zzuegg.brainiac.experimental.diffopt;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.random.RandomGenerator;

/**
 * OptNet: a differentiable quadratic-program layer (Amos and Kolter, ICML 2017,
 * <a href="https://arxiv.org/abs/1703.00443">arXiv:1703.00443</a>).
 *
 * <p>The forward pass solves {@code z* = argmin 1/2 z^T Q z + q^T z} subject to {@code A z = b} and
 * {@code G z <= h} for every row {@code q} of the input batch, by a dense primal-dual interior-point
 * method (Newton steps on the perturbed KKT conditions with a fraction-to-the-boundary step rule).
 * The backward pass differentiates the KKT conditions at the solution: with
 * {@code J = [[Q, G^T, A^T], [D(lambda) G, D(G z - h), 0], [A, 0, 0]]} and
 * {@code J^T y = (dL/dz, 0, 0)}, the gradients are {@code dL/dq = -y_z},
 * {@code dL/dQ = -y_z z^T}, {@code dL/dh = D(lambda) y_lambda}, {@code dL/db = y_nu},
 * {@code dL/dG = -lambda y_z^T - D(lambda) y_lambda z^T} and {@code dL/dA = -nu y_z^T - y_nu z^T}.
 * So every piece of the program, the constraints included, can be learned from examples: the
 * paper's showcase recovers the rules of 4x4 Sudoku as equality constraints.
 *
 * <p>The {@link Layer} form takes {@code q} as its input {@code [batch, n]} and learns some of
 * {@code Q = L^T L + eps I}, {@code G}, {@code h}, {@code A}; {@code b = A softplus(u)} is
 * parameterised through a learned point {@code softplus(u) >= 0} so the program stays feasible
 * whatever {@code A} becomes. Optional non-negativity {@code z >= 0} adds fixed rows to {@code G}.
 * {@link #solve} is the plain differentiable operation on any {@link Var}s.
 */
public final class QpLayer implements Layer {

    private final int n;
    private final double eps;
    private final Var lq;
    private final Var g;
    private final Var h;
    private final Var a;
    private final Var u;
    private final boolean nonNegative;

    private QpLayer(int n, double eps, Var lq, Var g, Var h, Var a, Var u, boolean nonNegative) {
        this.n = n;
        this.eps = eps;
        this.lq = lq;
        this.g = g;
        this.h = h;
        this.a = a;
        this.u = u;
        this.nonNegative = nonNegative;
    }

    /**
     * Creates a layer.
     *
     * @param n the variables
     * @param inequalities learned inequality rows {@code m} (0 for none)
     * @param equalities learned equality rows {@code p} (0 for none)
     * @param learnQuadratic learn {@code L} in {@code Q = L^T L + eps I}; otherwise {@code Q = eps I}
     * @param eps the ridge of {@code Q}, for example 0.1 or 1
     * @param nonNegative add the fixed constraints {@code z >= 0}
     * @param random initialises the parameters
     * @return the layer
     */
    public static QpLayer of(int n, int inequalities, int equalities, boolean learnQuadratic, double eps,
            boolean nonNegative, RandomGenerator random) {
        Var lq = learnQuadratic ? Var.param(Tensor.normal(Shape.of(n, n), 0, (float) (0.3 / Math.sqrt(n)), random))
                : null;
        Var g = null;
        Var h = null;
        if (inequalities > 0) {
            g = Var.param(Tensor.normal(Shape.of(inequalities, n), 0, (float) (1 / Math.sqrt(n)), random));
            h = Var.param(Tensor.full(Shape.of(1, inequalities), 1f));
        }
        Var a = null;
        Var u = null;
        if (equalities > 0) {
            a = Var.param(Tensor.normal(Shape.of(equalities, n), 0, (float) (1 / Math.sqrt(n)), random));
            u = Var.param(Tensor.full(Shape.of(n, 1), -1f));
        }
        return new QpLayer(n, eps, lq, g, h, a, u, nonNegative);
    }

    /**
     * Describes the layer for {@code Model.sequential().add(...)}: output width equals input width.
     *
     * @param inequalities learned inequality rows
     * @param equalities learned equality rows
     * @param learnQuadratic learn {@code Q}
     * @param eps the ridge of {@code Q}
     * @param nonNegative add {@code z >= 0}
     * @return the spec
     */
    public static LayerSpec spec(int inequalities, int equalities, boolean learnQuadratic, double eps,
            boolean nonNegative) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return input;
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return of(input.dim(-1), inequalities, equalities, learnQuadratic, eps, nonNegative, random);
            }
        };
    }

    @Override
    public Var forward(Var q) {
        Var qm;
        if (lq != null) {
            qm = lq.transpose().matmul(lq).add(identity(n, (float) eps));
        } else {
            qm = Var.of(identity(n, (float) eps));
        }
        Var gm = g;
        Var hm = h;
        if (nonNegative) {
            Var neg = Var.of(identity(n, -1f));
            Var zero = Var.of(Tensor.zeros(Shape.of(1, n)));
            gm = gm == null ? neg : Var.concat(0, gm, neg);
            hm = hm == null ? zero : Var.concat(1, hm, zero);
        }
        Var bm = a == null ? null : a.matmul(u.softplus()).transpose();
        return solve(qm, q, gm, hm, a, bm);
    }

    private static Tensor identity(int n, float value) {
        float[] v = new float[n * n];
        for (int i = 0; i < n; i++) {
            v[i * n + i] = value;
        }
        return Tensor.of(Shape.of(n, n), v);
    }

    @Override
    public List<Var> parameters() {
        List<Var> ps = new ArrayList<>();
        for (Var v : new Var[] {lq, g, h, a, u}) {
            if (v != null) {
                ps.add(v);
            }
        }
        return ps;
    }

    @Override
    public String toString() {
        return "QpLayer(" + n + ")";
    }

    /**
     * The learned inequality matrix including the fixed non-negativity rows, as doubles.
     *
     * @return {@code G}, or an empty array
     */
    public double[][] inequalityMatrix() {
        return g == null ? new double[0][] : rows(g.value());
    }

    /**
     * The learned inequality bounds.
     *
     * @return {@code h}, or an empty array
     */
    public double[] inequalityBounds() {
        return h == null ? new double[0] : h.value().toDoubleArray();
    }

    private static double[][] rows(Tensor t) {
        int r = t.shape().dim(0);
        int c = t.shape().dim(1);
        double[] v = t.toDoubleArray();
        double[][] out = new double[r][c];
        for (int i = 0; i < r; i++) {
            System.arraycopy(v, i * c, out[i], 0, c);
        }
        return out;
    }

    /**
     * Solves a batch of QPs that share {@code Q, G, h, A, b} and differ in {@code q}, as a
     * differentiable operation in all six.
     *
     * @param qm {@code Q [n, n]}, symmetric positive definite
     * @param q the linear terms {@code [batch, n]}
     * @param gm {@code G [m, n]} or null
     * @param hm {@code h [1, m]} or null
     * @param am {@code A [p, n]} or null
     * @param bm {@code b [1, p]} or null
     * @return the solutions {@code [batch, n]}
     */
    public static Var solve(Var qm, Var q, Var gm, Var hm, Var am, Var bm) {
        int batch = q.shape().dim(0);
        int n = q.shape().dim(1);
        int m = gm == null ? 0 : gm.shape().dim(0);
        int p = am == null ? 0 : am.shape().dim(0);
        List<Var> inputs = new ArrayList<>(List.of(qm, q));
        if (m > 0) {
            inputs.add(gm);
            inputs.add(hm);
        }
        if (p > 0) {
            inputs.add(am);
            inputs.add(bm);
        }
        double[][][] sol = new double[batch][][];
        return Var.custom(in -> {
            double[][] qq = rows(in[0]);
            double[] qv = in[1].toDoubleArray();
            double[][] gg = m > 0 ? rows(in[2]) : new double[0][n];
            double[] hh = m > 0 ? in[3].toDoubleArray() : new double[0];
            double[][] aa = p > 0 ? rows(in[m > 0 ? 4 : 2]) : new double[0][n];
            double[] bb = p > 0 ? in[m > 0 ? 5 : 3].toDoubleArray() : new double[0];
            float[] out = new float[batch * n];
            for (int b = 0; b < batch; b++) {
                double[] qb = java.util.Arrays.copyOfRange(qv, b * n, (b + 1) * n);
                sol[b] = interiorPoint(qq, qb, gg, hh, aa, bb, 40);
                for (int i = 0; i < n; i++) {
                    out[b * n + i] = (float) sol[b][0][i];
                }
            }
            return Tensor.of(Shape.of(batch, n), out);
        }, (vals, output, grad) -> {
            double[][] qq = rows(vals[0]);
            double[][] gg = m > 0 ? rows(vals[2]) : new double[0][n];
            double[] hh = m > 0 ? vals[3].toDoubleArray() : new double[0];
            double[][] aa = p > 0 ? rows(vals[m > 0 ? 4 : 2]) : new double[0][n];
            double[] gv = grad.toDoubleArray();
            double[] dQ = new double[n * n];
            float[] dq = new float[batch * n];
            double[] dG = new double[m * n];
            double[] dh = new double[m];
            double[] dA = new double[p * n];
            double[] db = new double[p];
            int k = n + m + p;
            for (int bi = 0; bi < batch; bi++) {
                double[] z = sol[bi][0];
                double[] lam = sol[bi][1];
                double[] nu = sol[bi][2];
                // J^T = [[Q, G^T D(lam), A^T], [G, D(Gz - h), 0], [A, 0, 0]]
                double[][] jt = new double[k][k];
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < n; j++) {
                        jt[i][j] = qq[i][j];
                    }
                    for (int r = 0; r < m; r++) {
                        jt[i][n + r] = gg[r][i] * lam[r];
                        jt[n + r][i] = gg[r][i];
                    }
                    for (int r = 0; r < p; r++) {
                        jt[i][n + m + r] = aa[r][i];
                        jt[n + m + r][i] = aa[r][i];
                    }
                }
                for (int r = 0; r < m; r++) {
                    double slack = -hh[r];
                    for (int j = 0; j < n; j++) {
                        slack += gg[r][j] * z[j];
                    }
                    jt[n + r][n + r] = slack;
                }
                for (int r = 0; r < p; r++) {
                    jt[n + m + r][n + m + r] = -1e-9;
                }
                double[] rhs = new double[k];
                for (int i = 0; i < n; i++) {
                    rhs[i] = gv[bi * n + i];
                }
                double[] y = luSolve(jt, rhs);
                for (int i = 0; i < n; i++) {
                    double yz = y[i];
                    dq[bi * n + i] = (float) -yz;
                    for (int j = 0; j < n; j++) {
                        dQ[i * n + j] -= yz * z[j];
                    }
                }
                for (int r = 0; r < m; r++) {
                    double yl = y[n + r];
                    dh[r] += lam[r] * yl;
                    for (int j = 0; j < n; j++) {
                        dG[r * n + j] += -lam[r] * y[j] - lam[r] * yl * z[j];
                    }
                }
                for (int r = 0; r < p; r++) {
                    double yn = y[n + m + r];
                    db[r] += yn;
                    for (int j = 0; j < n; j++) {
                        dA[r * n + j] += -nu[r] * y[j] - yn * z[j];
                    }
                }
            }
            List<Tensor> grads = new ArrayList<>();
            grads.add(Tensor.of(Shape.of(n, n), toFloat(dQ)));
            grads.add(Tensor.of(Shape.of(batch, n), dq));
            if (m > 0) {
                grads.add(Tensor.of(Shape.of(m, n), toFloat(dG)));
                grads.add(Tensor.of(Shape.of(1, m), toFloat(dh)));
            }
            if (p > 0) {
                grads.add(Tensor.of(Shape.of(p, n), toFloat(dA)));
                grads.add(Tensor.of(Shape.of(1, p), toFloat(db)));
            }
            return grads.toArray(new Tensor[0]);
        }, inputs.toArray(new Var[0]));
    }

    private static float[] toFloat(double[] x) {
        float[] f = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            f[i] = (float) x[i];
        }
        return f;
    }

    /**
     * A dense primal-dual interior-point solve.
     *
     * @param qq {@code Q}
     * @param q {@code q}
     * @param g {@code G} (may have no rows)
     * @param h {@code h}
     * @param a {@code A} (may have no rows)
     * @param b {@code b}
     * @param iterations Newton iterations
     * @return {@code {z, lambda, nu}}
     */
    public static double[][] interiorPoint(double[][] qq, double[] q, double[][] g, double[] h, double[][] a,
            double[] b, int iterations) {
        int n = q.length;
        int m = h.length;
        int p = b.length;
        double[] z = new double[n];
        double[] s = new double[m];
        double[] lam = new double[m];
        double[] nu = new double[p];
        for (int r = 0; r < m; r++) {
            s[r] = Math.max(1, h[r]);
            lam[r] = 1;
        }
        for (int it = 0; it < iterations; it++) {
            double mu = 0;
            for (int r = 0; r < m; r++) {
                mu += s[r] * lam[r] / m;
            }
            double[] rd = new double[n];
            for (int i = 0; i < n; i++) {
                double v = q[i];
                for (int j = 0; j < n; j++) {
                    v += qq[i][j] * z[j];
                }
                for (int r = 0; r < m; r++) {
                    v += g[r][i] * lam[r];
                }
                for (int r = 0; r < p; r++) {
                    v += a[r][i] * nu[r];
                }
                rd[i] = v;
            }
            double[] rin = new double[m];
            for (int r = 0; r < m; r++) {
                double v = s[r] - h[r];
                for (int j = 0; j < n; j++) {
                    v += g[r][j] * z[j];
                }
                rin[r] = v;
            }
            double[] req = new double[p];
            for (int r = 0; r < p; r++) {
                double v = -b[r];
                for (int j = 0; j < n; j++) {
                    v += a[r][j] * z[j];
                }
                req[r] = v;
            }
            double res = 0;
            for (double v : rd) {
                res = Math.max(res, Math.abs(v));
            }
            for (double v : rin) {
                res = Math.max(res, Math.abs(v));
            }
            for (double v : req) {
                res = Math.max(res, Math.abs(v));
            }
            if (res < 1e-9 && (m == 0 || mu < 1e-9)) {
                break;
            }
            double sigma = 0.1;
            // rc = lam s - sigma mu; reduced system in (dz, dnu)
            double[][] kk = new double[n + p][n + p];
            double[] rhs = new double[n + p];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    kk[i][j] = qq[i][j];
                }
                rhs[i] = -rd[i];
            }
            for (int r = 0; r < m; r++) {
                double dr = lam[r] / s[r];
                double rc = lam[r] * s[r] - sigma * mu;
                double t = (-rc + lam[r] * rin[r]) / s[r];
                for (int i = 0; i < n; i++) {
                    double gi = g[r][i];
                    if (gi == 0) {
                        continue;
                    }
                    rhs[i] -= gi * t;
                    for (int j = 0; j < n; j++) {
                        kk[i][j] += gi * dr * g[r][j];
                    }
                }
            }
            for (int r = 0; r < p; r++) {
                for (int i = 0; i < n; i++) {
                    kk[i][n + r] = a[r][i];
                    kk[n + r][i] = a[r][i];
                }
                kk[n + r][n + r] = -1e-10;
                rhs[n + r] = -req[r];
            }
            double[] step = luSolve(kk, rhs);
            double[] dz = java.util.Arrays.copyOf(step, n);
            double[] ds = new double[m];
            double[] dl = new double[m];
            for (int r = 0; r < m; r++) {
                double gdz = 0;
                for (int j = 0; j < n; j++) {
                    gdz += g[r][j] * dz[j];
                }
                ds[r] = -rin[r] - gdz;
                double rc = lam[r] * s[r] - sigma * mu;
                dl[r] = (-rc - lam[r] * ds[r]) / s[r];
            }
            double alpha = 1;
            for (int r = 0; r < m; r++) {
                if (ds[r] < 0) {
                    alpha = Math.min(alpha, -0.99 * s[r] / ds[r]);
                }
                if (dl[r] < 0) {
                    alpha = Math.min(alpha, -0.99 * lam[r] / dl[r]);
                }
            }
            for (int i = 0; i < n; i++) {
                z[i] += alpha * dz[i];
            }
            for (int r = 0; r < m; r++) {
                s[r] += alpha * ds[r];
                lam[r] += alpha * dl[r];
            }
            for (int r = 0; r < p; r++) {
                nu[r] += alpha * step[n + r];
            }
        }
        return new double[][] {z, lam, nu};
    }

    /**
     * Solves a dense square system by LU with partial pivoting.
     *
     * @param matrix the matrix (overwritten)
     * @param rhs the right-hand side
     * @return the solution
     */
    static double[] luSolve(double[][] matrix, double[] rhs) {
        int k = rhs.length;
        double[][] lu = matrix;
        double[] x = rhs.clone();
        for (int c = 0; c < k; c++) {
            int piv = c;
            for (int r = c + 1; r < k; r++) {
                if (Math.abs(lu[r][c]) > Math.abs(lu[piv][c])) {
                    piv = r;
                }
            }
            if (piv != c) {
                double[] t = lu[c];
                lu[c] = lu[piv];
                lu[piv] = t;
                double tv = x[c];
                x[c] = x[piv];
                x[piv] = tv;
            }
            double d = lu[c][c];
            if (Math.abs(d) < 1e-14) {
                d = d < 0 ? -1e-14 : 1e-14;
                lu[c][c] = d;
            }
            for (int r = c + 1; r < k; r++) {
                double f = lu[r][c] / d;
                if (f == 0) {
                    continue;
                }
                double[] lr = lu[r];
                double[] lc = lu[c];
                for (int j = c + 1; j < k; j++) {
                    lr[j] -= f * lc[j];
                }
                x[r] -= f * x[c];
            }
        }
        for (int r = k - 1; r >= 0; r--) {
            double v = x[r];
            for (int j = r + 1; j < k; j++) {
                v -= lu[r][j] * x[j];
            }
            x[r] = v / lu[r][r];
        }
        return x;
    }
}
