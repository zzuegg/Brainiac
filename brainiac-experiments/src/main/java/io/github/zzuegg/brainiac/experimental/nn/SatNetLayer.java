package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;
import java.util.random.RandomGenerator;
import java.util.stream.IntStream;

/**
 * SATNet: a differentiable MAXSAT layer. The clause matrix {@code S} (one row per
 * variable plus a "truth" row and auxiliary rows, one column per soft clause) is learned.
 * The forward pass solves the semidefinite relaxation
 * {@code min <S S^T, V^T V>} over unit vectors {@code v_i}: known inputs are clamped
 * to {@code v_i = -cos(pi z_i) v_0 + sin(pi z_i) r_i}, and the free (output and auxiliary)
 * vectors are found by the mixing method, block coordinate descent
 * {@code v_o = -g_o / |g_o|} with {@code g_o = sum_{j != o} (S S^T)_{oj} v_j}. Outputs are
 * read as {@code z_o = arccos(-v_o . v_0) / pi}.
 *
 * <p>The backward pass differentiates the fixed point analytically: with
 * {@code a_o = dl/dv_o}, it solves {@code w_o = P_o (a_o - sum_{j != o} C_oj w_j) / |g_o|}
 * ({@code P_o = I - v_o v_o^T}) by the same kind of coordinate sweeps, then
 * {@code dl/dC_oj = -w_o . v_j}, {@code dl/dS = (M + M^T) S} and
 * {@code dl/dv_i = -sum_o C_oi w_o} for the inputs. So the layer learns the rules of a
 * logic puzzle (the clauses) from input-output examples alone.
 *
 * <p>Input to {@link #forward(Var)} is {@code [batch, 2n]}: the {@code n} variable values in
 * {@code [0, 1]}, then {@code n} flags (1 = given, clamped; 0 = to be solved). Output is
 * {@code [batch, n]}: the givens unchanged and the solved values elsewhere.
 *
 * <p>Paper: Wang, Donti, Wilder, Kolter, "SATNet: Bridging deep learning and logical
 * reasoning using a differentiable satisfiability solver" (ICML 2019),
 * <a href="https://arxiv.org/abs/1905.12149">arXiv:1905.12149</a>.
 */
public final class SatNetLayer implements Layer {

    private final int n;
    private final int aux;
    private final int k;
    private final Var s;
    private final long seed;
    private int sweeps = 30;

    /**
     * Creates the layer.
     *
     * @param variables the visible variables {@code n}
     * @param clauses the soft clauses {@code m}
     * @param auxiliary hidden auxiliary variables
     * @param random initialises {@code S}
     */
    public SatNetLayer(int variables, int clauses, int auxiliary, RandomGenerator random) {
        this.n = variables;
        this.aux = auxiliary;
        int total = variables + auxiliary + 1;
        this.k = Math.max(8, (int) Math.ceil(Math.sqrt(2.0 * total)) + 1);
        float[] w = new float[total * clauses];
        double scale = 1.0 / Math.sqrt(total + clauses);
        for (int i = 0; i < w.length; i++) {
            w[i] = (float) (random.nextGaussian() * scale);
        }
        this.s = Var.param(Tensor.of(Shape.of(total, clauses), w));
        this.seed = random.nextLong();
    }

    /**
     * Describes the layer for {@code Model.sequential().add(...)}: the input width must be
     * {@code 2 * variables}.
     *
     * @param clauses the soft clauses
     * @param auxiliary auxiliary variables
     * @return the spec
     */
    public static LayerSpec of(int clauses, int auxiliary) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return Shape.of(input.dim(0), input.dim(1) / 2);
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return new SatNetLayer(input.dim(-1) / 2, clauses, auxiliary, random);
            }
        };
    }

    /**
     * Sets the coordinate-descent sweeps of the forward and backward solves.
     *
     * @param count sweeps over all free variables, such as 30
     * @return this layer
     */
    public SatNetLayer sweeps(int count) {
        this.sweeps = count;
        return this;
    }

    /** @return the clause matrix {@code S} */
    public Var clauses() {
        return s;
    }

    @Override
    public Var forward(Var input) {
        int b = input.shape().dim(0);
        Var z = input.slice(1, 0, n);
        Tensor mask = input.value().slice(1, n, 2 * n);
        return solve(z, mask);
    }

    /**
     * Solves for the free variables.
     *
     * @param z values {@code [batch, n]} in {@code [0, 1]} (free ones are ignored)
     * @param given flags {@code [batch, n]}, 1 where the value is clamped
     * @return {@code [batch, n]}
     */
    public Var solve(Var z, Tensor given) {
        int b = z.shape().dim(0);
        int total = n + aux + 1;
        float[] mask = given.toArray();
        double[][][] vs = new double[b][][];
        double[][] gnorm = new double[b][];
        double[][][] rand = new double[b][][];
        return Var.custom(in -> {
            float[] zv = in[0].toArray();
            float[] sv = in[1].toArray();
            int m = in[1].shape().dim(1);
            float[] out = new float[b * n];
            IntStream.range(0, b).parallel().forEach(i -> {
                double[][] v = new double[total][k];
                double[][] r = new double[total][k];
                SplittableRandom rng = new SplittableRandom(seed);
                v[0][0] = 1;
                for (int j = 1; j < total; j++) {
                    double norm = 0;
                    for (int d = 1; d < k; d++) {
                        r[j][d] = rng.nextGaussian();
                        norm += r[j][d] * r[j][d];
                    }
                    norm = Math.sqrt(norm);
                    for (int d = 1; d < k; d++) {
                        r[j][d] /= norm;
                    }
                    boolean isGiven = j <= n && mask[i * n + j - 1] > 0.5f;
                    if (isGiven) {
                        double a = Math.PI * zv[i * n + j - 1];
                        for (int d = 0; d < k; d++) {
                            v[j][d] = -Math.cos(a) * v[0][d] + Math.sin(a) * r[j][d];
                        }
                    } else {
                        System.arraycopy(r[j], 0, v[j], 0, k);
                        v[j][0] = rng.nextGaussian() * 0.1;
                        normalise(v[j]);
                    }
                }
                double[] g = new double[total];
                mix(v, sv, m, i, mask, g);
                vs[i] = v;
                gnorm[i] = g;
                rand[i] = r;
                for (int j = 1; j <= n; j++) {
                    boolean isGiven = mask[i * n + j - 1] > 0.5f;
                    out[i * n + j - 1] = isGiven ? zv[i * n + j - 1]
                            : (float) (Math.acos(Math.max(-1, Math.min(1, -v[j][0]))) / Math.PI);
                }
            });
            return Tensor.of(Shape.of(b, n), out);
        }, (in, output, grad) -> {
            float[] zv = in[0].toArray();
            float[] sv = in[1].toArray();
            float[] gv = grad.toArray();
            int m = in[1].shape().dim(1);
            float[] dz = new float[b * n];
            double[][] dsAll = new double[b][];
            IntStream.range(0, b).parallel().forEach(i -> {
                double[][] v = vs[i];
                double[][] c = gram(sv, total, m);
                double[][] a = new double[total][k];
                boolean[] free = new boolean[total];
                for (int j = 1; j < total; j++) {
                    free[j] = j > n || mask[i * n + j - 1] < 0.5f;
                }
                for (int j = 1; j <= n; j++) {
                    if (free[j]) {
                        double x = Math.max(-1 + 1e-6, Math.min(1 - 1e-6, -v[j][0]));
                        double f = gv[i * n + j - 1] / (Math.PI * Math.sqrt(1 - x * x));
                        a[j][0] = f;   // dz/dv = v_0 / (pi sqrt(1 - x^2)), v_0 = e_0
                    } else {
                        dz[i * n + j - 1] += gv[i * n + j - 1];   // the given passes through
                    }
                }
                double[][] w = new double[total][k];
                for (int sweep = 0; sweep < sweeps; sweep++) {
                    for (int o = 1; o < total; o++) {
                        if (!free[o]) {
                            continue;
                        }
                        double[] u = a[o].clone();
                        for (int j = 1; j < total; j++) {
                            if (j != o && free[j] && c[o][j] != 0) {
                                for (int d = 0; d < k; d++) {
                                    u[d] -= c[o][j] * w[j][d];
                                }
                            }
                        }
                        double dot = 0;
                        for (int d = 0; d < k; d++) {
                            dot += v[o][d] * u[d];
                        }
                        double gn = Math.max(gnorm[i][o], 1e-9);
                        for (int d = 0; d < k; d++) {
                            w[o][d] = (u[d] - dot * v[o][d]) / gn;
                        }
                    }
                }
                // dl/dC_oj = -w_o . v_j for free o, j != o; dl/dS = (M + M^T) S
                double[][] mm = new double[total][total];
                for (int o = 1; o < total; o++) {
                    if (!free[o]) {
                        continue;
                    }
                    for (int j = 0; j < total; j++) {
                        if (j != o) {
                            double dot = 0;
                            for (int d = 0; d < k; d++) {
                                dot += w[o][d] * v[j][d];
                            }
                            mm[o][j] = -dot;
                        }
                    }
                }
                double[] ds = new double[total * m];
                for (int p = 0; p < total; p++) {
                    for (int q = 0; q < total; q++) {
                        double coef = mm[p][q] + mm[q][p];
                        if (coef == 0) {
                            continue;
                        }
                        for (int col = 0; col < m; col++) {
                            ds[p * m + col] += coef * sv[q * m + col];
                        }
                    }
                }
                dsAll[i] = ds;
                // inputs: dl/dv_j = -sum_o C_oj w_o, dv_j/dz_j = pi (sin(pi z) v_0 + cos(pi z) r_j)
                for (int j = 1; j <= n; j++) {
                    if (free[j]) {
                        continue;
                    }
                    double[] dv = new double[k];
                    for (int o = 1; o < total; o++) {
                        if (free[o] && c[o][j] != 0) {
                            for (int d = 0; d < k; d++) {
                                dv[d] -= c[o][j] * w[o][d];
                            }
                        }
                    }
                    double ang = Math.PI * zv[i * n + j - 1];
                    double s0 = 0;
                    for (int d = 0; d < k; d++) {
                        double dir = (d == 0 ? Math.sin(ang) : 0) + Math.cos(ang) * rand[i][j][d];
                        s0 += dv[d] * dir;
                    }
                    dz[i * n + j - 1] += (float) (Math.PI * s0);
                }
            });
            float[] dsum = new float[total * m];
            for (double[] ds : dsAll) {
                for (int q = 0; q < ds.length; q++) {
                    dsum[q] += (float) ds[q];
                }
            }
            return new Tensor[] {Tensor.of(in[0].shape(), dz), Tensor.of(in[1].shape(), dsum)};
        }, z, s);
    }

    private static double[][] gram(float[] sv, int total, int m) {
        double[][] c = new double[total][total];
        for (int p = 0; p < total; p++) {
            for (int q = p; q < total; q++) {
                double dot = 0;
                for (int col = 0; col < m; col++) {
                    dot += sv[p * m + col] * sv[q * m + col];
                }
                c[p][q] = dot;
                c[q][p] = dot;
            }
        }
        return c;
    }

    /** The mixing method on one sample; fills {@code g} with the final |g_o|. */
    private void mix(double[][] v, float[] sv, int m, int sample, float[] mask, double[] g) {
        int total = v.length;
        double[][] omega = new double[k][m];   // V S
        for (int j = 0; j < total; j++) {
            for (int d = 0; d < k; d++) {
                if (v[j][d] != 0) {
                    for (int col = 0; col < m; col++) {
                        omega[d][col] += v[j][d] * sv[j * m + col];
                    }
                }
            }
        }
        double[] sq = new double[total];
        for (int j = 0; j < total; j++) {
            for (int col = 0; col < m; col++) {
                sq[j] += sv[j * m + col] * sv[j * m + col];
            }
        }
        double[] go = new double[k];
        for (int sweep = 0; sweep < sweeps; sweep++) {
            for (int o = 1; o < total; o++) {
                if (o <= n && mask[sample * n + o - 1] > 0.5f) {
                    continue;
                }
                double norm = 0;
                for (int d = 0; d < k; d++) {
                    double dot = 0;
                    for (int col = 0; col < m; col++) {
                        dot += omega[d][col] * sv[o * m + col];
                    }
                    go[d] = dot - sq[o] * v[o][d];
                    norm += go[d] * go[d];
                }
                norm = Math.sqrt(norm);
                g[o] = norm;
                if (norm < 1e-12) {
                    continue;
                }
                for (int d = 0; d < k; d++) {
                    double nv = -go[d] / norm;
                    double delta = nv - v[o][d];
                    if (delta != 0) {
                        for (int col = 0; col < m; col++) {
                            omega[d][col] += delta * sv[o * m + col];
                        }
                    }
                    v[o][d] = nv;
                }
            }
        }
    }

    private static void normalise(double[] x) {
        double s = 0;
        for (double v : x) {
            s += v * v;
        }
        s = Math.sqrt(s);
        for (int i = 0; i < x.length; i++) {
            x[i] /= s;
        }
    }

    @Override
    public List<Var> parameters() {
        return List.of(s);
    }

    @Override
    public int outputSize(int inputSize) {
        return inputSize / 2;
    }

    @Override
    public String toString() {
        return "SatNetLayer(" + n + " variables, " + s.shape().dim(1) + " clauses, " + aux + " auxiliary)";
    }
}
