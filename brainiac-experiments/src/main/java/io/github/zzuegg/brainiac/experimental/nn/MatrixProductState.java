package io.github.zzuegg.brainiac.experimental.nn;

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
 * A matrix product state (tensor train): a weight tensor over {@code sites} indices, each of
 * physical dimension {@code d}, stored as a chain of small cores {@code A_s[d, chi, chi]}
 * whose bond dimension {@code chi} sets the expressiveness. Two uses from the tensor-network
 * machine-learning papers:
 *
 * <ul>
 *   <li><b>Classifier</b> (Stoudenmire and Schwab, "Supervised learning with tensor
 *       networks", <a href="https://arxiv.org/abs/1605.05775">arXiv:1605.05775</a>): every
 *       input feature {@code x_s in [0, 1]} goes through the local feature map
 *       {@code phi(x) = [cos(pi x / 2), sin(pi x / 2)]}, the product state of those maps is
 *       contracted with the MPS, and a label core in the middle of the chain carries one
 *       output per class. As a {@link Layer} it maps {@code [batch, sites]} to
 *       {@code [batch, labels]}. Training is plain autograd on the cores (the gradient of the
 *       DMRG-style sweep without the bond-dimension adaptation); the left and right partial
 *       contractions are normalized at every site so long chains stay finite.</li>
 *   <li><b>Born machine</b> (Han et al., "Unsupervised generative modeling using matrix
 *       product states", <a href="https://arxiv.org/abs/1709.01662">arXiv:1709.01662</a>):
 *       over discrete variables {@code x_s in {0..d-1}}, {@code p(x) = |psi(x)|^2 / Z} with
 *       {@code psi(x) = A_1[x_1] ... A_N[x_N]}. The normalization {@code Z} is an exact
 *       contraction of transfer matrices ({@link #logNormalization}), so the negative
 *       log-likelihood is exact ({@link #nll}), and {@link #sample} draws exact, independent
 *       samples site by site from the conditionals, with no Markov chain.</li>
 * </ul>
 */
public final class MatrixProductState implements Layer {
    private final int sites;
    private final int physical;
    private final int bond;
    private final int labels;
    private final int labelSite;
    private final List<Var> cores = new ArrayList<>();
    private final Var labelCore;
    private final float scale;

    private MatrixProductState(int sites, int physical, int bond, int labels, RandomGenerator random) {
        if (sites < 2 || physical < 2 || bond < 1) {
            throw new IllegalArgumentException("need sites >= 2, physical >= 2, bond >= 1");
        }
        this.sites = sites;
        this.physical = physical;
        this.bond = bond;
        this.labels = labels;
        this.labelSite = sites / 2;
        for (int s = 0; s < sites; s++) {
            int in = s == 0 ? 1 : bond;
            int out = s == sites - 1 ? 1 : bond;
            float[] a = new float[physical * in * out];
            for (int k = 0; k < physical; k++) {
                for (int i = 0; i < in; i++) {
                    for (int j = 0; j < out; j++) {
                        double v = random.nextGaussian() * 0.1 / Math.sqrt(bond);
                        if (i == j || in == 1 || out == 1) {
                            v += 1.0 / Math.sqrt(physical);
                        }
                        a[(k * in + i) * out + j] = (float) v;
                    }
                }
            }
            cores.add(Var.param(Tensor.of(Shape.of(physical, in, out), a)));
        }
        if (labels > 0) {
            float[] t = new float[bond * labels * bond];
            for (int i = 0; i < t.length; i++) {
                t[i] = (float) (random.nextGaussian() * 0.5);
            }
            labelCore = Var.param(Tensor.of(Shape.of(bond, labels, bond), t));
        } else {
            labelCore = null;
        }
        this.scale = (float) Math.sqrt(bond) * 4;
    }

    /**
     * A classifier over real features in {@code [0, 1]}.
     *
     * @param sites the input features
     * @param bond the bond dimension
     * @param labels the classes
     * @param seed the seed
     * @return the MPS classifier
     */
    public static MatrixProductState classifier(int sites, int bond, int labels, long seed) {
        if (labels < 1) {
            throw new IllegalArgumentException("labels must be positive");
        }
        return new MatrixProductState(sites, 2, bond, labels, new SplittableRandom(seed));
    }

    /**
     * A Born machine over discrete variables.
     *
     * @param sites the variables
     * @param physical the values per variable (2 for bits)
     * @param bond the bond dimension
     * @param seed the seed
     * @return the generative MPS
     */
    public static MatrixProductState bornMachine(int sites, int physical, int bond, long seed) {
        return new MatrixProductState(sites, physical, bond, 0, new SplittableRandom(seed));
    }

    /**
     * Describes an MPS classifier for model builders; the input width is the number of sites.
     *
     * @param bond the bond dimension
     * @param labels the classes
     * @return the spec
     */
    public static LayerSpec spec(int bond, int labels) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return Shape.of(labels);
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return new MatrixProductState(input.dim(-1), 2, bond, labels, random);
            }
        };
    }

    /** @return the bond dimension */
    public int bond() {
        return bond;
    }

    /** @return the number of sites */
    public int sites() {
        return sites;
    }

    @Override
    public Var forward(Var input) {
        if (labelCore == null) {
            throw new IllegalStateException("a Born machine has no classifier forward pass; use nll or sample");
        }
        Shape s = input.shape();
        if (s.rank() != 2 || s.dim(1) != sites) {
            throw new IllegalArgumentException("MatrixProductState expects [batch, " + sites + "], got " + s);
        }
        int b = s.dim(0);
        Var angle = input.clamp(0, 1).mul((float) (Math.PI / 2));
        Var cos = angle.cos();
        Var sin = angle.sin();
        Var left = null;
        for (int site = 0; site < labelSite; site++) {
            left = absorb(left, local(cos, sin, site), site, b, false);
        }
        Var right = null;
        for (int site = sites - 1; site >= labelSite; site--) {
            right = absorb(right, local(cos, sin, site), site, b, true);
        }
        // left [b, 1, bond], right [b, bond, 1]
        Var mid = left.reshape(b, bond).matmul(labelCore.reshape(bond, labels * bond)).reshape(b, labels, bond);
        return mid.matmul(right).reshape(b, labels).mul(scale);
    }

    /** {@code [b, 2]} feature map of one site. */
    private static Var local(Var cos, Var sin, int site) {
        return Var.concat(1, cos.slice(1, site, site + 1), sin.slice(1, site, site + 1));
    }

    /**
     * Multiplies one site's matrix {@code sum_k phi_k A[k]} onto a running, normalized partial
     * contraction.
     */
    private Var absorb(Var acc, Var phi, int site, int b, boolean fromRight) {
        Var core = cores.get(site);
        int in = core.shape().dim(1);
        int out = core.shape().dim(2);
        Var m = phi.matmul(core.reshape(physical, in * out)).reshape(b, in, out);
        Var next;
        if (!fromRight) {
            next = acc == null ? m : acc.matmul(m);                  // [b, 1, out]
        } else {
            next = acc == null ? m : m.matmul(acc);                  // [b, in, 1]
        }
        return next.div(next.square().sum(1, true).sum(2, true).add(1e-12f).sqrt());
    }

    /**
     * The exact log-probabilities {@code log p(x) = log psi(x)^2 - log Z} of configurations.
     *
     * @param configurations values in {@code 0..physical-1}, one row per configuration
     * @return {@code [batch]} log-probabilities
     */
    public Var logProbability(int[][] configurations) {
        return logAmplitudeSquared(configurations).sub(logNormalization());
    }

    /**
     * The mean exact negative log-likelihood, in nats per configuration.
     *
     * @param configurations the data
     * @return the NLL
     */
    public Var nll(int[][] configurations) {
        return logProbability(configurations).mean().neg();
    }

    /**
     * {@code log psi(x)^2} of each configuration, unnormalized.
     *
     * @param configurations the configurations
     * @return {@code [batch]}
     */
    public Var logAmplitudeSquared(int[][] configurations) {
        int b = configurations.length;
        Var acc = null;
        Var logSum = null;
        for (int site = 0; site < sites; site++) {
            float[] onehot = new float[b * physical];
            for (int i = 0; i < b; i++) {
                onehot[i * physical + configurations[i][site]] = 1;
            }
            Var core = cores.get(site);
            int in = core.shape().dim(1);
            int out = core.shape().dim(2);
            Var m = Var.of(Tensor.of(Shape.of(b, physical), onehot)).matmul(core.reshape(physical, in * out))
                    .reshape(b, in, out);
            Var next = acc == null ? m : acc.matmul(m);
            Var norm2 = next.square().sum(1, true).sum(2, true).add(1e-30f);
            Var log = norm2.log().reshape(b);
            logSum = logSum == null ? log : logSum.add(log);
            acc = next.div(norm2.sqrt());
        }
        return logSum;
    }

    /**
     * The exact {@code log Z = log sum_x psi(x)^2}, contracted with transfer matrices
     * {@code L <- sum_k A[k]^T L A[k]} and renormalized at each site.
     *
     * @return the scalar {@code log Z}
     */
    public Var logNormalization() {
        Var env = null;
        Var logZ = null;
        for (int site = 0; site < sites; site++) {
            Var core = cores.get(site);
            int in = core.shape().dim(1);
            int out = core.shape().dim(2);
            Var next = null;
            for (int k = 0; k < physical; k++) {
                Var a = core.slice(0, k, k + 1).reshape(in, out);
                Var term = env == null ? a.transpose().matmul(a) : a.transpose().matmul(env).matmul(a);
                next = next == null ? term : next.add(term);
            }
            Var c = next.abs().sum().add(1e-30f);
            Var log = c.log();
            logZ = logZ == null ? log : logZ.add(log);
            env = next.div(c);
        }
        // env is now [1, 1] and holds Z / prod(c)
        return logZ.add(env.reshape(1).log().sum());
    }

    /**
     * Draws exact samples from the Born distribution, site by site from the conditionals
     * {@code p(x_s | x_<s)}, using the right environments {@code R_s = sum_k A[k] R_(s+1) A[k]^T}.
     *
     * @param n how many
     * @param random the random source
     * @return {@code n} configurations
     */
    public int[][] sample(int n, RandomGenerator random) {
        double[][][] a = new double[sites][][];
        for (int s = 0; s < sites; s++) {
            float[] v = cores.get(s).value().toArray();
            a[s] = new double[][] {toDouble(v)};
        }
        // right environments, renormalized (only ratios matter)
        double[][][] right = new double[sites + 1][][];
        right[sites] = new double[][] {{1}};
        for (int s = sites - 1; s >= 0; s--) {
            int in = dimIn(s);
            int out = dimOut(s);
            double[][] r = new double[in][in];
            double total = 0;
            for (int k = 0; k < physical; k++) {
                for (int i = 0; i < in; i++) {
                    for (int j = 0; j < in; j++) {
                        double acc = 0;
                        for (int p = 0; p < out; p++) {
                            for (int q = 0; q < out; q++) {
                                acc += at(a, s, k, i, p) * right[s + 1][p][q] * at(a, s, k, j, q);
                            }
                        }
                        r[i][j] += acc;
                    }
                }
            }
            for (double[] row : r) {
                for (double v : row) {
                    total += Math.abs(v);
                }
            }
            for (double[] row : r) {
                for (int j = 0; j < row.length; j++) {
                    row[j] /= total;
                }
            }
            right[s] = r;
        }
        int[][] out = new int[n][sites];
        for (int i = 0; i < n; i++) {
            double[] left = {1};
            for (int s = 0; s < sites; s++) {
                int dout = dimOut(s);
                double[] probs = new double[physical];
                double[][] cand = new double[physical][dout];
                double sum = 0;
                for (int k = 0; k < physical; k++) {
                    for (int q = 0; q < dout; q++) {
                        double acc = 0;
                        for (int p = 0; p < left.length; p++) {
                            acc += left[p] * at(a, s, k, p, q);
                        }
                        cand[k][q] = acc;
                    }
                    double pr = 0;
                    for (int p = 0; p < dout; p++) {
                        for (int q = 0; q < dout; q++) {
                            pr += cand[k][p] * right[s + 1][p][q] * cand[k][q];
                        }
                    }
                    probs[k] = Math.max(0, pr);
                    sum += probs[k];
                }
                double u = random.nextDouble() * sum;
                int pick = physical - 1;
                for (int k = 0; k < physical; k++) {
                    u -= probs[k];
                    if (u <= 0) {
                        pick = k;
                        break;
                    }
                }
                out[i][s] = pick;
                double norm = 0;
                for (double v : cand[pick]) {
                    norm += v * v;
                }
                norm = Math.sqrt(norm) + 1e-300;
                left = new double[dout];
                for (int q = 0; q < dout; q++) {
                    left[q] = cand[pick][q] / norm;
                }
            }
        }
        return out;
    }

    /**
     * One two-site DMRG-style learning sweep of the Born machine (Han et al. 2018): the chain is
     * brought into right-canonical form, then for each bond from left to right the merged
     * two-site tensor {@code B = A_s A_(s+1)} takes gradient steps on the exact NLL, which in
     * canonical form is {@code ln ||B||^2 - mean_x ln psi(x)^2}, and is split back by a truncated
     * SVD that keeps at most the bond dimension (left factor orthonormal, so the canonical
     * centre moves right). Far better conditioned than plain gradient descent on all cores.
     *
     * @param data the training configurations
     * @param learningRate the step size on the normalized two-site tensor
     * @param stepsPerBond gradient steps per bond
     */
    public void dmrgSweep(int[][] data, double learningRate, int stepsPerBond) {
        int n = data.length;
        double[][] a = new double[sites][];
        for (int s = 0; s < sites; s++) {
            a[s] = toDouble(cores.get(s).value().toArray());
        }
        // right-canonicalize from the last site: A_s[in, d*out] = U S Vt, A_s <- Vt, A_(s-1) <- A_(s-1) U S
        for (int s = sites - 1; s > 0; s--) {
            int in = dimIn(s);
            int out = dimOut(s);
            double[][] m = new double[in][physical * out];
            for (int k = 0; k < physical; k++) {
                for (int i = 0; i < in; i++) {
                    for (int j = 0; j < out; j++) {
                        m[i][k * out + j] = a[s][(k * in + i) * out + j];
                    }
                }
            }
            double[][][] usv = svd(m, in);
            double[][] u = usv[0];
            double[] sv = usv[1][0];
            double[][] vt = usv[2];
            for (int k = 0; k < physical; k++) {
                for (int i = 0; i < in; i++) {
                    for (int j = 0; j < out; j++) {
                        a[s][(k * in + i) * out + j] = vt[i][k * out + j];
                    }
                }
            }
            int pin = dimIn(s - 1);
            double[] prev = a[s - 1].clone();
            for (int k = 0; k < physical; k++) {
                for (int i = 0; i < pin; i++) {
                    for (int j = 0; j < in; j++) {
                        double acc = 0;
                        for (int q = 0; q < in; q++) {
                            acc += prev[(k * pin + i) * in + q] * u[q][j] * sv[j];
                        }
                        a[s - 1][(k * pin + i) * in + j] = acc;
                    }
                }
            }
        }
        // right environments of every sample: R_s(x) = A_s[x_s] ... A_(N-1)[x_(N-1)], a vector over in(s)
        double[][][] right = new double[sites + 1][][];
        right[sites] = new double[n][];
        for (int i = 0; i < n; i++) {
            right[sites][i] = new double[] {1};
        }
        for (int s = sites - 1; s >= 1; s--) {
            int in = dimIn(s);
            int out = dimOut(s);
            right[s] = new double[n][in];
            for (int i = 0; i < n; i++) {
                int k = data[i][s];
                for (int p = 0; p < in; p++) {
                    double acc = 0;
                    for (int q = 0; q < out; q++) {
                        acc += a[s][(k * in + p) * out + q] * right[s + 1][i][q];
                    }
                    right[s][i][p] = acc;
                }
            }
        }
        double[][] left = new double[n][];
        for (int i = 0; i < n; i++) {
            left[i] = new double[] {1};
        }
        for (int s = 0; s < sites - 1; s++) {
            int l = dimIn(s);
            int mid = dimOut(s);
            int r = dimOut(s + 1);
            int d = physical;
            // merged tensor B[a, k1, k2, b]
            double[] b = new double[l * d * d * r];
            for (int k1 = 0; k1 < d; k1++) {
                for (int k2 = 0; k2 < d; k2++) {
                    for (int p = 0; p < l; p++) {
                        for (int q = 0; q < r; q++) {
                            double acc = 0;
                            for (int c = 0; c < mid; c++) {
                                acc += a[s][(k1 * l + p) * mid + c] * a[s + 1][(k2 * mid + c) * r + q];
                            }
                            b[((p * d + k1) * d + k2) * r + q] = acc;
                        }
                    }
                }
            }
            scale(b, 1 / Math.sqrt(dot(b, b)));
            for (int step = 0; step < stepsPerBond; step++) {
                double[] grad = new double[b.length];
                for (int i = 0; i < n; i++) {
                    int k1 = data[i][s];
                    int k2 = data[i][s + 1];
                    double[] lv = left[i];
                    double[] rv = right[s + 2][i];
                    double psi = 0;
                    for (int p = 0; p < l; p++) {
                        for (int q = 0; q < r; q++) {
                            psi += lv[p] * b[((p * d + k1) * d + k2) * r + q] * rv[q];
                        }
                    }
                    double w = 2.0 / n / (Math.abs(psi) < 1e-150 ? Math.copySign(1e-150, psi) : psi);
                    for (int p = 0; p < l; p++) {
                        for (int q = 0; q < r; q++) {
                            grad[((p * d + k1) * d + k2) * r + q] -= w * lv[p] * rv[q];
                        }
                    }
                }
                for (int t = 0; t < b.length; t++) {
                    grad[t] += 2 * b[t];          // d ln ||B||^2 / dB with ||B|| = 1
                }
                double gn = Math.sqrt(dot(grad, grad));
                double stepSize = learningRate * Math.min(1, 1 / Math.max(gn, 1e-12));
                for (int t = 0; t < b.length; t++) {
                    b[t] -= stepSize * grad[t];
                }
                scale(b, 1 / Math.sqrt(dot(b, b)));
            }
            // split B[(a k1), (k2 b)] = U S Vt, keep at most the bond dimension
            double[][] m = new double[l * d][d * r];
            for (int p = 0; p < l; p++) {
                for (int k1 = 0; k1 < d; k1++) {
                    for (int k2 = 0; k2 < d; k2++) {
                        for (int q = 0; q < r; q++) {
                            m[p * d + k1][k2 * r + q] = b[((p * d + k1) * d + k2) * r + q];
                        }
                    }
                }
            }
            double[][][] usv = svd(m, mid);
            double[][] u = usv[0];
            double[] sv = usv[1][0];
            double[][] vt = usv[2];
            for (int k1 = 0; k1 < d; k1++) {
                for (int p = 0; p < l; p++) {
                    for (int c = 0; c < mid; c++) {
                        a[s][(k1 * l + p) * mid + c] = u[p * d + k1][c];
                    }
                }
            }
            for (int k2 = 0; k2 < d; k2++) {
                for (int c = 0; c < mid; c++) {
                    for (int q = 0; q < r; q++) {
                        a[s + 1][(k2 * mid + c) * r + q] = sv[c] * vt[c][k2 * r + q];
                    }
                }
            }
            for (int i = 0; i < n; i++) {
                int k = data[i][s];
                double[] nl = new double[mid];
                for (int c = 0; c < mid; c++) {
                    double acc = 0;
                    for (int p = 0; p < l; p++) {
                        acc += left[i][p] * a[s][(k * l + p) * mid + c];
                    }
                    nl[c] = acc;
                }
                left[i] = nl;
            }
        }
        for (int s = 0; s < sites; s++) {
            float[] f = new float[a[s].length];
            for (int t = 0; t < f.length; t++) {
                f[t] = (float) a[s][t];
            }
            cores.get(s).assign(Tensor.of(cores.get(s).shape(), f));
        }
    }

    private static double dot(double[] x, double[] y) {
        double s = 0;
        for (int i = 0; i < x.length; i++) {
            s += x[i] * y[i];
        }
        return s;
    }

    private static void scale(double[] x, double f) {
        for (int i = 0; i < x.length; i++) {
            x[i] *= f;
        }
    }

    /**
     * A thin SVD {@code M = U S Vt} truncated (and zero-padded) to exactly {@code rank}
     * components, from the eigendecomposition of {@code M M^T} by cyclic Jacobi rotations.
     *
     * @return {@code {U [m][rank], {S [rank]}, Vt [rank][n]}}
     */
    private static double[][][] svd(double[][] m, int rank) {
        int rows = m.length;
        int cols = m[0].length;
        double[][] g = new double[rows][rows];
        for (int i = 0; i < rows; i++) {
            for (int j = i; j < rows; j++) {
                double acc = dot(m[i], m[j]);
                g[i][j] = acc;
                g[j][i] = acc;
            }
        }
        double[][] vec = new double[rows][rows];
        for (int i = 0; i < rows; i++) {
            vec[i][i] = 1;
        }
        double scaleSq = 0;
        for (int i = 0; i < rows; i++) {
            scaleSq += g[i][i] * g[i][i];
        }
        for (int sweep = 0; sweep < 60; sweep++) {
            double off = 0;
            for (int p = 0; p < rows; p++) {
                for (int q = p + 1; q < rows; q++) {
                    off += g[p][q] * g[p][q];
                }
            }
            if (off <= 1e-28 * scaleSq) {
                break;
            }
            for (int p = 0; p < rows; p++) {
                for (int q = p + 1; q < rows; q++) {
                    if (g[p][q] == 0) {
                        continue;
                    }
                    double theta = (g[q][q] - g[p][p]) / (2 * g[p][q]);
                    double t = (theta >= 0 ? 1 : -1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
                    double c = 1 / Math.sqrt(t * t + 1);
                    double sn = t * c;
                    for (int k = 0; k < rows; k++) {
                        double gkp = g[k][p];
                        double gkq = g[k][q];
                        g[k][p] = c * gkp - sn * gkq;
                        g[k][q] = sn * gkp + c * gkq;
                    }
                    for (int k = 0; k < rows; k++) {
                        double gpk = g[p][k];
                        double gqk = g[q][k];
                        g[p][k] = c * gpk - sn * gqk;
                        g[q][k] = sn * gpk + c * gqk;
                    }
                    for (int k = 0; k < rows; k++) {
                        double vkp = vec[k][p];
                        double vkq = vec[k][q];
                        vec[k][p] = c * vkp - sn * vkq;
                        vec[k][q] = sn * vkp + c * vkq;
                    }
                }
            }
        }
        Integer[] order = new Integer[rows];
        for (int i = 0; i < rows; i++) {
            order[i] = i;
        }
        java.util.Arrays.sort(order, (x, y) -> Double.compare(g[y][y], g[x][x]));
        double[][] u = new double[rows][rank];
        double[] sv = new double[rank];
        double[][] vt = new double[rank][cols];
        double top = Math.max(g[order[0]][order[0]], 0);
        for (int c = 0; c < Math.min(rank, rows); c++) {
            int e = order[c];
            double lambda = g[e][e];
            if (lambda <= top * 1e-24 || lambda <= 0) {
                break;
            }
            double sigma = Math.sqrt(lambda);
            sv[c] = sigma;
            for (int i = 0; i < rows; i++) {
                u[i][c] = vec[i][e];
            }
            for (int j = 0; j < cols; j++) {
                double acc = 0;
                for (int i = 0; i < rows; i++) {
                    acc += vec[i][e] * m[i][j];
                }
                vt[c][j] = acc / sigma;
            }
        }
        return new double[][][] {u, {sv}, vt};
    }

    private int dimIn(int s) {
        return s == 0 ? 1 : bond;
    }

    private int dimOut(int s) {
        return s == sites - 1 ? 1 : bond;
    }

    private double at(double[][][] a, int s, int k, int i, int j) {
        return a[s][0][(k * dimIn(s) + i) * dimOut(s) + j];
    }

    private static double[] toDouble(float[] v) {
        double[] d = new double[v.length];
        for (int i = 0; i < v.length; i++) {
            d[i] = v[i];
        }
        return d;
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(cores);
        if (labelCore != null) {
            p.add(labelCore);
        }
        return p;
    }

    /** @return the number of trained weights */
    public int parameterCount() {
        long n = 0;
        for (Var v : parameters()) {
            n += v.value().size();
        }
        return Math.toIntExact(n);
    }

    @Override
    public int outputSize(int inputSize) {
        return labels;
    }
}
