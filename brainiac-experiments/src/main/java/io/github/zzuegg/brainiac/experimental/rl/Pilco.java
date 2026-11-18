package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.experimental.evolve.CmaEs;
import io.github.zzuegg.brainiac.experimental.ml.GaussianProcess;
import io.github.zzuegg.brainiac.experimental.ml.StationaryKernel;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * PILCO, probabilistic inference for learning control: a data-efficient model-based policy
 * search. Each iteration
 * <ol>
 *   <li>fits one Gaussian-process model per state dimension to the observed transitions
 *       {@code (x, u) -> x' - x} (squared-exponential kernel with per-input lengthscales: the
 *       inputs are standardised and the isotropic {@link GaussianProcess} hyperparameters are
 *       fitted there by marginal likelihood, which is an automatic-relevance kernel in the
 *       original units);</li>
 *   <li>propagates a Gaussian state distribution through the model and a linear policy
 *       {@code u = K x + c} for the whole horizon by exact moment matching (the analytic mean,
 *       covariance and input-output covariance of a GP prediction at a Gaussian input), and
 *       sums the expected saturating cost
 *       {@code 1 - exp(-(x - x*)^T W (x - x*) / 2)} in closed form;</li>
 *   <li>minimises that expected cost over the policy parameters (here with CMA-ES on the
 *       analytic objective rather than its analytic gradient, as the policy is small), and</li>
 *   <li>runs the policy once on the real system, adding the transitions to the data.</li>
 * </ol>
 * The policy acts every {@code repeat} system steps and holds its action in between; the model
 * predicts those macro steps. Action limits are applied on the real system but not inside the
 * moment-matched prediction.
 *
 * <p>Deisenroth and Rasmussen 2011, "PILCO: A Model-Based and Data-Efficient Approach to
 * Policy Search", ICML.
 */
public final class Pilco {

    private final ControlSystem system;
    private final double[] startMean;
    private final double[] startStd;
    private final int repeat;
    private final int horizon;
    private final int n;
    private final int m;
    private final SplittableRandom random;
    private final List<double[]> inputs = new ArrayList<>();
    private final List<double[]> targets = new ArrayList<>();
    private double[] target;
    private double[] weight;
    private int maxData = 60;
    private double[] policy;
    private long realSteps;
    private Model model;

    private Pilco(ControlSystem system, double[] startMean, double[] startStd, int repeat, int horizon, long seed) {
        this.system = Objects.requireNonNull(system);
        this.startMean = startMean.clone();
        this.startStd = startStd.clone();
        this.repeat = repeat;
        this.horizon = horizon;
        this.n = system.stateSize();
        this.m = system.actionSize();
        this.random = new SplittableRandom(seed);
        this.target = new double[n];
        this.weight = new double[n];
        this.policy = new double[m * (n + 1)];
    }

    /**
     * @param system the real system (only its {@code step} and {@code cost} are used, as a black box)
     * @param startMean the mean of the Gaussian start state
     * @param startStd its per-dimension standard deviations
     * @param repeat system steps per policy decision
     * @param horizon policy decisions per episode
     * @param seed the seed
     * @return a PILCO learner with a zero policy
     */
    public static Pilco of(ControlSystem system, double[] startMean, double[] startStd, int repeat, int horizon,
            long seed) {
        return new Pilco(system, startMean, startStd, repeat, horizon, seed);
    }

    /**
     * @param goal the target state {@code x*}
     * @param weights the diagonal of {@code W} (inverse squared widths; 0 ignores a dimension)
     * @return this
     */
    public Pilco saturatingCost(double[] goal, double[] weights) {
        this.target = goal.clone();
        this.weight = weights.clone();
        return this;
    }

    /**
     * @param size the most transitions the GP is conditioned on (a random subset beyond that)
     * @return this
     */
    public Pilco maxData(int size) {
        this.maxData = size;
        return this;
    }

    /** @return real system steps taken so far */
    public long realSteps() {
        return realSteps;
    }

    /** @return the policy parameters, {@code K} row by row then {@code c} */
    public double[] policy() {
        return policy.clone();
    }

    private double[] act(double[] w, double[] x) {
        double[] u = new double[m];
        double[] lo = system.actionLow();
        double[] hi = system.actionHigh();
        for (int a = 0; a < m; a++) {
            double v = w[m * n + a];
            for (int j = 0; j < n; j++) {
                v += w[a * n + j] * x[j];
            }
            u[a] = Math.max(lo[a], Math.min(hi[a], v));
        }
        return u;
    }

    /**
     * Runs one real episode with the current policy, or uniformly random actions.
     *
     * @param randomActions act uniformly at random (the initial data)
     * @param record whether to add the transitions to the data
     * @param r the start-state randomness
     * @return the episode's total system cost
     */
    public double episode(boolean randomActions, boolean record, SplittableRandom r) {
        double[] x = new double[n];
        for (int j = 0; j < n; j++) {
            x[j] = startMean[j] + startStd[j] * r.nextGaussian();
        }
        double cost = 0;
        for (int t = 0; t < horizon; t++) {
            double[] u;
            if (randomActions) {
                u = new double[m];
                for (int a = 0; a < m; a++) {
                    u[a] = r.nextDouble(system.actionLow()[a], system.actionHigh()[a]);
                }
            } else {
                u = act(policy, x);
            }
            double[] y = x;
            for (int k = 0; k < repeat; k++) {
                cost += system.cost(y, u);
                y = system.step(y, u);
            }
            if (record) {
                realSteps += repeat;
                double[] in = new double[n + m];
                System.arraycopy(x, 0, in, 0, n);
                System.arraycopy(u, 0, in, n, m);
                double[] d = new double[n];
                for (int j = 0; j < n; j++) {
                    d[j] = y[j] - x[j];
                }
                inputs.add(in);
                targets.add(d);
            }
            x = y;
        }
        return cost;
    }

    /**
     * Collects random-action episodes as the initial data.
     *
     * @param episodes the number of episodes
     * @return this
     */
    public Pilco randomEpisodes(int episodes) {
        for (int e = 0; e < episodes; e++) {
            episode(true, true, random.split());
        }
        return this;
    }

    /**
     * Runs PILCO iterations: model fit, policy optimisation, one real episode each.
     *
     * @param iterations iterations (= real episodes)
     * @param evaluations CMA-ES evaluations of the analytic objective per iteration
     * @return this
     */
    public Pilco learn(int iterations, int evaluations) {
        for (int it = 0; it < iterations; it++) {
            model = fitModel();
            policy = CmaEs.of(policy, 0.3).seed(random.nextLong()).minimize(this::expectedCost, evaluations);
            episode(false, true, random.split());
        }
        return this;
    }

    /**
     * The model's predicted (moment-matched) expected saturating cost of a policy, summed over
     * the horizon; requires a fitted model (after {@link #learn}).
     *
     * @param w policy parameters
     * @return the expected cost
     */
    public double expectedCost(double[] w) {
        double[] mu = startMean.clone();
        double[][] sigma = new double[n][n];
        for (int j = 0; j < n; j++) {
            sigma[j][j] = startStd[j] * startStd[j];
        }
        double total = 0;
        for (int t = 0; t < horizon; t++) {
            Object[] next = model.propagate(mu, sigma, w);
            mu = (double[]) next[0];
            sigma = (double[][]) next[1];
            double c = saturating(mu, sigma);
            if (!Double.isFinite(c)) {
                return horizon;
            }
            total += c;
        }
        return total;
    }

    private double saturating(double[] mu, double[][] sigma) {
        // E[1 - exp(-e^T W e / 2)] = 1 - |I + S W|^(-1/2) exp(-e^T W (I + S W)^-1 e / 2)
        double[][] s = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                s[i][j] = (i == j ? 1 : 0) + sigma[i][j] * weight[j];
            }
        }
        double det = determinant(s);
        double[][] inv = inverse(s);
        double quad = 0;
        for (int i = 0; i < n; i++) {
            double e = mu[i] - target[i];
            double row = 0;
            for (int j = 0; j < n; j++) {
                row += inv[i][j] * (mu[j] - target[j]);
            }
            quad += e * weight[i] * row;
        }
        return 1 - Math.exp(-0.5 * quad) / Math.sqrt(det);
    }

    /** One GP per output dimension on standardised inputs. */
    private Model fitModel() {
        int total = inputs.size();
        int size = Math.min(maxData, total);
        int[] idx = new int[total];
        for (int i = 0; i < total; i++) {
            idx[i] = i;
        }
        for (int i = 0; i < size; i++) {
            int j = i + random.nextInt(total - i);
            int tmp = idx[i];
            idx[i] = idx[j];
            idx[j] = tmp;
        }
        int d = n + m;
        double[][] z = new double[size][];
        for (int i = 0; i < size; i++) {
            z[i] = inputs.get(idx[i]);
        }
        double[] scale = new double[d];
        double[] mean = new double[d];
        for (int k = 0; k < d; k++) {
            for (double[] row : z) {
                mean[k] += row[k] / size;
            }
            for (double[] row : z) {
                scale[k] += (row[k] - mean[k]) * (row[k] - mean[k]) / size;
            }
            scale[k] = Math.sqrt(scale[k]) + 1e-6;
        }
        double[][] zs = new double[size][d];
        for (int i = 0; i < size; i++) {
            for (int k = 0; k < d; k++) {
                zs[i][k] = (z[i][k] - mean[k]) / scale[k];
            }
        }
        double[][] lambda = new double[n][d];
        double[] signal = new double[n];
        double[] noise = new double[n];
        double[][] beta = new double[n][];
        double[][][] kinv = new double[n][][];
        double[] offset = new double[n];
        for (int a = 0; a < n; a++) {
            double[] y = new double[size];
            double ym = 0;
            for (int i = 0; i < size; i++) {
                y[i] = targets.get(idx[i])[a];
                ym += y[i] / size;
            }
            double var = 0;
            for (double v : y) {
                var += (v - ym) * (v - ym) / size;
            }
            var = Math.max(var, 1e-10);
            GaussianProcess gp = GaussianProcess.optimize(zs, y, StationaryKernel.rbf(1, var), 0.01 * var, 60);
            double ell = gp.kernel().lengthscale();
            signal[a] = gp.kernel().variance();
            noise[a] = gp.noise();
            for (int k = 0; k < d; k++) {
                lambda[a][k] = ell * ell * scale[k] * scale[k];
            }
            double[][] kk = new double[size][size];
            for (int i = 0; i < size; i++) {
                for (int j = 0; j < size; j++) {
                    kk[i][j] = signal[a] * Math.exp(-0.5 * scaledDistance(z[i], z[j], lambda[a]))
                            + (i == j ? noise[a] : 0);
                }
            }
            kinv[a] = inverse(kk);
            beta[a] = new double[size];
            for (int i = 0; i < size; i++) {
                for (int j = 0; j < size; j++) {
                    beta[a][i] += kinv[a][i][j] * (y[j] - ym);
                }
            }
            offset[a] = ym;
        }
        return new Model(z, lambda, signal, beta, kinv, offset);
    }

    private static double scaledDistance(double[] a, double[] b, double[] lambda) {
        double s = 0;
        for (int k = 0; k < a.length; k++) {
            double v = a[k] - b[k];
            s += v * v / lambda[k];
        }
        return s;
    }

    /** The fitted GP dynamics and its moment-matching prediction. */
    private final class Model {
        private final double[][] z;
        private final double[][] lambda;
        private final double[] signal;
        private final double[][] beta;
        private final double[][][] kinv;
        private final double[] offset;

        Model(double[][] z, double[][] lambda, double[] signal, double[][] beta, double[][][] kinv,
                double[] offset) {
            this.z = z;
            this.lambda = lambda;
            this.signal = signal;
            this.beta = beta;
            this.kinv = kinv;
            this.offset = offset;
        }

        /** @return {mean, covariance} of the next state */
        Object[] propagate(double[] mu, double[][] sigma, double[] w) {
            int d = n + m;
            int size = z.length;
            // joint Gaussian of (x, u = K x + c)
            double[] muz = new double[d];
            double[][] sz = new double[d][d];
            System.arraycopy(mu, 0, muz, 0, n);
            double[][] kx = new double[m][n];
            for (int a = 0; a < m; a++) {
                muz[n + a] = w[m * n + a];
                for (int j = 0; j < n; j++) {
                    kx[a][j] = w[a * n + j];
                    muz[n + a] += kx[a][j] * mu[j];
                }
            }
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    sz[i][j] = sigma[i][j];
                }
                for (int a = 0; a < m; a++) {
                    double v = 0;
                    for (int k = 0; k < n; k++) {
                        v += sigma[i][k] * kx[a][k];
                    }
                    sz[i][n + a] = v;
                    sz[n + a][i] = v;
                }
            }
            for (int a = 0; a < m; a++) {
                for (int b = 0; b < m; b++) {
                    double v = 0;
                    for (int i = 0; i < n; i++) {
                        v += kx[a][i] * sz[i][n + b];
                    }
                    sz[n + a][n + b] = v;
                }
            }
            double[][] nu = new double[size][d];
            for (int i = 0; i < size; i++) {
                for (int k = 0; k < d; k++) {
                    nu[i][k] = z[i][k] - muz[k];
                }
            }
            double[] meanDelta = new double[n];
            double[][] cross = new double[d][n];
            double[][] q = new double[n][size];
            double[][] kmu = new double[n][size];
            for (int a = 0; a < n; a++) {
                double[][] t = new double[d][d];
                for (int i = 0; i < d; i++) {
                    for (int j = 0; j < d; j++) {
                        t[i][j] = sz[i][j] + (i == j ? lambda[a][i] : 0);
                    }
                }
                double[][] tinv = inverse(t);
                double detLambda = 1;
                for (int k = 0; k < d; k++) {
                    detLambda *= lambda[a][k];
                }
                double c = signal[a] * Math.sqrt(detLambda / determinant(t));
                double[] acc = new double[d];
                for (int i = 0; i < size; i++) {
                    double quad = 0;
                    for (int k = 0; k < d; k++) {
                        double row = 0;
                        for (int l = 0; l < d; l++) {
                            row += tinv[k][l] * nu[i][l];
                        }
                        quad += nu[i][k] * row;
                    }
                    q[a][i] = c * Math.exp(-0.5 * quad);
                    meanDelta[a] += beta[a][i] * q[a][i];
                    kmu[a][i] = signal[a] * Math.exp(-0.5 * scaledDistance(z[i], muz, lambda[a]));
                    for (int k = 0; k < d; k++) {
                        acc[k] += beta[a][i] * q[a][i] * nu[i][k];
                    }
                }
                // Cov[z, delta_a] = Sigma (Sigma + Lambda)^-1 sum_i beta_i q_i nu_i
                double[] tv = new double[d];
                for (int k = 0; k < d; k++) {
                    for (int l = 0; l < d; l++) {
                        tv[k] += tinv[k][l] * acc[l];
                    }
                }
                for (int k = 0; k < d; k++) {
                    for (int l = 0; l < d; l++) {
                        cross[k][a] += sz[k][l] * tv[l];
                    }
                }
            }
            double[][] covDelta = new double[n][n];
            for (int a = 0; a < n; a++) {
                for (int b = a; b < n; b++) {
                    double[][] r = new double[d][d];
                    for (int i = 0; i < d; i++) {
                        for (int j = 0; j < d; j++) {
                            r[i][j] = sz[i][j] * (1 / lambda[a][j] + 1 / lambda[b][j]) + (i == j ? 1 : 0);
                        }
                    }
                    double[][] rinv = inverse(r);
                    double[][] mm = new double[d][d];
                    for (int i = 0; i < d; i++) {
                        for (int j = 0; j < d; j++) {
                            for (int k = 0; k < d; k++) {
                                mm[i][j] += rinv[i][k] * sz[k][j];
                            }
                        }
                    }
                    double norm = 1 / Math.sqrt(determinant(r));
                    double[][] va = new double[size][d];
                    double[][] vb = new double[size][d];
                    double[][] mva = new double[size][d];
                    double[] selfA = new double[size];
                    double[] selfB = new double[size];
                    for (int i = 0; i < size; i++) {
                        for (int k = 0; k < d; k++) {
                            va[i][k] = nu[i][k] / lambda[a][k];
                            vb[i][k] = nu[i][k] / lambda[b][k];
                        }
                        for (int k = 0; k < d; k++) {
                            for (int l = 0; l < d; l++) {
                                mva[i][k] += mm[k][l] * va[i][l];
                            }
                        }
                        double sa = 0;
                        double sb = 0;
                        for (int k = 0; k < d; k++) {
                            sa += va[i][k] * mva[i][k];
                            double mb = 0;
                            for (int l = 0; l < d; l++) {
                                mb += mm[k][l] * vb[i][l];
                            }
                            sb += vb[i][k] * mb;
                        }
                        selfA[i] = Math.log(kmu[a][i]) + 0.5 * sa;
                        selfB[i] = Math.log(kmu[b][i]) + 0.5 * sb;
                    }
                    double bqb = 0;
                    double trace = 0;
                    for (int i = 0; i < size; i++) {
                        for (int j = 0; j < size; j++) {
                            double cr = 0;
                            for (int k = 0; k < d; k++) {
                                cr += mva[i][k] * vb[j][k];
                            }
                            double qij = norm * Math.exp(selfA[i] + selfB[j] + cr);
                            bqb += beta[a][i] * qij * beta[b][j];
                            if (a == b) {
                                trace += kinv[a][j][i] * qij;
                            }
                        }
                    }
                    double v = bqb - meanDelta[a] * meanDelta[b];
                    if (a == b) {
                        v += signal[a] - trace;
                    }
                    covDelta[a][b] = v;
                    covDelta[b][a] = v;
                }
            }
            double[] nextMu = new double[n];
            double[][] nextSigma = new double[n][n];
            for (int i = 0; i < n; i++) {
                nextMu[i] = mu[i] + meanDelta[i] + offset[i];
                for (int j = 0; j < n; j++) {
                    nextSigma[i][j] = sigma[i][j] + covDelta[i][j] + cross[i][j] + cross[j][i];
                }
            }
            for (int i = 0; i < n; i++) {
                for (int j = i + 1; j < n; j++) {
                    double v = 0.5 * (nextSigma[i][j] + nextSigma[j][i]);
                    nextSigma[i][j] = v;
                    nextSigma[j][i] = v;
                }
                nextSigma[i][i] = Math.max(nextSigma[i][i], 1e-10);
            }
            return new Object[] {nextMu, nextSigma};
        }
    }

    /** Gauss-Jordan inverse with partial pivoting. */
    static double[][] inverse(double[][] a) {
        int k = a.length;
        double[][] w = new double[k][2 * k];
        for (int i = 0; i < k; i++) {
            System.arraycopy(a[i], 0, w[i], 0, k);
            w[i][k + i] = 1;
        }
        for (int c = 0; c < k; c++) {
            int p = c;
            for (int r = c + 1; r < k; r++) {
                if (Math.abs(w[r][c]) > Math.abs(w[p][c])) {
                    p = r;
                }
            }
            double[] tmp = w[c];
            w[c] = w[p];
            w[p] = tmp;
            double piv = w[c][c];
            if (Math.abs(piv) < 1e-300) {
                piv = 1e-300;
            }
            for (int j = 0; j < 2 * k; j++) {
                w[c][j] /= piv;
            }
            for (int r = 0; r < k; r++) {
                if (r != c && w[r][c] != 0) {
                    double f = w[r][c];
                    for (int j = 0; j < 2 * k; j++) {
                        w[r][j] -= f * w[c][j];
                    }
                }
            }
        }
        double[][] out = new double[k][k];
        for (int i = 0; i < k; i++) {
            System.arraycopy(w[i], k, out[i], 0, k);
        }
        return out;
    }

    /** Determinant by Gaussian elimination with partial pivoting. */
    static double determinant(double[][] a) {
        int k = a.length;
        double[][] w = new double[k][];
        for (int i = 0; i < k; i++) {
            w[i] = a[i].clone();
        }
        double det = 1;
        for (int c = 0; c < k; c++) {
            int p = c;
            for (int r = c + 1; r < k; r++) {
                if (Math.abs(w[r][c]) > Math.abs(w[p][c])) {
                    p = r;
                }
            }
            if (p != c) {
                double[] tmp = w[c];
                w[c] = w[p];
                w[p] = tmp;
                det = -det;
            }
            det *= w[c][c];
            if (w[c][c] == 0) {
                return 0;
            }
            for (int r = c + 1; r < k; r++) {
                double f = w[r][c] / w[c][c];
                for (int j = c; j < k; j++) {
                    w[r][j] -= f * w[c][j];
                }
            }
        }
        return det;
    }
}
