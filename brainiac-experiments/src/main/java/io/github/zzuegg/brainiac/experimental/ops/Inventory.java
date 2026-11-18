package io.github.zzuegg.brainiac.experimental.ops;

import io.github.zzuegg.brainiac.experimental.data.OperationsSims;
import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Space;

import java.util.Arrays;
import java.util.SplittableRandom;

/**
 * Inventory control:
 *
 * <ul>
 *   <li>the feature-based newsvendor: order the critical-ratio quantile {@code tau = b / (b + h)}
 *       of demand. {@link #newsvendorErm} fits that quantile directly by empirical risk
 *       minimisation of the pinball loss (Ban &amp; Rudin, Operations Research 2019), linearly or on
 *       random Fourier features (the kernel variant); {@link #predictThenOptimise} fits the mean by
 *       least squares and adds a normal quantile of the residuals, the classic two-step that
 *       suffers when the demand model is misspecified;</li>
 *   <li>multi-period lost-sales inventory with a lead time ({@link OperationsSims.Inventory}):
 *       {@link #baseStock} and {@link #sS} policies with their parameters found by simulation
 *       search, the {@link #optimal} policy by value iteration on the full state (on-hand stock and
 *       the pipeline of outstanding orders; feasible up to lead time 3), and {@link Env}, the problem
 *       as a core {@link Environment} for RL (PPO in Gijsbrechts et al., MSOM 2022).</li>
 * </ul>
 */
public final class Inventory {

    private Inventory() {
    }

    // ------------------------------------------------------------------ newsvendor

    /** A fitted order-quantity model. */
    @FunctionalInterface
    public interface QuantityModel {
        /**
         * @param x the features
         * @return the order quantity
         */
        double order(double[] x);
    }

    /**
     * The newsvendor cost of an order.
     *
     * @param order the quantity ordered
     * @param demand the realised demand
     * @param tau the critical ratio (underage cost {@code tau}, overage cost {@code 1 - tau})
     * @return the cost
     */
    public static double newsvendorCost(double order, double demand, double tau) {
        return demand > order ? tau * (demand - order) : (1 - tau) * (order - demand);
    }

    /**
     * Fits the {@code tau}-quantile of demand by subgradient descent on the pinball loss (Adam,
     * standardised features plus a bias), optionally on random Fourier features of an RBF kernel.
     *
     * @param x features {@code [n][d]}
     * @param y demands
     * @param tau the critical ratio
     * @param fourierFeatures 0 for the linear model, else the number of random features
     * @param bandwidth the RBF bandwidth on standardised features
     * @param epochs passes over the data
     * @param seed the seed
     * @return the model
     */
    public static QuantityModel newsvendorErm(double[][] x, double[] y, double tau, int fourierFeatures,
            double bandwidth, int epochs, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int d = x[0].length;
        double[] mean = new double[d];
        double[] sd = new double[d];
        standardise(x, mean, sd);
        double[][] omega = null;
        double[] phase = null;
        if (fourierFeatures > 0) {
            omega = new double[fourierFeatures][d];
            phase = new double[fourierFeatures];
            for (int k = 0; k < fourierFeatures; k++) {
                for (int j = 0; j < d; j++) {
                    omega[k][j] = r.nextGaussian() / bandwidth;
                }
                phase[k] = r.nextDouble() * 2 * Math.PI;
            }
        }
        double[][] om = omega;
        double[] ph = phase;
        java.util.function.Function<double[], double[]> map = v -> {
            double[] z = new double[d];
            for (int j = 0; j < d; j++) {
                z[j] = (v[j] - mean[j]) / sd[j];
            }
            if (om == null) {
                double[] f = Arrays.copyOf(z, d + 1);
                f[d] = 1;
                return f;
            }
            double[] f = new double[om.length + d + 1];
            for (int k = 0; k < om.length; k++) {
                double s = ph[k];
                for (int j = 0; j < d; j++) {
                    s += om[k][j] * z[j];
                }
                f[k] = Math.sqrt(2.0 / om.length) * Math.cos(s);
            }
            System.arraycopy(z, 0, f, om.length, d);
            f[om.length + d] = 1;
            return f;
        };
        int n = x.length;
        double[][] feats = new double[n][];
        for (int i = 0; i < n; i++) {
            feats[i] = map.apply(x[i]);
        }
        double scale = Arrays.stream(y).average().orElse(1);
        int p = feats[0].length;
        double[] w = new double[p];
        w[p - 1] = scale;
        double[] m1 = new double[p];
        double[] m2 = new double[p];
        double lr = 0.05 * scale;
        int batch = 32;
        long step = 0;
        int[] idx = new int[n];
        for (int i = 0; i < n; i++) {
            idx[i] = i;
        }
        for (int e = 0; e < epochs; e++) {
            for (int i = n - 1; i > 0; i--) {
                int q = r.nextInt(i + 1);
                int t = idx[i];
                idx[i] = idx[q];
                idx[q] = t;
            }
            for (int b = 0; b < n; b += batch) {
                double[] g = new double[p];
                int end = Math.min(n, b + batch);
                for (int k = b; k < end; k++) {
                    double[] f = feats[idx[k]];
                    double pred = 0;
                    for (int j = 0; j < p; j++) {
                        pred += w[j] * f[j];
                    }
                    double sub = y[idx[k]] > pred ? -tau : 1 - tau;
                    for (int j = 0; j < p; j++) {
                        g[j] += sub * f[j] / (end - b);
                    }
                }
                step++;
                for (int j = 0; j < p; j++) {
                    m1[j] = 0.9 * m1[j] + 0.1 * g[j];
                    m2[j] = 0.999 * m2[j] + 0.001 * g[j] * g[j];
                    double mh = m1[j] / (1 - Math.pow(0.9, step));
                    double vh = m2[j] / (1 - Math.pow(0.999, step));
                    w[j] -= lr / Math.sqrt(1 + step / 200.0) * mh / (Math.sqrt(vh) + 1e-8);
                }
            }
        }
        return v -> {
            double[] f = map.apply(v);
            double s = 0;
            for (int j = 0; j < p; j++) {
                s += w[j] * f[j];
            }
            return Math.max(0, s);
        };
    }

    /**
     * Predict-then-optimise: least-squares mean plus the normal {@code tau}-quantile of the
     * residual standard deviation.
     *
     * @param x features
     * @param y demands
     * @param tau the critical ratio
     * @return the model
     */
    public static QuantityModel predictThenOptimise(double[][] x, double[] y, double tau) {
        int n = x.length;
        int d = x[0].length;
        double[] mean = new double[d];
        double[] sd = new double[d];
        standardise(x, mean, sd);
        int p = d + 1;
        double[][] a = new double[p][p + 1];
        for (int i = 0; i < n; i++) {
            double[] f = new double[p];
            for (int j = 0; j < d; j++) {
                f[j] = (x[i][j] - mean[j]) / sd[j];
            }
            f[d] = 1;
            for (int r = 0; r < p; r++) {
                for (int c = 0; c < p; c++) {
                    a[r][c] += f[r] * f[c];
                }
                a[r][p] += f[r] * y[i];
            }
        }
        for (int r = 0; r < p; r++) {
            a[r][r] += 1e-6;
        }
        double[] w = gauss(a);
        double rss = 0;
        for (int i = 0; i < n; i++) {
            double pred = w[d];
            for (int j = 0; j < d; j++) {
                pred += w[j] * (x[i][j] - mean[j]) / sd[j];
            }
            rss += (y[i] - pred) * (y[i] - pred);
        }
        double sigma = Math.sqrt(rss / Math.max(1, n - p));
        double z = OperationsSims.normalQuantile(tau);
        return v -> {
            double pred = w[d];
            for (int j = 0; j < d; j++) {
                pred += w[j] * (v[j] - mean[j]) / sd[j];
            }
            return Math.max(0, pred + sigma * z);
        };
    }

    private static void standardise(double[][] x, double[] mean, double[] sd) {
        int n = x.length;
        for (int j = 0; j < mean.length; j++) {
            double m = 0;
            for (double[] row : x) {
                m += row[j];
            }
            m /= n;
            double v = 0;
            for (double[] row : x) {
                v += (row[j] - m) * (row[j] - m);
            }
            mean[j] = m;
            sd[j] = Math.sqrt(v / n) + 1e-9;
        }
    }

    private static double[] gauss(double[][] a) {
        int n = a.length;
        for (int c = 0; c < n; c++) {
            int piv = c;
            for (int r = c + 1; r < n; r++) {
                if (Math.abs(a[r][c]) > Math.abs(a[piv][c])) {
                    piv = r;
                }
            }
            double[] t = a[piv];
            a[piv] = a[c];
            a[c] = t;
            for (int r = 0; r < n; r++) {
                if (r != c && a[c][c] != 0) {
                    double f = a[r][c] / a[c][c];
                    for (int k = c; k <= n; k++) {
                        a[r][k] -= f * a[c][k];
                    }
                }
            }
        }
        double[] w = new double[n];
        for (int i = 0; i < n; i++) {
            w[i] = a[i][i] == 0 ? 0 : a[i][n] / a[i][i];
        }
        return w;
    }

    // ------------------------------------------------------------------ multi-period lost sales

    /** An ordering rule for the lost-sales problem. */
    @FunctionalInterface
    public interface OrderPolicy {
        /**
         * @param onHand stock on hand
         * @param pipeline outstanding orders, the next arrival first
         * @return the quantity to order
         */
        int order(int onHand, int[] pipeline);
    }

    /**
     * @param level the order-up-to level of the inventory position
     * @param maxOrder the largest order
     * @return the base-stock policy
     */
    public static OrderPolicy baseStock(int level, int maxOrder) {
        return (onHand, pipeline) -> {
            int position = onHand + Arrays.stream(pipeline).sum();
            return Math.max(0, Math.min(maxOrder, level - position));
        };
    }

    /**
     * @param reorder order when the position is at most this
     * @param upTo the level to order up to
     * @param maxOrder the largest order
     * @return the (s, S) policy
     */
    public static OrderPolicy sS(int reorder, int upTo, int maxOrder) {
        return (onHand, pipeline) -> {
            int position = onHand + Arrays.stream(pipeline).sum();
            return position <= reorder ? Math.max(0, Math.min(maxOrder, upTo - position)) : 0;
        };
    }

    /**
     * Simulates a policy and returns the average cost per period.
     *
     * @param problem the problem
     * @param policy the policy
     * @param periods periods (after a 100-period warm-up)
     * @param seed the demand seed (common random numbers across policies)
     * @return the mean cost
     */
    public static double averageCost(OperationsSims.Inventory problem, OrderPolicy policy, int periods, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int l = problem.leadTime();
        int[] pipe = new int[l];
        int onHand = 0;
        double cost = 0;
        int warm = 100;
        for (int t = 0; t < periods + warm; t++) {
            int q = policy.order(onHand, pipe.clone());
            onHand += pipe[0];
            System.arraycopy(pipe, 1, pipe, 0, l - 1);
            pipe[l - 1] = q;
            int d = problem.sample(r);
            int lost = Math.max(0, d - onHand);
            onHand = Math.max(0, onHand - d);
            onHand = Math.min(onHand, problem.maxStock());
            if (t >= warm) {
                cost += problem.holding() * onHand + problem.penalty() * lost;
            }
        }
        return cost / periods;
    }

    /**
     * The best base-stock level by search.
     *
     * @param problem the problem
     * @param periods simulated periods per level
     * @param seed the demand seed
     * @return {level, its cost}
     */
    public static double[] bestBaseStock(OperationsSims.Inventory problem, int periods, long seed) {
        double best = Double.POSITIVE_INFINITY;
        int arg = 0;
        int max = problem.maxStock() + problem.leadTime() * problem.maxOrder();
        for (int s = 0; s <= max; s++) {
            double c = averageCost(problem, baseStock(s, problem.maxOrder()), periods, seed);
            if (c < best) {
                best = c;
                arg = s;
            }
        }
        return new double[] {arg, best};
    }

    /**
     * The best (s, S) policy by search.
     *
     * @param problem the problem
     * @param periods simulated periods per pair
     * @param seed the demand seed
     * @return {s, S, cost}
     */
    public static double[] bestSS(OperationsSims.Inventory problem, int periods, long seed) {
        double best = Double.POSITIVE_INFINITY;
        int bs = 0;
        int bS = 0;
        int max = problem.maxStock() + problem.leadTime() * problem.maxOrder();
        for (int upTo = 1; upTo <= max; upTo++) {
            for (int s = Math.max(0, upTo - problem.maxOrder()); s < upTo; s++) {
                double c = averageCost(problem, sS(s, upTo, problem.maxOrder()), periods, seed);
                if (c < best) {
                    best = c;
                    bs = s;
                    bS = upTo;
                }
            }
        }
        return new double[] {bs, bS, best};
    }

    /**
     * The optimal policy by discounted value iteration over (on-hand, pipeline).
     *
     * @param problem the problem (lead time up to 3)
     * @param gamma the discount (close to 1 approximates the average cost)
     * @param sweeps the most sweeps
     * @return the policy
     */
    public static OrderPolicy optimal(OperationsSims.Inventory problem, double gamma, int sweeps) {
        int l = problem.leadTime();
        int stock = problem.maxStock() + 1;
        int ord = problem.maxOrder() + 1;
        int states = stock;
        for (int k = 0; k < l; k++) {
            states *= ord;
        }
        double[] pmf = problem.demandPmf();
        double[] v = new double[states];
        int[] policy = new int[states];
        int[] pipe = new int[l];
        for (int sweep = 0; sweep < sweeps; sweep++) {
            double[] nv = new double[states];
            double change = 0;
            for (int s = 0; s < states; s++) {
                int onHand = decode(s, stock, ord, pipe);
                int avail = onHand + pipe[0];
                double best = Double.POSITIVE_INFINITY;
                int arg = 0;
                for (int q = 0; q < ord; q++) {
                    double e = 0;
                    for (int d = 0; d < pmf.length; d++) {
                        if (pmf[d] == 0) {
                            continue;
                        }
                        int left = Math.min(problem.maxStock(), Math.max(0, avail - d));
                        int lost = Math.max(0, d - avail);
                        int next = left;
                        for (int k = 1; k < l; k++) {
                            next = next * ord + pipe[k];
                        }
                        next = next * ord + q;
                        e += pmf[d] * (problem.holding() * left + problem.penalty() * lost + gamma * v[next]);
                    }
                    if (e < best) {
                        best = e;
                        arg = q;
                    }
                }
                nv[s] = best;
                policy[s] = arg;
                change = Math.max(change, Math.abs(best - v[s]));
            }
            v = nv;
            if (change < 1e-6) {
                break;
            }
        }
        return (onHand, p) -> {
            int s = Math.min(onHand, problem.maxStock());
            for (int k = 0; k < l; k++) {
                s = s * ord + Math.min(p[k], ord - 1);
            }
            return policy[s];
        };
    }

    private static int decode(int s, int stock, int ord, int[] pipe) {
        for (int k = pipe.length - 1; k >= 0; k--) {
            pipe[k] = s % ord;
            s /= ord;
        }
        return s;
    }

    /**
     * The lost-sales problem as a core environment: observations are the on-hand stock and the
     * pipeline divided by the stock cap, the action is the order quantity, the reward is minus
     * the period's cost divided by the lost-sales penalty, and episodes last a fixed number of
     * periods (truncated).
     */
    public static final class Env implements Environment {
        private final OperationsSims.Inventory problem;
        private final int length;
        private SplittableRandom random = new SplittableRandom(0);
        private int[] pipe;
        private int onHand;
        private int t;

        /**
         * @param problem the problem
         * @param length periods per episode
         */
        public Env(OperationsSims.Inventory problem, int length) {
            this.problem = problem;
            this.length = length;
            this.pipe = new int[problem.leadTime()];
        }

        @Override
        public Space observationSpace() {
            return Space.box(problem.leadTime() + 1, 0, 2);
        }

        @Override
        public Space actionSpace() {
            return Space.discrete(problem.maxOrder() + 1);
        }

        private float[] obs() {
            float[] o = new float[pipe.length + 1];
            o[0] = onHand / (float) problem.maxStock();
            for (int k = 0; k < pipe.length; k++) {
                o[k + 1] = pipe[k] / (float) problem.maxStock();
            }
            return o;
        }

        @Override
        public float[] reset(SplittableRandom random) {
            this.random = random;
            pipe = new int[problem.leadTime()];
            onHand = (int) Math.round(problem.meanDemand() * 2);
            t = 0;
            return obs();
        }

        @Override
        public Step step(float[] action) {
            int q = Math.max(0, Math.min(problem.maxOrder(), Math.round(action[0])));
            onHand += pipe[0];
            System.arraycopy(pipe, 1, pipe, 0, pipe.length - 1);
            pipe[pipe.length - 1] = q;
            int d = problem.sample(random);
            int lost = Math.max(0, d - onHand);
            onHand = Math.min(problem.maxStock(), Math.max(0, onHand - d));
            double cost = problem.holding() * onHand + problem.penalty() * lost;
            t++;
            return new Step(obs(), (float) (-cost / problem.penalty()), false, t >= length);
        }
    }
}
