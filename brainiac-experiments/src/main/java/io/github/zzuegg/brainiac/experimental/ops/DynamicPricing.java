package io.github.zzuegg.brainiac.experimental.ops;

import io.github.zzuegg.brainiac.experimental.bandit.BanditPolicy;
import io.github.zzuegg.brainiac.experimental.data.OperationsSims;
import io.github.zzuegg.brainiac.experimental.ml.LinearProgram;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * Dynamic pricing with demand learning: one customer per period buys at the posted price with an
 * unknown probability. The structured policies share a logistic demand model
 * {@code P(buy | p) = sigmoid(a - b p)} whose posterior is a Laplace approximation fitted by
 * Newton's method on the per-price sale counts:
 *
 * <ul>
 *   <li>{@link Method#CERTAINTY_EQUIVALENT}: price at the maximum-likelihood optimum, with forced
 *       exploration of a random price with probability {@code min(1, K / sqrt(t))} (den Boer,
 *       Surveys in OR&amp;MS 2015);</li>
 *   <li>{@link Method#THOMPSON_PARAMETRIC}: Thompson sampling of {@code (a, b)} and the
 *       revenue-maximising price for the sample;</li>
 *   <li>{@link Method#THOMPSON_INVENTORY_LP}: Thompson sampling with an inventory budget (Ferreira,
 *       Simchi-Levi &amp; Wang, Operations Research 2018): each period solve the LP
 *       {@code max sum_k p_k d_k x_k} subject to {@code sum_k d_k x_k <= remaining stock / remaining periods}
 *       and {@code sum_k x_k <= 1} for the sampled demands (wave 6 {@link LinearProgram}) and post price
 *       {@code k} with probability {@code x_k} (otherwise shut off);</li>
 *   <li>{@link #contextual}: contextual pricing from binary sale feedback (Javanmard &amp; Nazerzadeh,
 *       JMLR 2019): a logistic model of {@code P(buy | x, p) = sigmoid(w . x - beta p)}, i.e. a
 *       linear valuation with logistic noise, refitted by maximum likelihood, with decaying
 *       exploration; its baseline is disjoint LinUCB over a price grid.</li>
 * </ul>
 */
public final class DynamicPricing {

    private DynamicPricing() {
    }

    /** The pricing policy. */
    public enum Method {
        /** maximum likelihood plus forced exploration */
        CERTAINTY_EQUIVALENT,
        /** Thompson sampling of the demand parameters */
        THOMPSON_PARAMETRIC,
        /** Thompson sampling with the inventory LP */
        THOMPSON_INVENTORY_LP
    }

    /**
     * The outcome of a run.
     *
     * @param revenue realised revenue
     * @param regret the clairvoyant expected revenue minus {@code revenue}
     * @param stockOutPeriod the period the inventory ran out ({@code -1} if never)
     */
    public record Run(double revenue, double regret, int stockOutPeriod) {
    }

    /**
     * The clairvoyant benchmark: {@code T max_k p_k d(p_k)} without inventory, the fluid LP value
     * {@code T LP(inventory / T)} with it.
     *
     * @param problem the problem
     * @param horizon periods
     * @param inventory the stock ({@code < 0} for unlimited)
     * @return the benchmark revenue
     */
    public static double clairvoyant(OperationsSims.Pricing problem, int horizon, double inventory) {
        if (inventory < 0) {
            return horizon * problem.bestRevenue();
        }
        double[] p = problem.prices();
        double[] d = new double[p.length];
        for (int k = 0; k < p.length; k++) {
            d[k] = problem.demand(p[k]);
        }
        return horizon * lpRevenue(p, d, inventory / horizon)[p.length];
    }

    /** Solves the pricing LP; returns {x_0..x_{K-1}, value}. */
    private static double[] lpRevenue(double[] p, double[] d, double rate) {
        int k = p.length;
        double[] c = new double[k];
        double[][] a = new double[2][k];
        for (int i = 0; i < k; i++) {
            c[i] = p[i] * d[i];
            a[0][i] = d[i];
            a[1][i] = 1;
        }
        LinearProgram.Solution s = LinearProgram.maximize(c, a, new double[] {Math.max(0, rate), 1});
        double[] out = Arrays.copyOf(s.x(), k + 1);
        out[k] = s.objective();
        return out;
    }

    /**
     * Runs a structured pricing policy.
     *
     * @param method the policy
     * @param problem the problem
     * @param horizon periods
     * @param inventory the stock ({@code < 0} for unlimited)
     * @param seed the seed
     * @return the run
     */
    public static Run run(Method method, OperationsSims.Pricing problem, int horizon, double inventory, long seed) {
        Objects.requireNonNull(method, "method");
        SplittableRandom r = new SplittableRandom(seed);
        double[] prices = problem.prices();
        int k = prices.length;
        double pMax = Arrays.stream(prices).max().orElse(1);
        double[] n = new double[k];
        double[] sales = new double[k];
        double[] theta = {0, 0};
        double[][] cov = {{10, 0}, {0, 10}};
        double stock = inventory;
        double revenue = 0;
        int stockOut = -1;
        for (int t = 1; t <= horizon; t++) {
            if (inventory >= 0 && stock < 1) {
                if (stockOut < 0) {
                    stockOut = t;
                }
                break;
            }
            if (t % 10 == 1 || t < 50) {
                double[][] fit = laplace(prices, pMax, n, sales, theta);
                theta = fit[0];
                cov = new double[][] {fit[1], fit[2]};
            }
            int choice;
            switch (method) {
                case CERTAINTY_EQUIVALENT -> {
                    if (r.nextDouble() < Math.min(1, k / Math.sqrt(t))) {
                        choice = r.nextInt(k);
                    } else {
                        choice = bestPrice(prices, pMax, theta[0], theta[1]);
                    }
                }
                case THOMPSON_PARAMETRIC -> {
                    double[] s = sampleGaussian(theta, cov, r);
                    choice = bestPrice(prices, pMax, s[0], s[1]);
                }
                default -> {
                    double[] s = sampleGaussian(theta, cov, r);
                    double[] d = new double[k];
                    for (int i = 0; i < k; i++) {
                        d[i] = sigmoid(s[0] - s[1] * prices[i] / pMax);
                    }
                    double rate = inventory < 0 ? 1 : stock / (horizon - t + 1);
                    double[] x = lpRevenue(prices, d, rate);
                    double u = r.nextDouble();
                    choice = -1;
                    for (int i = 0; i < k; i++) {
                        u -= x[i];
                        if (u <= 0) {
                            choice = i;
                            break;
                        }
                    }
                }
            }
            if (choice < 0) {
                continue;
            }
            boolean sold = r.nextDouble() < problem.demand(prices[choice]);
            n[choice]++;
            if (sold) {
                sales[choice]++;
                revenue += prices[choice];
                stock--;
            }
        }
        return new Run(revenue, clairvoyant(problem, horizon, inventory) - revenue, stockOut);
    }

    /**
     * Runs an unstructured bandit over the price grid (reward: revenue divided by the highest price).
     *
     * @param policy the bandit policy over the prices
     * @param problem the problem
     * @param horizon periods
     * @param inventory the stock ({@code < 0} for unlimited)
     * @param seed the seed
     * @return the run
     */
    public static Run runBandit(BanditPolicy policy, OperationsSims.Pricing problem, int horizon, double inventory,
            long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[] prices = problem.prices();
        double pMax = Arrays.stream(prices).max().orElse(1);
        double stock = inventory;
        double revenue = 0;
        int stockOut = -1;
        for (int t = 1; t <= horizon; t++) {
            if (inventory >= 0 && stock < 1) {
                stockOut = t;
                break;
            }
            int k = policy.select();
            boolean sold = r.nextDouble() < problem.demand(prices[k]);
            if (sold) {
                revenue += prices[k];
                stock--;
            }
            policy.update(k, sold ? prices[k] / pMax : 0);
        }
        return new Run(revenue, clairvoyant(problem, horizon, inventory) - revenue, stockOut);
    }

    private static int bestPrice(double[] prices, double pMax, double a, double b) {
        int best = 0;
        double bv = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < prices.length; i++) {
            double v = prices[i] * sigmoid(a - b * prices[i] / pMax);
            if (v > bv) {
                bv = v;
                best = i;
            }
        }
        return best;
    }

    private static double sigmoid(double z) {
        return 1 / (1 + Math.exp(-z));
    }

    /**
     * Newton's method for the MAP of the logistic demand model (prices scaled by {@code pMax},
     * prior N(0, 10 I)); returns {mode, covariance row 0, covariance row 1}.
     */
    private static double[][] laplace(double[] prices, double pMax, double[] n, double[] sales, double[] start) {
        double a = start[0];
        double b = start[1];
        double[][] h = new double[2][2];
        for (int it = 0; it < 20; it++) {
            double ga = -a / 10;
            double gb = -b / 10;
            h = new double[][] {{1.0 / 10, 0}, {0, 1.0 / 10}};
            for (int i = 0; i < prices.length; i++) {
                if (n[i] == 0) {
                    continue;
                }
                double x = -prices[i] / pMax;
                double p = sigmoid(a + b * x);
                double res = sales[i] - n[i] * p;
                ga += res;
                gb += res * x;
                double w = n[i] * p * (1 - p);
                h[0][0] += w;
                h[0][1] += w * x;
                h[1][0] += w * x;
                h[1][1] += w * x * x;
            }
            double det = h[0][0] * h[1][1] - h[0][1] * h[1][0];
            double da = (h[1][1] * ga - h[0][1] * gb) / det;
            double db = (-h[1][0] * ga + h[0][0] * gb) / det;
            a += da;
            b += db;
            if (Math.abs(da) + Math.abs(db) < 1e-8) {
                break;
            }
        }
        double det = h[0][0] * h[1][1] - h[0][1] * h[1][0];
        return new double[][] {{a, b}, {h[1][1] / det, -h[0][1] / det}, {-h[1][0] / det, h[0][0] / det}};
    }

    private static double[] sampleGaussian(double[] mean, double[][] cov, SplittableRandom r) {
        double l11 = Math.sqrt(Math.max(1e-12, cov[0][0]));
        double l21 = cov[1][0] / l11;
        double l22 = Math.sqrt(Math.max(1e-12, cov[1][1] - l21 * l21));
        double z1 = r.nextGaussian();
        double z2 = r.nextGaussian();
        return new double[] {mean[0] + l11 * z1, mean[1] + l21 * z1 + l22 * z2};
    }

    // ------------------------------------------------------------------ contextual

    /** Contextual pricing policies. */
    public enum Contextual {
        /** logistic valuation model refitted by maximum likelihood */
        LINEAR_VALUATION,
        /** disjoint LinUCB over a price grid */
        LINUCB_GRID
    }

    /**
     * Runs a contextual pricing policy.
     *
     * @param policy the policy
     * @param valuations the customers
     * @param horizon periods
     * @param gridSize prices in the LinUCB grid (and the candidate prices of the valuation model)
     * @param seed the seed
     * @return the run (regret against the per-context clairvoyant price)
     */
    public static Run contextual(Contextual policy, OperationsSims.Valuations valuations, int horizon, int gridSize,
            long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int d = valuations.theta().length;
        double[] grid = new double[gridSize];
        for (int i = 0; i < gridSize; i++) {
            grid[i] = 0.5 + 11.5 * i / (gridSize - 1);
        }
        double[] fine = new double[200];
        for (int i = 0; i < fine.length; i++) {
            fine[i] = 0.06 * (i + 1);
        }
        double revenue = 0;
        double best = 0;
        List<double[]> xs = new ArrayList<>();
        List<Boolean> ys = new ArrayList<>();
        double[] w = new double[d + 1];
        double[][][] aInv = new double[gridSize][][];
        double[][] bVec = new double[gridSize][d];
        for (int k = 0; k < gridSize; k++) {
            aInv[k] = identity(d);
        }
        for (int t = 1; t <= horizon; t++) {
            double[] x = valuations.context(r);
            double pStar = valuations.bestPrice(x);
            best += pStar * valuations.buy(x, pStar);
            double price;
            int arm = -1;
            if (policy == Contextual.LINEAR_VALUATION) {
                if (t % 50 == 0 || t == 20) {
                    w = logisticFit(xs, ys, w);
                }
                if (t <= 20 || r.nextDouble() < Math.min(1, 2 / Math.sqrt(t))) {
                    price = 0.5 + 11.5 * r.nextDouble();
                } else {
                    price = fine[0];
                    double bv = Double.NEGATIVE_INFINITY;
                    for (double p : fine) {
                        double z = -w[d] * p;
                        for (int j = 0; j < d; j++) {
                            z += w[j] * x[j];
                        }
                        double v = p * sigmoid(z);
                        if (v > bv) {
                            bv = v;
                            price = p;
                        }
                    }
                }
            } else {
                double bv = Double.NEGATIVE_INFINITY;
                for (int k = 0; k < gridSize; k++) {
                    double[] th = matVec(aInv[k], bVec[k]);
                    double ucb = dot(th, x) + 1.0 * Math.sqrt(quad(aInv[k], x)) * grid[gridSize - 1] / 4;
                    if (ucb > bv) {
                        bv = ucb;
                        arm = k;
                    }
                }
                price = grid[arm];
            }
            boolean sold = r.nextDouble() < valuations.buy(x, price);
            if (sold) {
                revenue += price;
            }
            if (policy == Contextual.LINEAR_VALUATION) {
                double[] row = Arrays.copyOf(x, d + 1);
                row[d] = price;
                xs.add(row);
                ys.add(sold);
            } else {
                shermanMorrison(aInv[arm], x);
                for (int j = 0; j < d; j++) {
                    bVec[arm][j] += (sold ? price : 0) * x[j];
                }
            }
        }
        return new Run(revenue, best - revenue, -1);
    }

    /** The ridge of the logistic fit, which keeps early (separable) fits finite. */
    private static final double RIDGE = 1.0;

    /** MLE of {@code P(buy) = sigmoid(sum_j w_j x_j - w_d p)} by damped Newton steps (ridge {@link #RIDGE}). */
    private static double[] logisticFit(List<double[]> xs, List<Boolean> ys, double[] start) {
        int p = start.length;
        double[] w = start.clone();
        for (int it = 0; it < 15; it++) {
            double[] g = new double[p];
            double[][] h = new double[p][p + 1];
            for (int j = 0; j < p; j++) {
                g[j] = -RIDGE * w[j];
                h[j][j] = RIDGE;
            }
            for (int i = 0; i < xs.size(); i++) {
                double[] row = xs.get(i);
                double[] f = row.clone();
                f[p - 1] = -row[p - 1];
                double z = dot(w, f);
                double q = sigmoid(z);
                double res = (ys.get(i) ? 1 : 0) - q;
                double wt = q * (1 - q);
                for (int a = 0; a < p; a++) {
                    g[a] += res * f[a];
                    for (int b = 0; b < p; b++) {
                        h[a][b] += wt * f[a] * f[b];
                    }
                }
            }
            for (int a = 0; a < p; a++) {
                h[a][p] = g[a];
            }
            double[] step = solve(h);
            double norm = 0;
            for (int a = 0; a < p; a++) {
                norm += Math.abs(step[a]);
            }
            if (!Double.isFinite(norm)) {
                break;
            }
            double damp = norm > 5 ? 5 / norm : 1;
            for (int a = 0; a < p; a++) {
                w[a] += damp * step[a];
            }
            if (norm < 1e-8) {
                break;
            }
        }
        return w;
    }

    private static double[] solve(double[][] a) {
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
                if (r != c) {
                    double f = a[r][c] / a[c][c];
                    for (int k = c; k <= n; k++) {
                        a[r][k] -= f * a[c][k];
                    }
                }
            }
        }
        double[] x = new double[n];
        for (int i = 0; i < n; i++) {
            x[i] = a[i][n] / a[i][i];
        }
        return x;
    }

    private static double[][] identity(int d) {
        double[][] m = new double[d][d];
        for (int i = 0; i < d; i++) {
            m[i][i] = 1;
        }
        return m;
    }

    private static double[] matVec(double[][] m, double[] v) {
        double[] out = new double[m.length];
        for (int i = 0; i < m.length; i++) {
            out[i] = dot(m[i], v);
        }
        return out;
    }

    private static double quad(double[][] m, double[] v) {
        return dot(v, matVec(m, v));
    }

    private static void shermanMorrison(double[][] aInv, double[] x) {
        double[] ax = matVec(aInv, x);
        double den = 1 + dot(x, ax);
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x.length; j++) {
                aInv[i][j] -= ax[i] * ax[j] / den;
            }
        }
    }

    private static double dot(double[] a, double[] b) {
        double s = 0;
        for (int i = 0; i < Math.min(a.length, b.length); i++) {
            s += a[i] * b[i];
        }
        return s;
    }
}
