package io.github.zzuegg.brainiac.experimental.markets;

import java.util.Arrays;
import java.util.SplittableRandom;

/**
 * Online portfolio selection on price relatives {@code x_t[i] = p_t[i] / p_(t-1)[i]}.
 *
 * <ul>
 *   <li>{@link Method#UNIFORM_BUY_AND_HOLD} and {@link Method#UCRP}: the uniform portfolio held
 *       or rebalanced every period.</li>
 *   <li>{@link Method#UNIVERSAL}: Cover's universal portfolio, the wealth-weighted mean of
 *       constant-rebalanced portfolios, by Monte Carlo over Dirichlet(1) draws (Cover 1991).</li>
 *   <li>{@link Method#EG}: exponentiated gradient {@code b_i ~ b_i exp(eta x_i / b.x)}
 *       (Helmbold et al. 1998).</li>
 *   <li>{@link Method#ONS}: online Newton step on {@code -ln b.x} with the generalised projection
 *       onto the simplex in the {@code A_t} norm (Agarwal et al., ICML 2006).</li>
 *   <li>{@link Method#OLMAR}: on-line moving-average reversion, predicting
 *       {@code x_(t+1) = MA_w(p) / p_t} and moving to the closest portfolio whose predicted
 *       return reaches {@code epsilon} (Li and Hoi, ICML 2012).</li>
 * </ul>
 *
 * <p>Proportional transaction costs are charged on the turnover between the drifted and the new
 * portfolio. {@link #bestConstantRebalanced} computes the hindsight BCRP by exponentiated
 * gradient to convergence.
 */
public final class OnlinePortfolios {

    /** A strategy. */
    public enum Method {
        /** buy the uniform portfolio and hold. */
        UNIFORM_BUY_AND_HOLD,
        /** uniform constant rebalanced portfolio. */
        UCRP,
        /** Cover's universal portfolio by Monte Carlo. */
        UNIVERSAL,
        /** exponentiated gradient. */
        EG,
        /** online Newton step. */
        ONS,
        /** on-line moving average reversion. */
        OLMAR
    }

    private OnlinePortfolios() {
    }

    /**
     * Runs a strategy.
     *
     * @param method the strategy
     * @param x price relatives {@code [period][asset]}
     * @param cost the proportional transaction cost
     * @param seed the seed (universal portfolio draws)
     * @return the final log wealth
     */
    public static double logWealth(Method method, double[][] x, double cost, long seed) {
        int m = x[0].length;
        double[] b = uniform(m);
        double[] held = new double[m];
        double logW = 0;
        // state
        SplittableRandom rnd = new SplittableRandom(seed);
        int draws = 400;
        double[][] crps = new double[draws][];
        double[] crpLogW = new double[draws];
        if (method == Method.UNIVERSAL) {
            for (int k = 0; k < draws; k++) {
                crps[k] = dirichlet(m, rnd);
            }
        }
        double[][] a = new double[m][m];
        double[] grad = new double[m];
        for (int i = 0; i < m; i++) {
            a[i][i] = 1;
        }
        double[] price = new double[m];
        Arrays.fill(price, 1);
        double[][] history = new double[x.length][];
        for (int t = 0; t < x.length; t++) {
            double[] target = switch (method) {
                case UNIFORM_BUY_AND_HOLD -> t == 0 ? uniform(m) : held;
                default -> b;
            };
            double turnover = t == 0 ? 0 : l1(target, held);
            double ret = dot(target, x[t]);
            logW += Math.log(ret) + Math.log(Math.max(1 - cost * turnover, 1e-12));
            for (int i = 0; i < m; i++) {
                held[i] = target[i] * x[t][i] / ret;
                price[i] *= x[t][i];
            }
            history[t] = price.clone();
            b = switch (method) {
                case UNIFORM_BUY_AND_HOLD, UCRP -> uniform(m);
                case UNIVERSAL -> {
                    double max = Double.NEGATIVE_INFINITY;
                    for (int k = 0; k < draws; k++) {
                        crpLogW[k] += Math.log(dot(crps[k], x[t]));
                        max = Math.max(max, crpLogW[k]);
                    }
                    double[] next = new double[m];
                    double z = 0;
                    for (int k = 0; k < draws; k++) {
                        double w = Math.exp(crpLogW[k] - max);
                        z += w;
                        for (int i = 0; i < m; i++) {
                            next[i] += w * crps[k][i];
                        }
                    }
                    for (int i = 0; i < m; i++) {
                        next[i] /= z;
                    }
                    yield next;
                }
                case EG -> {
                    double[] next = new double[m];
                    double z = 0;
                    for (int i = 0; i < m; i++) {
                        next[i] = target[i] * Math.exp(0.05 * x[t][i] / ret);
                        z += next[i];
                    }
                    for (int i = 0; i < m; i++) {
                        next[i] /= z;
                    }
                    yield next;
                }
                case ONS -> ons(target, x[t], ret, a, grad);
                case OLMAR -> olmar(target, history, t, 5, 10);
            };
        }
        return logW;
    }

    private static double[] ons(double[] b, double[] x, double ret, double[][] a, double[] grad) {
        int m = b.length;
        double beta = 1;
        double delta = 0.125;
        double[] g = new double[m];
        for (int i = 0; i < m; i++) {
            g[i] = -x[i] / ret;
            grad[i] += g[i];
        }
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                a[i][j] += g[i] * g[j];
            }
        }
        // b = Proj_A(-(1 / beta) A^-1 sum grad) in Agarwal et al.'s form
        double[] p = solve(a, grad);
        double[] y = new double[m];
        for (int i = 0; i < m; i++) {
            y[i] = -p[i] / beta;
        }
        double[] proj = projectInNorm(y, a);
        double[] out = new double[m];
        for (int i = 0; i < m; i++) {
            out[i] = (1 - delta) * proj[i] + delta / m;
        }
        return out;
    }

    private static double[] solve(double[][] a, double[] b) {
        int n = b.length;
        double[][] m = new double[n][n + 1];
        for (int i = 0; i < n; i++) {
            System.arraycopy(a[i], 0, m[i], 0, n);
            m[i][n] = b[i];
        }
        for (int c = 0; c < n; c++) {
            int piv = c;
            for (int r = c + 1; r < n; r++) {
                if (Math.abs(m[r][c]) > Math.abs(m[piv][c])) {
                    piv = r;
                }
            }
            double[] tmp = m[c];
            m[c] = m[piv];
            m[piv] = tmp;
            for (int r = 0; r < n; r++) {
                if (r != c) {
                    double f = m[r][c] / m[c][c];
                    for (int k = c; k <= n; k++) {
                        m[r][k] -= f * m[c][k];
                    }
                }
            }
        }
        double[] x = new double[n];
        for (int i = 0; i < n; i++) {
            x[i] = m[i][n] / m[i][i];
        }
        return x;
    }

    /** Projection onto the simplex in the norm {@code (z - y)^T A (z - y)}, by projected gradient. */
    private static double[] projectInNorm(double[] y, double[][] a) {
        int m = y.length;
        double lmax = 0;
        for (double[] row : a) {
            double s = 0;
            for (double v : row) {
                s += Math.abs(v);
            }
            lmax = Math.max(lmax, s);
        }
        double[] z = projectSimplex(y);
        for (int it = 0; it < 100; it++) {
            double[] next = new double[m];
            for (int i = 0; i < m; i++) {
                double g = 0;
                for (int j = 0; j < m; j++) {
                    g += a[i][j] * (z[j] - y[j]);
                }
                next[i] = z[i] - g / lmax;
            }
            z = projectSimplex(next);
        }
        return z;
    }

    /**
     * Euclidean projection onto the probability simplex.
     *
     * @param v a vector
     * @return the closest point of the simplex
     */
    public static double[] projectSimplex(double[] v) {
        int m = v.length;
        double[] u = v.clone();
        Arrays.sort(u);
        double cum = 0;
        double theta = 0;
        // the largest rho with u_rho - (sum of the top rho - 1) / rho > 0 sets the shift
        for (int r = 1; r <= m; r++) {
            cum += u[m - r];
            double t = (cum - 1) / r;
            if (u[m - r] - t > 0) {
                theta = t;
            }
        }
        double[] out = new double[m];
        for (int i = 0; i < m; i++) {
            out[i] = Math.max(v[i] - theta, 0);
        }
        return out;
    }

    private static double[] olmar(double[] b, double[][] history, int t, int window, double epsilon) {
        int m = b.length;
        double[] pred = new double[m];
        for (int i = 0; i < m; i++) {
            double ma = 0;
            int count = 0;
            for (int k = Math.max(0, t - window + 1); k <= t; k++) {
                ma += history[k][i];
                count++;
            }
            pred[i] = ma / count / history[t][i];
        }
        double mean = Arrays.stream(pred).average().orElse(1);
        double dev = 0;
        for (double p : pred) {
            dev += (p - mean) * (p - mean);
        }
        double lambda = dev == 0 ? 0 : Math.max(0, (epsilon - dot(b, pred)) / dev);
        double[] next = new double[m];
        for (int i = 0; i < m; i++) {
            next[i] = b[i] + lambda * (pred[i] - mean);
        }
        return projectSimplex(next);
    }

    /**
     * The best constant-rebalanced portfolio in hindsight, by exponentiated gradient on the mean
     * log return to convergence.
     *
     * @param x price relatives
     * @param iterations iterations
     * @return {@code {log wealth, weights...}}
     */
    public static double[] bestConstantRebalanced(double[][] x, int iterations) {
        int m = x[0].length;
        double[] b = uniform(m);
        for (int it = 0; it < iterations; it++) {
            double[] g = new double[m];
            for (double[] xt : x) {
                double r = dot(b, xt);
                for (int i = 0; i < m; i++) {
                    g[i] += xt[i] / r / x.length;
                }
            }
            double z = 0;
            for (int i = 0; i < m; i++) {
                b[i] *= Math.exp(2 * g[i]);
                z += b[i];
            }
            for (int i = 0; i < m; i++) {
                b[i] /= z;
            }
        }
        double logW = 0;
        for (double[] xt : x) {
            logW += Math.log(dot(b, xt));
        }
        double[] out = new double[m + 1];
        out[0] = logW;
        System.arraycopy(b, 0, out, 1, m);
        return out;
    }

    private static double[] uniform(int m) {
        double[] b = new double[m];
        Arrays.fill(b, 1.0 / m);
        return b;
    }

    private static double[] dirichlet(int m, SplittableRandom rnd) {
        double[] b = new double[m];
        double z = 0;
        for (int i = 0; i < m; i++) {
            b[i] = -Math.log(1 - rnd.nextDouble());
            z += b[i];
        }
        for (int i = 0; i < m; i++) {
            b[i] /= z;
        }
        return b;
    }

    private static double dot(double[] a, double[] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) {
            s += a[i] * b[i];
        }
        return s;
    }

    private static double l1(double[] a, double[] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) {
            s += Math.abs(a[i] - b[i]);
        }
        return s;
    }
}
