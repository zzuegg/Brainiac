package io.github.zzuegg.brainiac.experimental.data;

import java.util.Arrays;
import java.util.SplittableRandom;

/**
 * Synthetic benchmarks for causal discovery, each with its true graph for scoring:
 * <ul>
 *   <li>{@link #randomDag}: Erdős-Rényi DAGs over a random order with a target expected degree,
 *       edge weights {@code +-U(0.5, 2)};</li>
 *   <li>{@link #linearSem}: linear SEMs with Gaussian, uniform or Laplace noise, equal or random
 *       noise scales;</li>
 *   <li>{@link #nonlinearAnm}: nonlinear additive-noise SEMs with random Fourier-feature mechanisms
 *       (draws from a Gaussian-process-like prior) and Gaussian noise;</li>
 *   <li>{@link #confounded}: linear-Gaussian data with hidden common causes (for FCI), and
 *       {@link #selected}, with selection on a hidden common effect;</li>
 *   <li>{@link #pair}: bivariate cause-effect pairs (additive noise, multiplicative noise,
 *       near-deterministic mechanisms, linear-Gaussian controls) in a random orientation;</li>
 *   <li>{@link #laggedVar}, {@link #nonlinearLagged} and {@link #lorenz96}: multivariate time
 *       series with known lagged cross-links.</li>
 * </ul>
 */
public final class DiscoveryBenchmarks {

    private DiscoveryBenchmarks() {
    }

    /** Noise distributions of {@link #linearSem}, all with unit variance before scaling. */
    public enum Noise {
        /** Gaussian. */
        GAUSSIAN,
        /** Uniform. */
        UNIFORM,
        /** Laplace. */
        LAPLACE
    }

    private static double noise(Noise kind, SplittableRandom r) {
        return switch (kind) {
            case GAUSSIAN -> r.nextGaussian();
            case UNIFORM -> Math.sqrt(3) * (2 * r.nextDouble() - 1);
            case LAPLACE -> {
                double u = r.nextDouble() - 0.5;
                yield -Math.signum(u) * Math.log(1 - 2 * Math.abs(u)) / Math.sqrt(2);
            }
        };
    }

    /**
     * A random weighted DAG.
     *
     * @param d nodes
     * @param expectedDegree the expected number of edges per node (in plus out)
     * @param seed the seed
     * @return {@code w[i][j] != 0} meaning {@code i -> j}
     */
    public static double[][] randomDag(int d, double expectedDegree, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int[] perm = new int[d];
        Arrays.setAll(perm, i -> i);
        for (int i = d - 1; i > 0; i--) {
            int j = r.nextInt(i + 1);
            int t = perm[i];
            perm[i] = perm[j];
            perm[j] = t;
        }
        double p = Math.min(1, expectedDegree / Math.max(1, d - 1));
        double[][] w = new double[d][d];
        for (int a = 0; a < d; a++) {
            for (int b = a + 1; b < d; b++) {
                if (r.nextDouble() < p) {
                    w[perm[a]][perm[b]] = (r.nextBoolean() ? 1 : -1) * (0.5 + 1.5 * r.nextDouble());
                }
            }
        }
        return w;
    }

    /**
     * A topological order of a DAG.
     *
     * @param w the DAG
     * @return the order
     */
    public static int[] topologicalOrder(double[][] w) {
        int d = w.length;
        int[] indeg = new int[d];
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) {
                indeg[j] += w[i][j] != 0 ? 1 : 0;
            }
        }
        int[] order = new int[d];
        boolean[] done = new boolean[d];
        for (int k = 0; k < d; k++) {
            int pick = -1;
            for (int j = 0; j < d && pick < 0; j++) {
                if (!done[j] && indeg[j] == 0) {
                    pick = j;
                }
            }
            done[pick] = true;
            order[k] = pick;
            for (int j = 0; j < d; j++) {
                indeg[j] -= w[pick][j] != 0 ? 1 : 0;
            }
        }
        return order;
    }

    /**
     * Samples a linear SEM {@code x_j = sum_i w_ij x_i + s_j e_j}.
     *
     * @param w the weighted DAG
     * @param n samples
     * @param kind the noise distribution
     * @param equalVariance unit noise scales when true, else {@code s_j ~ U(0.5, 1.5)}
     * @param seed the seed
     * @return {@code [n][d]}
     */
    public static double[][] linearSem(double[][] w, int n, Noise kind, boolean equalVariance, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int d = w.length;
        int[] order = topologicalOrder(w);
        double[] s = new double[d];
        for (int j = 0; j < d; j++) {
            s[j] = equalVariance ? 1 : 0.5 + r.nextDouble();
        }
        double[][] x = new double[n][d];
        for (int t = 0; t < n; t++) {
            for (int j : order) {
                double v = s[j] * noise(kind, r);
                for (int i = 0; i < d; i++) {
                    v += w[i][j] * x[t][i];
                }
                x[t][j] = v;
            }
        }
        return x;
    }

    /**
     * Samples a nonlinear additive-noise model: {@code x_j = sum over parents f_ij(x_i) + s_j e_j}
     * with {@code f_ij(u) = sum_m a_m cos(omega_m u + phi_m)} (five random Fourier features) and
     * Gaussian noise of scale {@code s_j ~ U(0.3, 0.6)}.
     *
     * @param dag the DAG (weights ignored)
     * @param n samples
     * @param seed the seed
     * @return {@code [n][d]}
     */
    public static double[][] nonlinearAnm(double[][] dag, int n, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int d = dag.length;
        int m = 5;
        double[][][] f = new double[d][d][];
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) {
                if (dag[i][j] != 0) {
                    double[] p = new double[3 * m];
                    for (int k = 0; k < m; k++) {
                        p[3 * k] = r.nextGaussian() * Math.sqrt(2.0 / m) * 1.5;
                        p[3 * k + 1] = r.nextGaussian();
                        p[3 * k + 2] = 2 * Math.PI * r.nextDouble();
                    }
                    f[i][j] = p;
                }
            }
        }
        double[] s = new double[d];
        for (int j = 0; j < d; j++) {
            s[j] = 0.3 + 0.3 * r.nextDouble();
        }
        int[] order = topologicalOrder(dag);
        double[][] x = new double[n][d];
        for (int t = 0; t < n; t++) {
            for (int j : order) {
                double v = s[j] * r.nextGaussian();
                for (int i = 0; i < d; i++) {
                    if (f[i][j] != null) {
                        for (int k = 0; k < m; k++) {
                            v += f[i][j][3 * k] * Math.cos(f[i][j][3 * k + 1] * x[t][i] + f[i][j][3 * k + 2]);
                        }
                    }
                }
                x[t][j] = v;
            }
        }
        return x;
    }

    /**
     * Data with hidden variables.
     *
     * @param dag the full DAG, observed nodes first
     * @param observed the observed nodes
     * @param data observed columns only, {@code [n][observed.length]}
     */
    public record Hidden(double[][] dag, int[] observed, double[][] data) {
    }

    /**
     * Linear-Gaussian data from a random DAG over {@code d} observed nodes plus {@code hidden}
     * latent roots, each pointing into two observed nodes that are not adjacent.
     *
     * @param d observed nodes
     * @param expectedDegree the observed DAG's expected degree
     * @param hidden latent confounders
     * @param n samples
     * @param seed the seed
     * @return the data and full graph
     */
    public static Hidden confounded(int d, double expectedDegree, int hidden, int n, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][] obs = randomDag(d, expectedDegree, seed);
        int total = d + hidden;
        double[][] w = new double[total][total];
        for (int i = 0; i < d; i++) {
            w[i] = Arrays.copyOf(obs[i], total);
        }
        for (int h = 0; h < hidden; h++) {
            int a;
            int b;
            int guard = 0;
            do {
                a = r.nextInt(d);
                b = r.nextInt(d);
            } while ((a == b || obs[a][b] != 0 || obs[b][a] != 0) && ++guard < 200);
            w[d + h][a] = (r.nextBoolean() ? 1 : -1) * (1 + r.nextDouble());
            w[d + h][b] = (r.nextBoolean() ? 1 : -1) * (1 + r.nextDouble());
        }
        double[][] full = linearSem(w, n, Noise.GAUSSIAN, false, seed + 1);
        int[] observed = new int[d];
        Arrays.setAll(observed, i -> i);
        double[][] data = new double[n][d];
        for (int t = 0; t < n; t++) {
            data[t] = Arrays.copyOf(full[t], d);
        }
        return new Hidden(w, observed, data);
    }

    /**
     * Linear-Gaussian data with selection bias: a hidden common effect {@code S} of two
     * non-adjacent observed nodes, keeping only rows with {@code S > 0}.
     *
     * @param d observed nodes
     * @param expectedDegree the expected degree
     * @param n samples to keep
     * @param seed the seed
     * @return the selected data; the full graph has {@code S} as its last node
     */
    public static Hidden selected(int d, double expectedDegree, int n, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][] obs = randomDag(d, expectedDegree, seed);
        double[][] w = new double[d + 1][d + 1];
        for (int i = 0; i < d; i++) {
            w[i] = Arrays.copyOf(obs[i], d + 1);
        }
        int a;
        int b;
        do {
            a = r.nextInt(d);
            b = r.nextInt(d);
        } while (a == b || obs[a][b] != 0 || obs[b][a] != 0);
        w[a][d] = 1.5;
        w[b][d] = 1.5;
        double[][] full = linearSem(w, 3 * n, Noise.GAUSSIAN, false, seed + 1);
        double[][] data = new double[n][];
        int k = 0;
        for (double[] row : full) {
            if (k < n && row[d] > 0) {
                data[k++] = Arrays.copyOf(row, d);
            }
        }
        data = Arrays.copyOf(data, k);
        int[] observed = new int[d];
        Arrays.setAll(observed, i -> i);
        return new Hidden(w, observed, data);
    }

    /** The mechanism class of a bivariate pair. */
    public enum PairKind {
        /** {@code y = f(x) + e}, nonlinear {@code f}, Gaussian {@code e}. */
        ADDITIVE,
        /** {@code y = f(x) exp(e)}. */
        MULTIPLICATIVE,
        /** {@code y = f(x)} monotone, with tiny noise. */
        DETERMINISTIC,
        /** {@code y = a x + e}, both Gaussian: not identifiable. */
        LINEAR_GAUSSIAN
    }

    /**
     * A bivariate pair.
     *
     * @param x the first variable
     * @param y the second
     * @param xCausesY whether the truth is {@code X -> Y}
     */
    public record Pair(double[] x, double[] y, boolean xCausesY) {
    }

    /**
     * Draws a cause-effect pair in a random orientation.
     *
     * @param kind the mechanism class
     * @param n samples
     * @param seed the seed
     * @return the pair
     */
    public static Pair pair(PairKind kind, int n, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[] c = new double[n];
        double[] e = new double[n];
        double a1 = 0.5 + r.nextDouble();
        double a2 = 0.5 + r.nextDouble();
        double power = 1.5 + 2 * r.nextDouble();
        double sign = r.nextBoolean() ? 1 : -1;
        for (int i = 0; i < n; i++) {
            switch (kind) {
                case ADDITIVE -> {
                    c[i] = r.nextGaussian() * (0.6 + 0.4 * r.nextDouble());
                    e[i] = a1 * Math.tanh(2 * c[i]) + sign * a2 * c[i] * c[i] * 0.5 + 0.2 * r.nextGaussian();
                }
                case MULTIPLICATIVE -> {
                    c[i] = 0.2 + 2 * r.nextDouble();
                    e[i] = (a1 * c[i] + Math.sin(a2 * c[i])) * Math.exp(0.3 * r.nextGaussian());
                }
                case DETERMINISTIC -> {
                    c[i] = r.nextDouble();
                    e[i] = Math.pow(c[i], power) + 0.3 * Math.sin(3 * a1 * c[i]) * c[i] + 0.002 * r.nextGaussian();
                }
                case LINEAR_GAUSSIAN -> {
                    c[i] = r.nextGaussian();
                    e[i] = sign * a1 * c[i] + r.nextGaussian();
                }
                default -> throw new IllegalStateException();
            }
        }
        boolean forward = r.nextBoolean();
        return forward ? new Pair(c, e, true) : new Pair(e, c, false);
    }

    /**
     * A multivariate time series with known lagged links.
     *
     * @param x the series {@code [T][d]}
     * @param truth {@code truth[i][j]}: a cross-link {@code i -> j} exists at some lag
     */
    public record Series(double[][] x, boolean[][] truth) {
    }

    /**
     * A stable VAR(2): every series has autoregression {@code autocorrelation} at lag 1, and
     * about {@code d} random cross-links of magnitude {@code U(0.25, 0.45)} at lag 1 or 2.
     *
     * @param d series
     * @param length time steps (after a burn-in)
     * @param autocorrelation the lag-1 self coefficient, for example 0.5 or 0.95
     * @param seed the seed
     * @return the series and its truth
     */
    public static Series laggedVar(int d, int length, double autocorrelation, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][][] a = new double[2][d][d];   // a[lag][source][target]
        boolean[][] truth = new boolean[d][d];
        for (int e = 0; e < d; e++) {
            int i = r.nextInt(d);
            int j = r.nextInt(d);
            if (i == j || truth[i][j]) {
                continue;
            }
            truth[i][j] = true;
            a[r.nextInt(2)][i][j] = (r.nextBoolean() ? 1 : -1) * (0.25 + 0.2 * r.nextDouble());
        }
        for (int j = 0; j < d; j++) {
            a[0][j][j] = autocorrelation;
        }
        double scale = 1;
        double[][] x;
        while (true) {
            x = simulate(a, scale, length, r.split());
            if (x != null) {
                break;
            }
            scale *= 0.8;
        }
        return new Series(x, truth);
    }

    private static double[][] simulate(double[][][] a, double scale, int length, SplittableRandom r) {
        int d = a[0].length;
        int burn = 200;
        double[][] x = new double[length + burn][d];
        for (int t = 2; t < length + burn; t++) {
            for (int j = 0; j < d; j++) {
                double v = r.nextGaussian();
                for (int i = 0; i < d; i++) {
                    double c1 = i == j ? a[0][i][j] : a[0][i][j] * scale;
                    v += c1 * x[t - 1][i] + a[1][i][j] * scale * x[t - 2][i];
                }
                if (Math.abs(v) > 1e3) {
                    return null;
                }
                x[t][j] = v;
            }
        }
        return Arrays.copyOfRange(x, burn, length + burn);
    }

    /**
     * A nonlinear lagged system: {@code x_j(t) = 0.4 x_j(t-1) + sum over parents
     * g(x_i(t-1)) + e} with {@code g(u) = 1.2 sin(1.5 u)} or {@code 0.6 u^2 - 0.6}, links drawn
     * as in {@link #laggedVar}.
     *
     * @param d series
     * @param length steps
     * @param seed the seed
     * @return the series and truth
     */
    public static Series nonlinearLagged(int d, int length, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        boolean[][] truth = new boolean[d][d];
        int[][] kind = new int[d][d];
        for (int e = 0; e < d; e++) {
            int i = r.nextInt(d);
            int j = r.nextInt(d);
            if (i != j) {
                truth[i][j] = true;
                kind[i][j] = r.nextInt(2);
            }
        }
        int burn = 100;
        double[][] x = new double[length + burn][d];
        for (int t = 1; t < length + burn; t++) {
            for (int j = 0; j < d; j++) {
                double v = 0.4 * x[t - 1][j] + 0.5 * r.nextGaussian();
                for (int i = 0; i < d; i++) {
                    if (truth[i][j]) {
                        double u = Math.max(-3, Math.min(3, x[t - 1][i]));
                        v += kind[i][j] == 0 ? 1.2 * Math.sin(1.5 * u) : 0.6 * u * u - 0.6;
                    }
                }
                x[t][j] = Math.max(-10, Math.min(10, v));
            }
        }
        return new Series(Arrays.copyOfRange(x, burn, length + burn), truth);
    }

    /**
     * Lorenz-96, {@code dx_i/dt = (x_{i+1} - x_{i-2}) x_{i-1} - x_i + F}, integrated by RK4 and
     * sampled every {@code dt}, with small observation noise; the true parents of {@code i} are
     * {@code i-2, i-1, i+1} (cyclically).
     *
     * @param d series, at least 4
     * @param forcing {@code F}, for example 10
     * @param length samples
     * @param dt the sampling step
     * @param seed the seed
     * @return the series and truth
     */
    public static Series lorenz96(int d, double forcing, int length, double dt, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[] s = new double[d];
        for (int i = 0; i < d; i++) {
            s[i] = forcing + 0.5 * r.nextGaussian();
        }
        int burn = 300;
        double[][] x = new double[length][d];
        int sub = 5;
        double h = dt / sub;
        for (int t = 0; t < length + burn; t++) {
            for (int k = 0; k < sub; k++) {
                double[] k1 = l96(s, forcing);
                double[] k2 = l96(axpy(s, k1, h / 2), forcing);
                double[] k3 = l96(axpy(s, k2, h / 2), forcing);
                double[] k4 = l96(axpy(s, k3, h), forcing);
                for (int i = 0; i < d; i++) {
                    s[i] += h / 6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
                }
            }
            if (t >= burn) {
                for (int i = 0; i < d; i++) {
                    x[t - burn][i] = s[i] + 0.1 * r.nextGaussian();
                }
            }
        }
        boolean[][] truth = new boolean[d][d];
        for (int i = 0; i < d; i++) {
            truth[(i - 2 + d) % d][i] = true;
            truth[(i - 1 + d) % d][i] = true;
            truth[(i + 1) % d][i] = true;
        }
        return new Series(x, truth);
    }

    private static double[] axpy(double[] s, double[] k, double h) {
        double[] o = new double[s.length];
        for (int i = 0; i < s.length; i++) {
            o[i] = s[i] + h * k[i];
        }
        return o;
    }

    private static double[] l96(double[] s, double f) {
        int d = s.length;
        double[] o = new double[d];
        for (int i = 0; i < d; i++) {
            o[i] = (s[(i + 1) % d] - s[(i - 2 + d) % d]) * s[(i - 1 + d) % d] - s[i] + f;
        }
        return o;
    }
}
