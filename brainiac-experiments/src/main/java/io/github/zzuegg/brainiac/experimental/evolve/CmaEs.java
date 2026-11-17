package io.github.zzuegg.brainiac.experimental.evolve;

import java.util.Arrays;
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.function.ToDoubleFunction;

/**
 * Covariance Matrix Adaptation Evolution Strategy: samples candidates from a multivariate
 * Gaussian, and moves its mean, scale and full covariance matrix toward the best samples.
 * It learns the shape of the landscape (such as a narrow curved valley), so it is the
 * strongest black-box optimizer for small problems, up to a few hundred dimensions; for
 * larger ones use {@link EvolutionStrategy}.
 *
 * <p>Follows Hansen, "The CMA Evolution Strategy: A Tutorial",
 * <a href="https://arxiv.org/abs/1604.00772">arXiv:1604.00772</a>: weighted recombination,
 * cumulative step-size adaptation, rank-one and rank-mu covariance updates. Costs are
 * <b>minimised</b>.
 *
 * <pre>{@code
 * CmaEs cma = CmaEs.of(new double[4], 0.5).seed(3);
 * double[] best = cma.minimize(x -> rosenbrock(x), 4_000);
 * }</pre>
 */
public final class CmaEs {

    private final int n;
    private final double[] mean;
    private double sigma;
    private int lambda;
    private long seed = 1;
    private double[] bestX;
    private double bestCost = Double.POSITIVE_INFINITY;
    private int evaluations;

    private CmaEs(double[] start, double sigma) {
        this.n = start.length;
        this.mean = start.clone();
        this.sigma = sigma;
        this.lambda = 4 + (int) Math.floor(3 * Math.log(n));
    }

    /**
     * Starts a search.
     *
     * @param start the initial mean
     * @param sigma the initial step size
     * @return the search, to configure
     */
    public static CmaEs of(double[] start, double sigma) {
        Objects.requireNonNull(start, "start");
        if (start.length == 0 || !(sigma > 0)) {
            throw new IllegalArgumentException("need a non-empty start and a positive sigma");
        }
        return new CmaEs(start, sigma);
    }

    /**
     * Sets the population size (default {@code 4 + 3 ln n}).
     *
     * @param size the samples per generation
     * @return this
     */
    public CmaEs population(int size) {
        if (size < 4) {
            throw new IllegalArgumentException("population must be at least 4");
        }
        this.lambda = size;
        return this;
    }

    /**
     * Sets the random seed.
     *
     * @param s the seed
     * @return this
     */
    public CmaEs seed(long s) {
        this.seed = s;
        return this;
    }

    /** @return the best point seen */
    public double[] best() {
        return bestX == null ? mean.clone() : bestX.clone();
    }

    /** @return the cost of the best point seen */
    public double bestCost() {
        return bestCost;
    }

    /** @return how many times the cost was evaluated */
    public int evaluations() {
        return evaluations;
    }

    /** @return the current step size */
    public double sigma() {
        return sigma;
    }

    /**
     * Minimises a cost.
     *
     * @param cost the function to minimise
     * @param maxEvaluations the evaluation budget
     * @return the best point found
     */
    public double[] minimize(ToDoubleFunction<double[]> cost, int maxEvaluations) {
        SplittableRandom random = new SplittableRandom(seed);
        int mu = lambda / 2;
        double[] weights = new double[mu];
        double sum = 0;
        for (int i = 0; i < mu; i++) {
            weights[i] = Math.log(mu + 0.5) - Math.log(i + 1);
            sum += weights[i];
        }
        double sumSq = 0;
        for (int i = 0; i < mu; i++) {
            weights[i] /= sum;
            sumSq += weights[i] * weights[i];
        }
        double mueff = 1 / sumSq;
        double cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n);
        double cs = (mueff + 2) / (n + mueff + 5);
        double c1 = 2 / ((n + 1.3) * (n + 1.3) + mueff);
        double cmu = Math.min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) * (n + 2) + mueff));
        double damps = 1 + 2 * Math.max(0, Math.sqrt((mueff - 1) / (n + 1)) - 1) + cs;
        double chiN = Math.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21.0 * n * n));

        double[] pc = new double[n];
        double[] ps = new double[n];
        double[][] c = identity(n);
        double[][] b = identity(n);
        double[] d = new double[n];
        Arrays.fill(d, 1);
        int eigenAt = 0;
        int generation = 0;
        while (evaluations + lambda <= maxEvaluations) {
            double[][] z = new double[lambda][n];
            double[][] y = new double[lambda][n];
            double[][] x = new double[lambda][n];
            double[] costs = new double[lambda];
            for (int k = 0; k < lambda; k++) {
                for (int i = 0; i < n; i++) {
                    z[k][i] = random.nextGaussian();
                }
                for (int i = 0; i < n; i++) {
                    double s = 0;
                    for (int j = 0; j < n; j++) {
                        s += b[i][j] * d[j] * z[k][j];
                    }
                    y[k][i] = s;
                    x[k][i] = mean[i] + sigma * s;
                }
                costs[k] = cost.applyAsDouble(x[k]);
                evaluations++;
                if (costs[k] < bestCost) {
                    bestCost = costs[k];
                    bestX = x[k].clone();
                }
            }
            Integer[] order = new Integer[lambda];
            for (int k = 0; k < lambda; k++) {
                order[k] = k;
            }
            Arrays.sort(order, (p, q) -> Double.compare(costs[p], costs[q]));
            double[] yw = new double[n];
            for (int r = 0; r < mu; r++) {
                for (int i = 0; i < n; i++) {
                    yw[i] += weights[r] * y[order[r]][i];
                }
            }
            for (int i = 0; i < n; i++) {
                mean[i] += sigma * yw[i];
            }
            // C^{-1/2} yw = B D^{-1} B^T yw
            double[] tmp = new double[n];
            for (int j = 0; j < n; j++) {
                double s = 0;
                for (int i = 0; i < n; i++) {
                    s += b[i][j] * yw[i];
                }
                tmp[j] = s / d[j];
            }
            double psNorm = 0;
            for (int i = 0; i < n; i++) {
                double s = 0;
                for (int j = 0; j < n; j++) {
                    s += b[i][j] * tmp[j];
                }
                ps[i] = (1 - cs) * ps[i] + Math.sqrt(cs * (2 - cs) * mueff) * s;
                psNorm += ps[i] * ps[i];
            }
            psNorm = Math.sqrt(psNorm);
            generation++;
            boolean hsig = psNorm / Math.sqrt(1 - Math.pow(1 - cs, 2 * generation)) / chiN < 1.4 + 2.0 / (n + 1);
            for (int i = 0; i < n; i++) {
                pc[i] = (1 - cc) * pc[i] + (hsig ? Math.sqrt(cc * (2 - cc) * mueff) * yw[i] : 0);
            }
            double deltaH = hsig ? 0 : cc * (2 - cc);
            for (int i = 0; i < n; i++) {
                for (int j = 0; j <= i; j++) {
                    double rankMu = 0;
                    for (int r = 0; r < mu; r++) {
                        rankMu += weights[r] * y[order[r]][i] * y[order[r]][j];
                    }
                    double v = (1 - c1 - cmu) * c[i][j] + c1 * (pc[i] * pc[j] + deltaH * c[i][j]) + cmu * rankMu;
                    c[i][j] = v;
                    c[j][i] = v;
                }
            }
            sigma *= Math.exp((cs / damps) * (psNorm / chiN - 1));
            if (evaluations - eigenAt > lambda / (c1 + cmu) / n / 10) {
                eigenAt = evaluations;
                double[][] vectors = new double[n][n];
                double[] values = jacobiEigen(c, vectors);
                for (int i = 0; i < n; i++) {
                    d[i] = Math.sqrt(Math.max(values[i], 1e-20));
                }
                b = vectors;
            }
            if (sigma * d[maxIndex(d)] < 1e-14) {
                break;   // converged
            }
        }
        return best();
    }

    private static int maxIndex(double[] a) {
        int m = 0;
        for (int i = 1; i < a.length; i++) {
            if (a[i] > a[m]) {
                m = i;
            }
        }
        return m;
    }

    private static double[][] identity(int n) {
        double[][] m = new double[n][n];
        for (int i = 0; i < n; i++) {
            m[i][i] = 1;
        }
        return m;
    }

    /**
     * Eigen-decomposes a symmetric matrix with cyclic Jacobi rotations.
     *
     * @param matrix the symmetric matrix (not changed)
     * @param vectors receives the eigenvectors as columns
     * @return the eigenvalues
     */
    static double[] jacobiEigen(double[][] matrix, double[][] vectors) {
        int n = matrix.length;
        double[][] a = new double[n][];
        for (int i = 0; i < n; i++) {
            a[i] = matrix[i].clone();
            Arrays.fill(vectors[i], 0);
            vectors[i][i] = 1;
        }
        for (int sweep = 0; sweep < 100; sweep++) {
            double off = 0;
            for (int i = 0; i < n; i++) {
                for (int j = i + 1; j < n; j++) {
                    off += a[i][j] * a[i][j];
                }
            }
            if (off < 1e-22) {
                break;
            }
            for (int p = 0; p < n; p++) {
                for (int q = p + 1; q < n; q++) {
                    if (Math.abs(a[p][q]) < 1e-300) {
                        continue;
                    }
                    double theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
                    double t = Math.signum(theta) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
                    if (theta == 0) {
                        t = 1;
                    }
                    double cs = 1 / Math.sqrt(t * t + 1);
                    double sn = t * cs;
                    for (int k = 0; k < n; k++) {
                        double akp = a[k][p];
                        double akq = a[k][q];
                        a[k][p] = cs * akp - sn * akq;
                        a[k][q] = sn * akp + cs * akq;
                    }
                    for (int k = 0; k < n; k++) {
                        double apk = a[p][k];
                        double aqk = a[q][k];
                        a[p][k] = cs * apk - sn * aqk;
                        a[q][k] = sn * apk + cs * aqk;
                    }
                    for (int k = 0; k < n; k++) {
                        double vkp = vectors[k][p];
                        double vkq = vectors[k][q];
                        vectors[k][p] = cs * vkp - sn * vkq;
                        vectors[k][q] = sn * vkp + cs * vkq;
                    }
                }
            }
        }
        double[] values = new double[n];
        for (int i = 0; i < n; i++) {
            values[i] = a[i][i];
        }
        return values;
    }
}
