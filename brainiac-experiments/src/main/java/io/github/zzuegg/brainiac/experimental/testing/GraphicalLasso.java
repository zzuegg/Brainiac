package io.github.zzuegg.brainiac.experimental.testing;

import io.github.zzuegg.brainiac.experimental.discovery.LeastSquares;
import io.github.zzuegg.brainiac.experimental.ml.MlMatrices;
import io.github.zzuegg.brainiac.experimental.stats.PenalizedRegression;

/**
 * Sparse Gaussian graphical models: the conditional-independence graph of a Gaussian is the
 * off-diagonal support of its precision matrix {@code Theta}.
 *
 * <ul>
 *   <li>{@link Method#GLASSO}: the graphical lasso (Friedman, Hastie and Tibshirani,
 *       Biostatistics 2008, <a href="https://arxiv.org/abs/0708.3517">arXiv:0708.3517</a>)
 *       maximises {@code log det Theta - tr(S Theta) - lambda ||Theta||_1} (off-diagonal) by block
 *       coordinate descent on the covariance {@code W}: each column solves a lasso in the
 *       covariance form, then {@code Theta} is read off the blocks.</li>
 *   <li>{@link Method#NEIGHBOURHOOD_AND}/{@link Method#NEIGHBOURHOOD_OR}: neighbourhood selection
 *       (Meinshausen and Buhlmann, Annals 2006,
 *       <a href="https://arxiv.org/abs/math/0608017">arXiv:math/0608017</a>) regresses every
 *       variable on the others with the {@link PenalizedRegression} lasso and joins the supports
 *       by AND or OR.</li>
 * </ul>
 * Lambda is picked by the extended BIC (Foygel and Drton 2010) with {@code gamma = 0.5}.
 */
public final class GraphicalLasso {

    /** The estimator. */
    public enum Method {
        /** Penalised likelihood, block coordinate descent. */
        GLASSO,
        /** Per-node lasso, an edge when both endpoints select each other. */
        NEIGHBOURHOOD_AND,
        /** Per-node lasso, an edge when either endpoint selects the other. */
        NEIGHBOURHOOD_OR
    }

    /**
     * An estimate.
     *
     * @param precision the estimated precision (GLASSO), or {@code null} for neighbourhood
     *     selection, which estimates the graph only
     * @param edges the estimated graph
     * @param lambda the chosen level
     */
    public record Estimate(double[][] precision, boolean[][] edges, double lambda) {
    }

    private GraphicalLasso() {
    }

    /**
     * Fits with the level chosen by EBIC over a log grid of {@code count} levels.
     *
     * @param method the estimator
     * @param data samples {@code [n][p]}
     * @param count grid size
     * @param ebicGamma the EBIC parameter (0.5 is standard)
     * @return the chosen estimate
     */
    public static Estimate fitEbic(Method method, double[][] data, int count, double ebicGamma) {
        int n = data.length;
        int p = data[0].length;
        double[][] s = correlation(data);
        double maxOff = 0;
        for (int i = 0; i < p; i++) {
            for (int j = 0; j < p; j++) {
                if (i != j) {
                    maxOff = Math.max(maxOff, Math.abs(s[i][j]));
                }
            }
        }
        if (method != Method.GLASSO) {
            return neighbourhoods(data, method == Method.NEIGHBOURHOOD_AND, ebicGamma);
        }
        Estimate best = null;
        double bestScore = Double.POSITIVE_INFINITY;
        double[][] warm = null;
        for (double lam : PenalizedRegression.grid(maxOff, count, 0.05)) {
            double[][][] wt = glasso(s, constant(p, lam), warm);
            warm = wt[0];
            double[][] theta = wt[1];
            // score the support by its (nearly) unpenalised refit: the shrunk estimate's
            // likelihood is biased low and would push EBIC towards empty graphs
            double[][] refitPenalty = new double[p][p];
            for (int i = 0; i < p; i++) {
                for (int j = 0; j < p; j++) {
                    refitPenalty[i][j] = i == j ? 0 : theta[i][j] != 0 ? 0.01 * lam : 1e3;
                }
            }
            double[][] refit = glasso(s, refitPenalty, wt[0])[1];
            int edges = 0;
            for (int i = 0; i < p; i++) {
                for (int j = i + 1; j < p; j++) {
                    if (theta[i][j] != 0) {
                        edges++;
                    }
                }
            }
            double ll = n / 2.0 * (MlMatrices.logDet(MlMatrices.cholesky(refit)) - trace(s, refit));
            double score = -2 * ll + edges * Math.log(n) + 4 * ebicGamma * edges * Math.log(p);
            if (score < bestScore) {
                bestScore = score;
                best = new Estimate(rescale(theta, data), support(theta), lam);
            }
        }
        return best;
    }

    /**
     * The graphical lasso at one level on a covariance or correlation matrix.
     *
     * @param s the empirical covariance {@code [p][p]}
     * @param lambda the l1 level on off-diagonal entries
     * @return the precision estimate
     */
    public static double[][] precision(double[][] s, double lambda) {
        return glasso(s, constant(s.length, lambda), null)[1];
    }

    private static double[][] constant(int p, double lambda) {
        double[][] l = new double[p][p];
        for (int i = 0; i < p; i++) {
            for (int j = 0; j < p; j++) {
                l[i][j] = lambda;
            }
        }
        return l;
    }

    /** Returns {W, Theta}; {@code lambda} is an elementwise penalty matrix. */
    private static double[][][] glasso(double[][] s, double[][] lambda, double[][] warm) {
        int p = s.length;
        double[][] w = new double[p][p];
        double[][] beta = new double[p][p - 1];
        for (int i = 0; i < p; i++) {
            for (int j = 0; j < p; j++) {
                w[i][j] = warm != null ? warm[i][j] : s[i][j];
            }
            w[i][i] = s[i][i] + lambda[i][i];
        }
        for (int sweep = 0; sweep < 100; sweep++) {
            double change = 0;
            for (int j = 0; j < p; j++) {
                int[] idx = new int[p - 1];
                for (int a = 0, k = 0; a < p; a++) {
                    if (a != j) {
                        idx[k++] = a;
                    }
                }
                double[] b = beta[j];
                // initialise from W: b solves W11 b = w12 at the previous lasso; keep the last b
                for (int inner = 0; inner < 200; inner++) {
                    double dmax = 0;
                    for (int k = 0; k < p - 1; k++) {
                        double r = s[idx[k]][j];
                        for (int l = 0; l < p - 1; l++) {
                            if (l != k) {
                                r -= w[idx[k]][idx[l]] * b[l];
                            }
                        }
                        double nb = PenalizedRegression.soft(r, lambda[idx[k]][j]) / w[idx[k]][idx[k]];
                        dmax = Math.max(dmax, Math.abs(nb - b[k]));
                        b[k] = nb;
                    }
                    if (dmax < 1e-6) {
                        break;
                    }
                }
                for (int k = 0; k < p - 1; k++) {
                    double v = 0;
                    for (int l = 0; l < p - 1; l++) {
                        v += w[idx[k]][idx[l]] * b[l];
                    }
                    change = Math.max(change, Math.abs(v - w[idx[k]][j]));
                    w[idx[k]][j] = w[j][idx[k]] = v;
                }
            }
            if (change < 1e-5) {
                break;
            }
        }
        double[][] theta = new double[p][p];
        for (int j = 0; j < p; j++) {
            double[] b = beta[j];
            double q = w[j][j];
            int k = 0;
            for (int a = 0; a < p; a++) {
                if (a != j) {
                    q -= w[a][j] * b[k++];
                }
            }
            double t22 = 1 / q;
            theta[j][j] = t22;
            k = 0;
            for (int a = 0; a < p; a++) {
                if (a != j) {
                    theta[a][j] = -b[k++] * t22;
                }
            }
        }
        // symmetrise by the smaller magnitude (keeps exact zeros)
        for (int i = 0; i < p; i++) {
            for (int j = i + 1; j < p; j++) {
                double a = theta[i][j];
                double c = theta[j][i];
                double v = Math.abs(a) < Math.abs(c) ? a : c;
                theta[i][j] = theta[j][i] = v;
            }
        }
        return new double[][][] {w, theta};
    }

    private static Estimate neighbourhoods(double[][] data, boolean and, double ebicGamma) {
        int n = data.length;
        int p = data[0].length;
        boolean[][] sel = new boolean[p][p];
        PenalizedRegression lasso = PenalizedRegression.of(PenalizedRegression.Penalty.LASSO);
        for (int j = 0; j < p; j++) {
            double[][] x = new double[n][p - 1];
            double[] y = new double[n];
            for (int i = 0; i < n; i++) {
                y[i] = data[i][j];
                for (int a = 0, k = 0; a < p; a++) {
                    if (a != j) {
                        x[i][k++] = data[i][a];
                    }
                }
            }
            PenalizedRegression.Path path = lasso.path(x, y, 25, 0.02);
            // EBIC of the least-squares refit on each support along the path
            double bestScore = Double.POSITIVE_INFINITY;
            boolean[] best = new boolean[p - 1];
            for (PenalizedRegression.Fit f : path.fits()) {
                int[] cols = java.util.stream.IntStream.range(0, p - 1).filter(k -> f.beta()[k] != 0).toArray();
                if (cols.length >= n / 2) {
                    break;
                }
                double rss = LeastSquares.fit(LeastSquares.columns(x, cols), y, true, 1e-9).rss();
                double score = n * Math.log(rss / n) + cols.length * (Math.log(n) + 2 * ebicGamma * Math.log(p - 1));
                if (score < bestScore) {
                    bestScore = score;
                    best = f.support();
                }
            }
            for (int a = 0, k = 0; a < p; a++) {
                if (a != j) {
                    sel[j][a] = best[k++];
                }
            }
        }
        boolean[][] e = new boolean[p][p];
        for (int i = 0; i < p; i++) {
            for (int j = 0; j < p; j++) {
                if (i != j) {
                    e[i][j] = and ? sel[i][j] && sel[j][i] : sel[i][j] || sel[j][i];
                }
            }
        }
        return new Estimate(null, e, Double.NaN);
    }

    /**
     * The empirical covariance with divisor {@code n}.
     *
     * @param data samples {@code [n][p]}
     * @return the covariance
     */
    public static double[][] covariance(double[][] data) {
        int n = data.length;
        int p = data[0].length;
        double[] m = MlMatrices.columnMeans(data);
        double[][] s = new double[p][p];
        for (double[] row : data) {
            for (int i = 0; i < p; i++) {
                double di = row[i] - m[i];
                for (int j = i; j < p; j++) {
                    s[i][j] += di * (row[j] - m[j]);
                }
            }
        }
        for (int i = 0; i < p; i++) {
            for (int j = i; j < p; j++) {
                s[i][j] /= n;
                s[j][i] = s[i][j];
            }
        }
        return s;
    }

    private static double[][] correlation(double[][] data) {
        double[][] s = covariance(data);
        int p = s.length;
        double[][] c = new double[p][p];
        for (int i = 0; i < p; i++) {
            for (int j = 0; j < p; j++) {
                c[i][j] = s[i][j] / Math.sqrt(s[i][i] * s[j][j]);
            }
        }
        return c;
    }

    /** Maps a precision of the correlation back to the covariance scale. */
    private static double[][] rescale(double[][] theta, double[][] data) {
        double[][] s = covariance(data);
        int p = theta.length;
        double[][] out = new double[p][p];
        for (int i = 0; i < p; i++) {
            for (int j = 0; j < p; j++) {
                out[i][j] = theta[i][j] / Math.sqrt(s[i][i] * s[j][j]);
            }
        }
        return out;
    }

    private static boolean[][] support(double[][] theta) {
        int p = theta.length;
        boolean[][] e = new boolean[p][p];
        for (int i = 0; i < p; i++) {
            for (int j = 0; j < p; j++) {
                e[i][j] = i != j && theta[i][j] != 0;
            }
        }
        return e;
    }

    private static double trace(double[][] a, double[][] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a.length; j++) {
                s += a[i][j] * b[j][i];
            }
        }
        return s;
    }

    /**
     * Edge F1 of an estimated graph against the truth (upper triangle).
     *
     * @param estimate the estimated graph
     * @param truth the true graph
     * @return the F1 score
     */
    public static double edgeF1(boolean[][] estimate, boolean[][] truth) {
        int tp = 0;
        int fp = 0;
        int fn = 0;
        for (int i = 0; i < truth.length; i++) {
            for (int j = i + 1; j < truth.length; j++) {
                if (estimate[i][j] && truth[i][j]) {
                    tp++;
                } else if (estimate[i][j]) {
                    fp++;
                } else if (truth[i][j]) {
                    fn++;
                }
            }
        }
        return tp == 0 ? 0 : 2.0 * tp / (2.0 * tp + fp + fn);
    }
}
