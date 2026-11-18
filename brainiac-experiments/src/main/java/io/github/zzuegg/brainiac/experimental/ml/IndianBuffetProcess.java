package io.github.zzuegg.brainiac.experimental.ml;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * The linear-Gaussian latent feature model with an Indian buffet process prior (Griffiths and
 * Ghahramani, NeurIPS 2005): {@code X = Z A + E} with a binary feature matrix {@code Z [N][K]}
 * whose number of columns is unbounded, feature loadings {@code A ~ N(0, sigmaA^2)} and noise
 * {@code E ~ N(0, sigmaX^2)}.
 *
 * <p>Inference is collapsed Gibbs sampling: {@code A} is integrated out analytically,
 * {@code p(X | Z) ∝ sigmaX^-(N-K)D sigmaA^-KD |M|^(D/2) exp(-tr(X^T (I - Z M Z^T) X) / 2 sigmaX^2)},
 * {@code M = (Z^T Z + sigmaX^2 / sigmaA^2 I)^-1}. Each {@code z_ik} of a feature that another
 * object also has is resampled from {@code m_{-i,k} / N} times the likelihood; the features
 * owned by object {@code i} alone are replaced by a Metropolis-Hastings move that proposes a
 * {@code Poisson(alpha / N)} number of fresh features and accepts on the likelihood ratio (Meeds
 * et al., NeurIPS 2007). The IBP concentration {@code alpha} has a {@code Gamma(1, 1)} prior and
 * is resampled from its conjugate posterior every sweep.
 */
public final class IndianBuffetProcess {

    /**
     * The final sample.
     *
     * @param z the feature matrix {@code [N][K]}, columns ordered by popularity
     * @param loadings the posterior mean loadings {@code E[A | X, Z]} {@code [K][D]}
     * @param alpha the final IBP concentration
     * @param featureTrace the number of features after each sweep
     */
    public record Result(int[][] z, double[][] loadings, double alpha, int[] featureTrace) {

        /**
         * The number of features.
         *
         * @return {@code K}
         */
        public int features() {
            return loadings.length;
        }

        /**
         * The reconstruction {@code Z E[A]}.
         *
         * @return {@code [N][D]}
         */
        public double[][] reconstruction() {
            int n = z.length;
            int d = loadings.length == 0 ? 0 : loadings[0].length;
            double[][] out = new double[n][d];
            for (int i = 0; i < n; i++) {
                for (int k = 0; k < loadings.length; k++) {
                    if (z[i][k] == 1) {
                        for (int j = 0; j < d; j++) {
                            out[i][j] += loadings[k][j];
                        }
                    }
                }
            }
            return out;
        }
    }

    private IndianBuffetProcess() {
    }

    /**
     * Runs collapsed Gibbs sampling from a single feature held by every object.
     *
     * @param x the data {@code [N][D]}
     * @param sigmaX the noise standard deviation
     * @param sigmaA the loading standard deviation
     * @param alpha the initial IBP concentration
     * @param sweeps Gibbs sweeps
     * @param seed the seed
     * @return the final sample
     */
    public static Result gibbs(double[][] x, double sigmaX, double sigmaA, double alpha, int sweeps, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int n = x.length;
        int d = x[0].length;
        double trXX = 0;
        for (double[] row : x) {
            for (double v : row) {
                trXX += v * v;
            }
        }
        List<int[]> cols = new ArrayList<>();
        int[] first = new int[n];
        java.util.Arrays.fill(first, 1);
        cols.add(first);
        double a = alpha;
        double harmonic = 0;
        for (int i = 1; i <= n; i++) {
            harmonic += 1.0 / i;
        }
        Model model = new Model(x, trXX, sigmaX, sigmaA);
        int[] trace = new int[sweeps];
        for (int s = 0; s < sweeps; s++) {
            for (int i = 0; i < n; i++) {
                // existing features shared with another object
                for (int k = 0; k < cols.size(); k++) {
                    int[] c = cols.get(k);
                    int others = 0;
                    for (int j = 0; j < n; j++) {
                        others += j == i ? 0 : c[j];
                    }
                    if (others == 0) {
                        continue;
                    }
                    c[i] = 0;
                    double l0 = model.logLikelihood(cols);
                    c[i] = 1;
                    double l1 = model.logLikelihood(cols);
                    double lp1 = Math.log(others) + l1;
                    double lp0 = Math.log(n - others) + l0;
                    double p1 = 1 / (1 + Math.exp(lp0 - lp1));
                    c[i] = r.nextDouble() < p1 ? 1 : 0;
                }
                // Metropolis-Hastings on the features object i holds alone
                List<int[]> singles = new ArrayList<>();
                List<int[]> rest = new ArrayList<>();
                for (int[] c : cols) {
                    int total = 0;
                    for (int v : c) {
                        total += v;
                    }
                    if (total == 1 && c[i] == 1) {
                        singles.add(c);
                    } else if (total > 0) {
                        rest.add(c);
                    }
                }
                int proposed = poisson(a / n, r);
                if (proposed != singles.size()) {
                    List<int[]> current = new ArrayList<>(rest);
                    current.addAll(singles);
                    List<int[]> candidate = new ArrayList<>(rest);
                    for (int k = 0; k < proposed; k++) {
                        int[] c = new int[n];
                        c[i] = 1;
                        candidate.add(c);
                    }
                    double ratio = model.logLikelihood(candidate) - model.logLikelihood(current);
                    cols = Math.log(r.nextDouble()) < ratio ? candidate : current;
                } else {
                    cols = new ArrayList<>(rest);
                    cols.addAll(singles);
                }
            }
            cols.removeIf(c -> java.util.Arrays.stream(c).sum() == 0);
            a = gamma(1 + cols.size(), r) / (1 + harmonic);
            trace[s] = cols.size();
        }
        cols.sort((p, q) -> Integer.compare(java.util.Arrays.stream(q).sum(), java.util.Arrays.stream(p).sum()));
        int k = cols.size();
        int[][] z = new int[n][k];
        for (int c = 0; c < k; c++) {
            for (int i = 0; i < n; i++) {
                z[i][c] = cols.get(c)[i];
            }
        }
        return new Result(z, model.loadings(cols), a, trace);
    }

    /** The collapsed likelihood. */
    private static final class Model {
        final double[][] x;
        final double trXX;
        final double sx;
        final double sa;

        Model(double[][] x, double trXX, double sx, double sa) {
            this.x = x;
            this.trXX = trXX;
            this.sx = sx;
            this.sa = sa;
        }

        private double[][] precision(List<int[]> cols) {
            int k = cols.size();
            int n = x.length;
            double[][] p = new double[k][k];
            for (int a = 0; a < k; a++) {
                int[] ca = cols.get(a);
                for (int b = a; b < k; b++) {
                    int[] cb = cols.get(b);
                    int s = 0;
                    for (int i = 0; i < n; i++) {
                        s += ca[i] & cb[i];
                    }
                    p[a][b] = s;
                    p[b][a] = s;
                }
                p[a][a] += sx * sx / (sa * sa);
            }
            return p;
        }

        private double[][] ztx(List<int[]> cols) {
            int k = cols.size();
            int d = x[0].length;
            double[][] m = new double[k][d];
            for (int a = 0; a < k; a++) {
                int[] c = cols.get(a);
                for (int i = 0; i < x.length; i++) {
                    if (c[i] == 1) {
                        for (int j = 0; j < d; j++) {
                            m[a][j] += x[i][j];
                        }
                    }
                }
            }
            return m;
        }

        double logLikelihood(List<int[]> cols) {
            int n = x.length;
            int d = x[0].length;
            int k = cols.size();
            double base = -0.5 * n * d * Math.log(2 * Math.PI) - (n - k) * d * Math.log(sx) - k * d * Math.log(sa);
            if (k == 0) {
                return base - trXX / (2 * sx * sx);
            }
            double[][] l = MlMatrices.cholesky(precision(cols));
            double[][] zx = ztx(cols);
            double quad = 0;
            double[] col = new double[k];
            for (int j = 0; j < d; j++) {
                for (int a = 0; a < k; a++) {
                    col[a] = zx[a][j];
                }
                double[] v = MlMatrices.forwardSolve(l, col);
                quad += MlMatrices.dot(v, v);
            }
            return base - 0.5 * d * MlMatrices.logDet(l) - (trXX - quad) / (2 * sx * sx);
        }

        double[][] loadings(List<int[]> cols) {
            int k = cols.size();
            int d = x[0].length;
            if (k == 0) {
                return new double[0][d];
            }
            double[][] l = MlMatrices.cholesky(precision(cols));
            double[][] zx = ztx(cols);
            double[][] out = new double[k][d];
            double[] col = new double[k];
            for (int j = 0; j < d; j++) {
                for (int a = 0; a < k; a++) {
                    col[a] = zx[a][j];
                }
                double[] v = MlMatrices.choleskySolve(l, col);
                for (int a = 0; a < k; a++) {
                    out[a][j] = v[a];
                }
            }
            return out;
        }
    }

    private static int poisson(double lambda, SplittableRandom r) {
        double l = Math.exp(-lambda);
        int k = 0;
        double p = r.nextDouble();
        while (p > l) {
            k++;
            p *= r.nextDouble();
        }
        return k;
    }

    /** A {@code Gamma(shape, 1)} draw (Marsaglia and Tsang). */
    static double gamma(double shape, SplittableRandom r) {
        if (shape < 1) {
            return gamma(shape + 1, r) * Math.pow(r.nextDouble(), 1 / shape);
        }
        double dd = shape - 1.0 / 3;
        double c = 1 / Math.sqrt(9 * dd);
        while (true) {
            double z = r.nextGaussian();
            double v = 1 + c * z;
            if (v <= 0) {
                continue;
            }
            v = v * v * v;
            double u = r.nextDouble();
            if (Math.log(u) < 0.5 * z * z + dd - dd * v + dd * Math.log(v)) {
                return dd * v;
            }
        }
    }
}
