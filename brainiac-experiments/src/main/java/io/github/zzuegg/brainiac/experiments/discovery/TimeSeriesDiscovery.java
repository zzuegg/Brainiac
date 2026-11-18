package io.github.zzuegg.brainiac.experiments.discovery;

import io.github.zzuegg.brainiac.experimental.data.DiscoveryBenchmarks;
import io.github.zzuegg.brainiac.experimental.discovery.Distributions;
import io.github.zzuegg.brainiac.experimental.discovery.GrangerCausality;
import io.github.zzuegg.brainiac.experimental.discovery.LeastSquares;
import io.github.zzuegg.brainiac.experimental.discovery.NeuralGranger;
import io.github.zzuegg.brainiac.experimental.discovery.Pcmci;
import io.github.zzuegg.brainiac.experimental.ml.Notears;
import io.github.zzuegg.brainiac.experimental.sciml.Sindy;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Wave 11 item 11.3.5: lagged causal discovery on {@link DiscoveryBenchmarks} time series. Every
 * method gives a score per ordered pair {@code i -> j} ({@code i != j}); the lagged-edge AUROC is
 * computed against the true cross-links, and for p-value methods the false-positive rate is the
 * share of non-links with {@code p < 0.05} (score-only methods: any non-zero score).
 * <ul>
 *   <li>{@link GrangerCausality} (conditional VAR F-tests, lag by BIC), pairwise bivariate Granger
 *       tests, {@link Pcmci} with partial correlations and {@link NeuralGranger};</li>
 *   <li>baselines: the largest lagged Pearson correlation (Fisher-z p-value), NOTEARS on stacked
 *       lags (weights from the lag columns into the present columns) and SINDy-style sparse
 *       regression of {@code x_t} on {@code x_{t-1}} and {@code x_{t-1}^2} read as a graph.</li>
 * </ul>
 */
public final class TimeSeriesDiscovery {

    private TimeSeriesDiscovery() {
    }

    /** The systems compared. */
    public enum Kind {
        /** A VAR(2) with lag-1 autocorrelation 0.5. */
        VAR,
        /** A VAR(2) with lag-1 autocorrelation 0.95. */
        AUTOCORRELATED,
        /** Nonlinear lagged links. */
        NONLINEAR,
        /** Lorenz-96 with forcing 10. */
        LORENZ96
    }

    private static DiscoveryBenchmarks.Series series(Kind kind, int d, int length, long seed) {
        return switch (kind) {
            case VAR -> DiscoveryBenchmarks.laggedVar(d, length, 0.5, seed);
            case AUTOCORRELATED -> DiscoveryBenchmarks.laggedVar(d, length, 0.95, seed);
            case NONLINEAR -> DiscoveryBenchmarks.nonlinearLagged(d, length, seed);
            case LORENZ96 -> DiscoveryBenchmarks.lorenz96(d, 10, length, 0.05, seed);
        };
    }

    /**
     * The AUROC of scores for the true links.
     *
     * @param score higher means more likely a link
     * @param truth the true links
     * @return the AUROC over off-diagonal pairs (ties count half)
     */
    public static double auroc(double[][] score, boolean[][] truth) {
        List<Double> pos = new ArrayList<>();
        List<Double> neg = new ArrayList<>();
        for (int i = 0; i < score.length; i++) {
            for (int j = 0; j < score.length; j++) {
                if (i != j) {
                    (truth[i][j] ? pos : neg).add(score[i][j]);
                }
            }
        }
        if (pos.isEmpty() || neg.isEmpty()) {
            return Double.NaN;
        }
        double s = 0;
        for (double p : pos) {
            for (double q : neg) {
                s += p > q ? 1 : p == q ? 0.5 : 0;
            }
        }
        return s / (pos.size() * (double) neg.size());
    }

    private static double[] pScore(double[][] p, boolean[][] truth) {
        int d = p.length;
        double[][] s = new double[d][d];
        int fp = 0;
        int neg = 0;
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) {
                if (i == j) {
                    continue;
                }
                s[i][j] = -p[i][j];
                if (!truth[i][j]) {
                    neg++;
                    fp += p[i][j] < 0.05 ? 1 : 0;
                }
            }
        }
        return new double[] {auroc(s, truth), fp / (double) Math.max(1, neg)};
    }

    private static double[] magnitudeScore(double[][] s, boolean[][] truth) {
        int d = s.length;
        int fp = 0;
        int neg = 0;
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) {
                if (i != j && !truth[i][j]) {
                    neg++;
                    fp += Math.abs(s[i][j]) > 1e-9 ? 1 : 0;
                }
            }
        }
        double[][] a = new double[d][d];
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) {
                a[i][j] = Math.abs(s[i][j]);
            }
        }
        return new double[] {auroc(a, truth), fp / (double) Math.max(1, neg)};
    }

    private static double[][] laggedCorrelation(double[][] x, int lags) {
        int d = x[0].length;
        int n = x.length;
        double[][] p = new double[d][d];
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) {
                double best = 1;
                for (int l = 1; l <= lags; l++) {
                    double[] a = new double[n - l];
                    double[] b = new double[n - l];
                    for (int t = l; t < n; t++) {
                        a[t - l] = x[t - l][i];
                        b[t - l] = x[t][j];
                    }
                    double r = Math.max(-0.999999, Math.min(0.999999, LeastSquares.correlation(a, b)));
                    double z = 0.5 * Math.log((1 + r) / (1 - r)) * Math.sqrt(n - l - 3);
                    best = Math.min(best, Distributions.normalTwoSided(z) * lags);
                }
                p[i][j] = Math.min(1, best);
            }
        }
        return p;
    }

    private static double[][] stackedNotears(double[][] x, int lags) {
        int d = x[0].length;
        int n = x.length - lags;
        double[][] s = new double[n][d * (lags + 1)];
        for (int t = 0; t < n; t++) {
            for (int l = 0; l <= lags; l++) {
                for (int j = 0; j < d; j++) {
                    s[t][l * d + j] = x[t + lags - l][j];
                }
            }
        }
        double[][] w = new Notears(0.05, 0.3, 300, 10).fit(LeastSquares.standardise(s));
        double[][] out = new double[d][d];
        for (int l = 1; l <= lags; l++) {
            for (int i = 0; i < d; i++) {
                for (int j = 0; j < d; j++) {
                    out[i][j] += Math.abs(w[l * d + i][j]);
                }
            }
        }
        return out;
    }

    private static double[][] sindy(double[][] x) {
        int d = x[0].length;
        int n = x.length - 1;
        double[][] theta = new double[n][2 * d];
        double[][] next = new double[n][d];
        double[][] z = LeastSquares.standardise(x);
        for (int t = 0; t < n; t++) {
            for (int i = 0; i < d; i++) {
                theta[t][i] = z[t][i];
                theta[t][d + i] = z[t][i] * z[t][i];
                next[t][i] = z[t + 1][i];
            }
        }
        double[][] xi = Sindy.stlsq(theta, next, 0.05, 1e-8, 10);
        double[][] out = new double[d][d];
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) {
                out[i][j] = Math.abs(xi[i][j]) + Math.abs(xi[d + i][j]);
            }
        }
        return out;
    }

    /**
     * Compares the methods on one kind of system.
     *
     * @param kind the system
     * @param d series
     * @param length time steps
     * @param systems independent systems averaged
     * @param neuralSteps proximal-gradient steps of neural Granger (0 skips it)
     * @param seed the seed
     * @return method to {@code {AUROC, false-positive rate}}
     */
    public static Map<String, double[]> compare(Kind kind, int d, int length, int systems, int neuralSteps,
            long seed) {
        Map<String, double[]> out = new LinkedHashMap<>();
        for (int s = 0; s < systems; s++) {
            DiscoveryBenchmarks.Series data = series(kind, d, length, seed + s);
            double[][] x = data.x();
            boolean[][] truth = data.truth();
            int lags = Math.max(2, GrangerCausality.selectLag(x, 4));
            acc(out, "Granger (conditional VAR)", pScore(GrangerCausality.conditional(x, lags), truth), systems);
            acc(out, "Granger (pairwise)", pScore(GrangerCausality.pairwise(x, lags), truth), systems);
            Pcmci pcmci = new Pcmci(2, 0.2, 3, Pcmci.Test.PARTIAL_CORRELATION);
            acc(out, "PCMCI (ParCorr)", pScore(Pcmci.minOverLags(pcmci.fit(x)), truth), systems);
            if (neuralSteps > 0) {
                double[][] ng = new NeuralGranger(2, 8, 0.1, 0.05, neuralSteps).fit(x, seed + s);
                acc(out, "neural Granger (cMLP)", magnitudeScore(ng, truth), systems);
            }
            acc(out, "lagged correlation", pScore(laggedCorrelation(x, 2), truth), systems);
            acc(out, "NOTEARS on stacked lags", magnitudeScore(stackedNotears(x, 2), truth), systems);
            acc(out, "SINDy as a graph", magnitudeScore(sindy(x), truth), systems);
        }
        return out;
    }

    private static void acc(Map<String, double[]> out, String key, double[] v, int runs) {
        double[] a = out.computeIfAbsent(key, k -> new double[v.length]);
        for (int i = 0; i < v.length; i++) {
            a[i] += v[i] / runs;
        }
    }
}
