package io.github.zzuegg.brainiac.experimental.stats;

/**
 * A generalised additive model {@code g(E[y]) = b0 + sum_j f_j(x_j)} with one
 * {@link PenalizedSplines} smooth per feature (Hastie and Tibshirani, 1990; Wood, JRSS-B 2011),
 * fitted by backfitting: cycle over features, smoothing the partial residuals
 * {@code y - b0 - sum_{k != j} f_k} and centring each smooth. Logistic GAMs use penalised IRLS
 * (P-IRLS): an outer loop forms the working response and weights, the inner loop backfits them.
 * Each smooth's penalty is chosen by GCV in the first sweep and then held fixed, so backfitting
 * converges.
 */
public final class AdditiveModel {

    /** The response family. */
    public enum Family {
        /** Identity link, squared error. */
        GAUSSIAN,
        /** Logit link, 0/1 targets. */
        LOGISTIC
    }

    private final Family family;
    private final double intercept;
    private final PenalizedSplines[] smooths;
    private final double[] centres;

    private AdditiveModel(Family family, double intercept, PenalizedSplines[] smooths, double[] centres) {
        this.family = family;
        this.intercept = intercept;
        this.smooths = smooths;
        this.centres = centres;
    }

    /**
     * Fits a GAM.
     *
     * @param x the design {@code [n][p]}
     * @param y the response
     * @param family the family
     * @param segments B-spline segments per smooth
     * @return the model
     */
    public static AdditiveModel fit(double[][] x, double[] y, Family family, int segments) {
        int n = y.length;
        int p = x[0].length;
        double[][] cols = new double[p][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < p; j++) {
                cols[j][i] = x[i][j];
            }
        }
        double[][] f = new double[p][n];
        PenalizedSplines[] sm = new PenalizedSplines[p];
        double[] centres = new double[p];
        double[] lambdas = new double[p];
        java.util.Arrays.fill(lambdas, Double.NaN);
        double b0;
        if (family == Family.GAUSSIAN) {
            b0 = mean(y);
            backfit(cols, y, null, b0, f, sm, centres, lambdas, segments, 30);
        } else {
            double m = Math.min(0.99, Math.max(0.01, mean(y)));
            b0 = Math.log(m / (1 - m));
            double[] eta = new double[n];
            for (int outer = 0; outer < 25; outer++) {
                double[] z = new double[n];
                double[] w = new double[n];
                for (int i = 0; i < n; i++) {
                    double s = b0;
                    for (int j = 0; j < p; j++) {
                        s += f[j][i];
                    }
                    eta[i] = s;
                    double pr = 1 / (1 + Math.exp(-s));
                    w[i] = Math.max(pr * (1 - pr), 1e-4);
                    z[i] = s + (y[i] - pr) / w[i];
                }
                double wz = 0;
                double ws = 0;
                for (int i = 0; i < n; i++) {
                    double s = z[i];
                    for (int j = 0; j < p; j++) {
                        s -= f[j][i];
                    }
                    wz += w[i] * s;
                    ws += w[i];
                }
                double nb0 = wz / ws;
                double change = Math.abs(nb0 - b0);
                b0 = nb0;
                double[] before = eta.clone();
                backfit(cols, z, w, b0, f, sm, centres, lambdas, segments, 5);
                for (int i = 0; i < n; i++) {
                    double s = b0;
                    for (int j = 0; j < p; j++) {
                        s += f[j][i];
                    }
                    change = Math.max(change, Math.abs(s - before[i]));
                }
                if (change < 1e-5) {
                    break;
                }
            }
        }
        return new AdditiveModel(family, b0, sm, centres);
    }

    private static void backfit(double[][] cols, double[] y, double[] w, double b0, double[][] f,
            PenalizedSplines[] sm, double[] centres, double[] lambdas, int segments, int sweeps) {
        int p = cols.length;
        int n = y.length;
        double[] partial = new double[n];
        for (int sweep = 0; sweep < sweeps; sweep++) {
            double change = 0;
            for (int j = 0; j < p; j++) {
                for (int i = 0; i < n; i++) {
                    double s = y[i] - b0;
                    for (int k = 0; k < p; k++) {
                        if (k != j) {
                            s -= f[k][i];
                        }
                    }
                    partial[i] = s;
                }
                PenalizedSplines s = PenalizedSplines.fit(cols[j], partial, w, segments, lambdas[j]);
                lambdas[j] = s.lambda();
                double c = 0;
                double ws = 0;
                double[] nf = new double[n];
                for (int i = 0; i < n; i++) {
                    nf[i] = s.predict(cols[j][i]);
                    double wi = w == null ? 1 : w[i];
                    c += wi * nf[i];
                    ws += wi;
                }
                c /= ws;
                for (int i = 0; i < n; i++) {
                    nf[i] -= c;
                    change = Math.max(change, Math.abs(nf[i] - f[j][i]));
                }
                f[j] = nf;
                sm[j] = s;
                centres[j] = c;
            }
            if (change < 1e-6) {
                break;
            }
        }
    }

    private static double mean(double[] v) {
        double s = 0;
        for (double x : v) {
            s += x;
        }
        return s / v.length;
    }

    /**
     * The centred component of one feature.
     *
     * @param feature the feature
     * @param value its value
     * @return {@code f_j(value)}
     */
    public double shape(int feature, double value) {
        return smooths[feature].predict(value) - centres[feature];
    }

    /**
     * The linear predictor.
     *
     * @param row a row
     * @return {@code b0 + sum f_j}
     */
    public double raw(double[] row) {
        double s = intercept;
        for (int j = 0; j < smooths.length; j++) {
            s += shape(j, row[j]);
        }
        return s;
    }

    /**
     * Predicts the mean (a probability for logistic GAMs).
     *
     * @param row a row
     * @return the prediction
     */
    public double predict(double[] row) {
        double r = raw(row);
        return family == Family.GAUSSIAN ? r : 1 / (1 + Math.exp(-r));
    }
}
