package io.github.zzuegg.brainiac.experimental.science;

import io.github.zzuegg.brainiac.experimental.stats.PenalizedRegression;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Species distribution models from presence-only records on a raster.
 *
 * <ul>
 *   <li>{@link Method#IPP}: an inhomogeneous Poisson point process,
 *       {@code log lambda(x) = b0 + b f(x)}, fitted by maximising
 *       {@code sum_presences eta(x_i) - sum_cells a lambda(c)}: the Berman–Turner quadrature with
 *       every raster cell as a quadrature point (Warton and Shepherd, AoAS 2010).</li>
 *   <li>{@link Method#MAXENT}: the same likelihood with an l1 penalty on {@code b} and MaxEnt's
 *       feature classes (linear, quadratic, hinge, product), fitted by proximal gradient; Renner
 *       and Warton (Biometrics 2013) showed MaxEnt is exactly this model (Phillips, Anderson and
 *       Schapire, Ecol. Model. 2006).</li>
 *   <li>{@link Method#PRESENCE_BACKGROUND}: logistic regression of presences against random
 *       background cells (wave 12's {@link PenalizedRegression}), which approaches the IPP only
 *       as the number of background points grows.</li>
 * </ul>
 * Sampling bias is corrected by adding a bias covariate (closeness to roads) to the fit and
 * setting it to a constant when predicting.
 */
public final class SpeciesDistribution {

    /** The fitting method. */
    public enum Method {
        /** Poisson point process, no penalty. */
        IPP,
        /** Poisson point process with l1 and MaxEnt features. */
        MAXENT,
        /** Presence vs background logistic regression. */
        PRESENCE_BACKGROUND
    }

    private final double[] mean;
    private final double[] sd;
    private final double intercept;
    private final double[] beta;
    private final boolean rich;
    private final double[][] knots;
    private final int biasColumn;
    private final double biasValue;

    private SpeciesDistribution(double[] mean, double[] sd, double intercept, double[] beta, boolean rich,
            double[][] knots, int biasColumn, double biasValue) {
        this.mean = mean;
        this.sd = sd;
        this.intercept = intercept;
        this.beta = beta;
        this.rich = rich;
        this.knots = knots;
        this.biasColumn = biasColumn;
        this.biasValue = biasValue;
    }

    private static double[][] knotsOf(double[][] cov) {
        int k = cov[0].length;
        double[][] q = new double[k][3];
        for (int j = 0; j < k; j++) {
            double[] col = new double[cov.length];
            for (int i = 0; i < cov.length; i++) {
                col[i] = cov[i][j];
            }
            Arrays.sort(col);
            for (int t = 0; t < 3; t++) {
                q[j][t] = col[(int) ((t + 1) * 0.25 * (col.length - 1))];
            }
        }
        return q;
    }

    /**
     * The feature vector: linear and quadratic terms, plus (when {@code rich}) hinges at the
     * quartiles and pairwise products.
     *
     * @param c raw covariates
     * @param rich whether to add hinge and product features
     * @param knots hinge knots per covariate
     * @return the features
     */
    static double[] features(double[] c, boolean rich, double[][] knots) {
        List<Double> f = new ArrayList<>();
        for (double v : c) {
            f.add(v);
        }
        for (double v : c) {
            f.add(v * v);
        }
        if (rich) {
            for (int j = 0; j < c.length; j++) {
                for (double k : knots[j]) {
                    f.add(Math.max(0, c[j] - k));
                    f.add(Math.max(0, k - c[j]));
                }
            }
            for (int a = 0; a < c.length; a++) {
                for (int b = a + 1; b < c.length; b++) {
                    f.add(c[a] * c[b]);
                }
            }
        }
        double[] out = new double[f.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = f.get(i);
        }
        return out;
    }

    private static double[][] withBias(double[][] cov, double[] bias) {
        if (bias == null) {
            return cov;
        }
        double[][] out = new double[cov.length][];
        for (int i = 0; i < cov.length; i++) {
            out[i] = Arrays.copyOf(cov[i], cov[i].length + 1);
            out[i][cov[i].length] = bias[i];
        }
        return out;
    }

    /**
     * Fits a model.
     *
     * @param method the method
     * @param covariates covariates per cell
     * @param presences presence cells
     * @param bias the bias covariate per cell to correct for (null for none)
     * @param l1 the l1 strength for MAXENT (per presence), or the logistic lambda
     * @param background background points for PRESENCE_BACKGROUND
     * @param seed the background seed
     * @return the model
     */
    public static SpeciesDistribution fit(Method method, double[][] covariates, int[] presences, double[] bias,
            double l1, int background, long seed) {
        double[][] cov = withBias(covariates, bias);
        int cells = cov.length;
        boolean rich = method == Method.MAXENT;
        double[][] knots = knotsOf(cov);
        double[][] f = new double[cells][];
        for (int c = 0; c < cells; c++) {
            f[c] = features(cov[c], rich, knots);
        }
        int p = f[0].length;
        double[] mu = new double[p];
        double[] sd = new double[p];
        for (double[] row : f) {
            for (int j = 0; j < p; j++) {
                mu[j] += row[j] / cells;
            }
        }
        for (double[] row : f) {
            for (int j = 0; j < p; j++) {
                sd[j] += (row[j] - mu[j]) * (row[j] - mu[j]) / cells;
            }
        }
        for (int j = 0; j < p; j++) {
            sd[j] = Math.sqrt(sd[j]) + 1e-12;
        }
        double[][] z = new double[cells][p];
        for (int c = 0; c < cells; c++) {
            for (int j = 0; j < p; j++) {
                z[c][j] = (f[c][j] - mu[j]) / sd[j];
            }
        }
        int biasCol = bias == null ? -1 : covariates[0].length;
        double biasValue = 0;
        if (bias != null) {
            for (double v : bias) {
                biasValue = Math.max(biasValue, v);
            }
        }
        double b0;
        double[] b;
        if (method == Method.PRESENCE_BACKGROUND) {
            SplittableRandom r = new SplittableRandom(seed);
            int n = presences.length + background;
            double[][] x = new double[n][];
            double[] y = new double[n];
            for (int i = 0; i < presences.length; i++) {
                x[i] = z[presences[i]];
                y[i] = 1;
            }
            for (int i = 0; i < background; i++) {
                x[presences.length + i] = z[r.nextInt(cells)];
            }
            PenalizedRegression.Fit fit = PenalizedRegression.of(PenalizedRegression.Penalty.LASSO)
                    .family(PenalizedRegression.Family.LOGISTIC).fit(x, y, Math.max(l1, 1e-6));
            b0 = fit.intercept();
            b = fit.beta();
        } else {
            double[] res = poissonProcess(z, presences, method == Method.MAXENT ? l1 : 0, 500);
            b0 = res[p];
            b = Arrays.copyOf(res, p);
        }
        return new SpeciesDistribution(mu, sd, b0, b, rich, knots, biasCol, biasValue);
    }

    /**
     * Maximises the discretised Poisson-process likelihood with an l1 penalty by proximal
     * gradient ascent with backtracking (the intercept is unpenalised).
     *
     * @param z standardised features per cell
     * @param presences presence cells
     * @param l1 the penalty per presence record
     * @param iterations the iterations
     * @return {beta..., intercept}
     */
    static double[] poissonProcess(double[][] z, int[] presences, double l1, int iterations) {
        int cells = z.length;
        int p = z[0].length;
        int n = presences.length;
        double[] b = new double[p];
        double b0 = Math.log((double) n / cells);
        double step = 0.5;
        double[] grad = new double[p + 1];
        double value = objective(z, presences, b, b0, l1);
        for (int it = 0; it < iterations; it++) {
            Arrays.fill(grad, 0);
            for (int i : presences) {
                for (int j = 0; j < p; j++) {
                    grad[j] += z[i][j] / n;
                }
                grad[p] += 1.0 / n;
            }
            for (int c = 0; c < cells; c++) {
                double eta = b0;
                for (int j = 0; j < p; j++) {
                    eta += b[j] * z[c][j];
                }
                double lam = Math.exp(eta) / n;
                for (int j = 0; j < p; j++) {
                    grad[j] -= lam * z[c][j];
                }
                grad[p] -= lam;
            }
            double[] nb = new double[p];
            double nb0;
            double nv;
            while (true) {
                for (int j = 0; j < p; j++) {
                    nb[j] = PenalizedRegression.soft(b[j] + step * grad[j], step * l1);
                }
                nb0 = b0 + step * grad[p];
                nv = objective(z, presences, nb, nb0, l1);
                if (nv >= value - 1e-12 || step < 1e-8) {
                    break;
                }
                step /= 2;
            }
            double change = Math.abs(nv - value);
            b = nb;
            b0 = nb0;
            value = nv;
            step = Math.min(step * 1.5, 4);
            if (change < 1e-10) {
                break;
            }
        }
        double[] out = Arrays.copyOf(b, p + 1);
        out[p] = b0;
        return out;
    }

    private static double objective(double[][] z, int[] presences, double[] b, double b0, double l1) {
        int n = presences.length;
        double v = 0;
        for (int i : presences) {
            v += eta(z[i], b, b0) / n;
        }
        for (double[] row : z) {
            v -= Math.exp(eta(row, b, b0)) / n;
        }
        for (double x : b) {
            v -= l1 * Math.abs(x);
        }
        return v;
    }

    private static double eta(double[] z, double[] b, double b0) {
        double e = b0;
        for (int j = 0; j < b.length; j++) {
            e += b[j] * z[j];
        }
        return e;
    }

    /**
     * The predicted linear predictor (log relative intensity, or logit for presence–background)
     * for a cell's covariates, with the bias covariate held at its maximum (full survey effort).
     *
     * @param covariates the cell's raw covariates (without the bias column)
     * @return the score
     */
    public double score(double[] covariates) {
        double[] c = covariates;
        if (biasColumn >= 0) {
            c = Arrays.copyOf(covariates, covariates.length + 1);
            c[biasColumn] = biasValue;
        }
        double[] f = features(c, rich, knots);
        double e = intercept;
        for (int j = 0; j < beta.length; j++) {
            e += beta[j] * (f[j] - mean[j]) / sd[j];
        }
        return e;
    }

    /** @return the number of non-zero coefficients */
    public int nonZero() {
        int k = 0;
        for (double v : beta) {
            k += v != 0 ? 1 : 0;
        }
        return k;
    }
}
