package io.github.zzuegg.brainiac.experimental.sciml;

import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Linalg;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.SplittableRandom;
import java.util.function.UnaryOperator;

/**
 * The stochastic ensemble Kalman filter (Evensen, 1994; perturbed observations after Burgers,
 * van Leeuwen and Evensen, 1998). An ensemble is propagated through a nonlinear model, and at
 * each observation every member is updated with a Kalman gain estimated from the ensemble
 * covariance,
 *
 * <pre>
 * K = P H^T (H P H^T + R)^-1,   x_i += K (y + eps_i - H x_i),   eps_i ~ N(0, R)
 * </pre>
 *
 * <p>so it needs no Jacobians and scales to high dimensions where the extended Kalman filter
 * and the particle filter do not. Two standard fixes for small ensembles are built in:
 * multiplicative inflation of the forecast spread, and covariance localisation by the
 * Gaspari-Cohn taper of the (optionally periodic) distance between state indices, as for
 * Lorenz-96. Observations are a subset of state components with independent Gaussian noise.
 *
 * <p>Paper: <a href="https://doi.org/10.1029/94JC00572">Evensen, JGR 1994</a>.
 */
public final class EnsembleKalmanFilter {

    /**
     * Filter output.
     *
     * @param means the analysis mean at each time {@code [T][n]}
     * @param spreads the ensemble spread (RMS standard deviation) at each time
     */
    public record Estimate(double[][] means, double[] spreads) {
    }

    private final UnaryOperator<double[]> model;
    private final int[] observed;
    private final double noise;
    private final double inflation;
    private final double localization;
    private final boolean periodic;

    private EnsembleKalmanFilter(UnaryOperator<double[]> model, int[] observed, double noise, double inflation,
            double localization, boolean periodic) {
        this.model = model;
        this.observed = observed.clone();
        this.noise = noise;
        this.inflation = inflation;
        this.localization = localization;
        this.periodic = periodic;
    }

    /**
     * Creates the filter.
     *
     * @param model one forecast step
     * @param observed the observed state indices
     * @param noise the observation noise standard deviation
     * @param inflation the multiplicative inflation, for example 1.05; 1 for none
     * @param localization the Gaspari-Cohn half-width in index units; 0 for none
     * @param periodic whether index distance wraps around (a ring such as Lorenz-96)
     * @return the filter
     */
    public static EnsembleKalmanFilter of(UnaryOperator<double[]> model, int[] observed, double noise,
            double inflation, double localization, boolean periodic) {
        return new EnsembleKalmanFilter(model, observed, noise, inflation, localization, periodic);
    }

    /**
     * The Gaspari-Cohn fifth-order taper.
     *
     * @param r distance over the half-width
     * @return the correlation weight, 1 at 0 and 0 beyond 2
     */
    public static double gaspariCohn(double r) {
        r = Math.abs(r);
        if (r >= 2) {
            return 0;
        }
        if (r <= 1) {
            return -0.25 * Math.pow(r, 5) + 0.5 * Math.pow(r, 4) + 0.625 * Math.pow(r, 3) - 5.0 / 3 * r * r + 1;
        }
        return Math.pow(r, 5) / 12 - 0.5 * Math.pow(r, 4) + 0.625 * Math.pow(r, 3) + 5.0 / 3 * r * r - 5 * r + 4
                - 2.0 / (3 * r);
    }

    /**
     * Runs the filter; observations at time 0 update the initial ensemble.
     *
     * @param ensemble initial members {@code [members][n]} (copied)
     * @param observations {@code [T][observed.length]}
     * @param seed the seed for observation perturbations
     * @return the estimates
     */
    public Estimate run(double[][] ensemble, double[][] observations, long seed) {
        SplittableRandom random = new SplittableRandom(seed);
        int members = ensemble.length;
        int n = ensemble[0].length;
        int m = observed.length;
        double[][] x = MechanicalSystems.copy(ensemble);
        double[][] means = new double[observations.length][];
        double[] spreads = new double[observations.length];
        double[][] taper = new double[n][m];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                double d = Math.abs(i - observed[j]);
                if (periodic) {
                    d = Math.min(d, n - d);
                }
                taper[i][j] = localization > 0 ? gaspariCohn(d / localization) : 1;
            }
        }
        for (int t = 0; t < observations.length; t++) {
            if (t > 0) {
                for (int k = 0; k < members; k++) {
                    x[k] = model.apply(x[k]);
                }
            }
            double[] mean = mean(x);
            for (int k = 0; k < members; k++) {
                for (int i = 0; i < n; i++) {
                    x[k][i] = mean[i] + inflation * (x[k][i] - mean[i]);
                }
            }
            // P H^T [n][m] and H P H^T [m][m] from anomalies
            double[][] pht = new double[n][m];
            double[][] hph = new double[m][m];
            for (int k = 0; k < members; k++) {
                for (int i = 0; i < n; i++) {
                    double ai = x[k][i] - mean[i];
                    for (int j = 0; j < m; j++) {
                        pht[i][j] += ai * (x[k][observed[j]] - mean[observed[j]]) / (members - 1);
                    }
                }
            }
            for (int a = 0; a < m; a++) {
                for (int b = 0; b < m; b++) {
                    hph[a][b] = pht[observed[a]][b];
                    if (localization > 0) {
                        double d = Math.abs(observed[a] - observed[b]);
                        if (periodic) {
                            d = Math.min(d, n - d);
                        }
                        hph[a][b] *= gaspariCohn(d / localization);
                    }
                }
                hph[a][a] += noise * noise;
            }
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < m; j++) {
                    pht[i][j] *= taper[i][j];
                }
            }
            // innovations for every member: y + eps - H x_k, solved together
            double[] innov = new double[m * members];
            for (int k = 0; k < members; k++) {
                for (int j = 0; j < m; j++) {
                    innov[j * members + k] = observations[t][j] + noise * random.nextGaussian() - x[k][observed[j]];
                }
            }
            double[] flat = new double[m * m];
            for (int a = 0; a < m; a++) {
                System.arraycopy(hph[a], 0, flat, a * m, m);
            }
            double[] sol = Linalg.solve(Tensor.of(Shape.of(m, m), flat, DType.FLOAT64, Device.cpu()),
                    Tensor.of(Shape.of(m, members), innov, DType.FLOAT64, Device.cpu())).toDoubleArray();
            for (int k = 0; k < members; k++) {
                for (int i = 0; i < n; i++) {
                    double s = 0;
                    for (int j = 0; j < m; j++) {
                        s += pht[i][j] * sol[j * members + k];
                    }
                    x[k][i] += s;
                }
            }
            means[t] = mean(x);
            double var = 0;
            for (double[] member : x) {
                for (int i = 0; i < n; i++) {
                    var += (member[i] - means[t][i]) * (member[i] - means[t][i]);
                }
            }
            spreads[t] = Math.sqrt(var / ((members - 1.0) * n));
        }
        return new Estimate(means, spreads);
    }

    private static double[] mean(double[][] x) {
        double[] m = new double[x[0].length];
        for (double[] r : x) {
            for (int i = 0; i < m.length; i++) {
                m[i] += r[i] / x.length;
            }
        }
        return m;
    }
}
