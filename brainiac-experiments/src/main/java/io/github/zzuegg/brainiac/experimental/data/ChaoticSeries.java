package io.github.zzuegg.brainiac.experimental.data;

/**
 * Benchmark time series from chaotic dynamical systems, the standard tasks of reservoir
 * computing: the Lorenz-63 attractor (Lorenz, 1963), integrated with fourth-order
 * Runge-Kutta, and the Mackey-Glass delay equation (Mackey and Glass, 1977) as used by
 * Jaeger and Haas, <a href="https://doi.org/10.1126/science.1091277">Science 2004</a>.
 */
public final class ChaoticSeries {
    private ChaoticSeries() {
    }

    /**
     * Integrates Lorenz-63 with the classic parameters (sigma 10, rho 28, beta 8/3).
     *
     * @param n the number of samples to return
     * @param dt the time between samples
     * @param discard samples to integrate and drop first, so the trajectory is on the attractor
     * @return {@code [n][3]} states (x, y, z)
     */
    public static double[][] lorenz(int n, double dt, int discard) {
        double[] s = {1, 1, 1};
        double[][] out = new double[n][];
        for (int i = -discard; i < n; i++) {
            if (i >= 0) {
                out[i] = s.clone();
            }
            s = rk4(s, dt);
        }
        return out;
    }

    private static double[] lorenzRate(double[] s) {
        return new double[] {10 * (s[1] - s[0]), s[0] * (28 - s[2]) - s[1], s[0] * s[1] - 8.0 / 3 * s[2]};
    }

    private static double[] rk4(double[] s, double h) {
        double[] k1 = lorenzRate(s);
        double[] k2 = lorenzRate(axpy(s, k1, h / 2));
        double[] k3 = lorenzRate(axpy(s, k2, h / 2));
        double[] k4 = lorenzRate(axpy(s, k3, h));
        double[] r = new double[3];
        for (int j = 0; j < 3; j++) {
            r[j] = s[j] + h / 6 * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j]);
        }
        return r;
    }

    private static double[] axpy(double[] s, double[] k, double h) {
        double[] r = new double[3];
        for (int j = 0; j < 3; j++) {
            r[j] = s[j] + h * k[j];
        }
        return r;
    }

    /**
     * Integrates {@code dx/dt = 0.2 x(t - tau) / (1 + x(t - tau)^10) - 0.1 x(t)} with Euler
     * substeps of 0.1 and samples it every {@code sampleEvery} time units.
     *
     * @param n the number of samples
     * @param tau the delay, 17 for the classic mildly chaotic case
     * @param sampleEvery time units between samples
     * @param discard samples to drop first
     * @return the series
     */
    public static double[] mackeyGlass(int n, double tau, double sampleEvery, int discard) {
        double h = 0.1;
        int lag = (int) Math.round(tau / h);
        int every = Math.max(1, (int) Math.round(sampleEvery / h));
        int total = (n + discard) * every;
        double[] x = new double[total + lag + 1];
        java.util.Arrays.fill(x, 0, lag + 1, 1.2);
        for (int i = lag; i < x.length - 1; i++) {
            double d = x[i - lag];
            x[i + 1] = x[i] + h * (0.2 * d / (1 + Math.pow(d, 10)) - 0.1 * x[i]);
        }
        double[] out = new double[n];
        for (int i = 0; i < n; i++) {
            out[i] = x[lag + (discard + i) * every];
        }
        return out;
    }
}
