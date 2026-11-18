package io.github.zzuegg.brainiac.experimental.data;

/**
 * Benchmark time series from chaotic dynamical systems, the standard tasks of reservoir
 * computing: the Lorenz-63 attractor (Lorenz, 1963), integrated with fourth-order
 * Runge-Kutta, and the Mackey-Glass delay equation (Mackey and Glass, 1977) as used by
 * Jaeger and Haas, <a href="https://doi.org/10.1126/science.1091277">Science 2004</a>, and
 * Lorenz-96 (Lorenz, 1996), which wraps {@link io.github.zzuegg.brainiac.experimental.sciml.MechanicalSystems#lorenz96} from sciml.
 */
public final class ChaoticSeries {
    private ChaoticSeries() {
    }

    /**
     * Integrates Lorenz-96 {@code dx_i/dt = (x_i+1 - x_i-2) x_i-1 - x_i + F} with RK4 (substeps
     * of at most 0.01), starting from {@code x = F} with {@code x_0} nudged by 0.01 and dropping a
     * transient of 20 time units, then adds Gaussian observation noise.
     *
     * @param dims the number of sites {@code n}
     * @param length the number of samples
     * @param forcing {@code F}, 8 for the classic chaotic case
     * @param dt the time between samples
     * @param noise observation noise as a share of the series' standard deviation (0 for none)
     * @param seed the noise seed
     * @return {@code [length][dims]} states
     */
    public static double[][] lorenz96(int dims, int length, double forcing, double dt, double noise, long seed) {
        io.github.zzuegg.brainiac.experimental.sciml.MechanicalSystems.Ode ode =
                io.github.zzuegg.brainiac.experimental.sciml.MechanicalSystems.lorenz96(dims, forcing);
        int sub = Math.max(1, (int) Math.ceil(dt / 0.01));
        double h = dt / sub;
        double[] x = new double[dims];
        java.util.Arrays.fill(x, forcing);
        x[0] += 0.01;
        int burnIn = (int) Math.ceil(20 / h);
        for (int i = 0; i < burnIn; i++) {
            x = io.github.zzuegg.brainiac.experimental.sciml.MechanicalSystems.rk4(ode::derivative, x, h);
        }
        double[][] out = new double[length][];
        for (int t = 0; t < length; t++) {
            out[t] = x.clone();
            for (int s = 0; s < sub; s++) {
                x = io.github.zzuegg.brainiac.experimental.sciml.MechanicalSystems.rk4(ode::derivative, x, h);
            }
        }
        if (noise > 0) {
            double m = 0;
            double v = 0;
            for (double[] row : out) {
                for (double a : row) {
                    m += a;
                    v += a * a;
                }
            }
            int n = length * dims;
            double sd = Math.sqrt(Math.max(0, v / n - (m / n) * (m / n)));
            java.util.SplittableRandom r = new java.util.SplittableRandom(seed);
            for (double[] row : out) {
                for (int j = 0; j < dims; j++) {
                    row[j] += noise * sd * r.nextGaussian();
                }
            }
        }
        return out;
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
