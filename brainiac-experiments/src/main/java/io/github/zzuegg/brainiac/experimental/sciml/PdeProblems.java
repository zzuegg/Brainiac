package io.github.zzuegg.brainiac.experimental.sciml;

import io.github.zzuegg.brainiac.tensor.spi.Fft;

import java.util.SplittableRandom;

/**
 * Reference solvers and random data for one- and two-dimensional PDEs: the ground truth for
 * physics-informed networks and the train/test pairs for neural operators.
 *
 * <p>Periodic problems are solved pseudo-spectrally with an integrating-factor fourth-order
 * Runge-Kutta scheme ({@code u_t = L u + N(u)} with the stiff linear part {@code L} diagonal
 * in Fourier space and applied exactly, the nonlinear part evaluated on the grid with 2/3-rule
 * dealiasing), using the core FFT:
 *
 * <ul>
 *   <li>{@link Equation#HEAT}: {@code u_t = nu u_xx};
 *   <li>{@link Equation#ADVECTION}: {@code u_t + c u_x = 0} (exact phase shift);
 *   <li>{@link Equation#BURGERS}: {@code u_t + u u_x = nu u_xx};
 *   <li>{@link Equation#ALLEN_CAHN}: {@code u_t = eps u_xx + 5 u - 5 u^3};
 *   <li>{@link Equation#KDV}: {@code u_t + u u_x + c u_xxx = 0};
 *   <li>{@link Equation#KURAMOTO_SIVASHINSKY}: {@code u_t + u u_x + u_xx + u_xxxx = 0}.
 * </ul>
 *
 * <p>Boundary-value problems use finite differences: 1-D Poisson {@code -u'' = f} and Darcy
 * flow {@code -(a u')' = f} (tridiagonal solves) and 2-D Poisson {@code -lap u = f}
 * (successive over-relaxation), all with zero Dirichlet boundaries on the unit interval or
 * square. {@link #randomField} draws smooth random periodic functions (a truncated Fourier
 * series with decaying amplitudes, a cheap Gaussian random field) for initial conditions and
 * coefficient fields.
 */
public final class PdeProblems {

    private PdeProblems() {
    }

    /** The periodic evolution equations {@link #solve} handles. */
    public enum Equation {
        /** {@code u_t = nu u_xx}; the coefficient is {@code nu}. */
        HEAT,
        /** {@code u_t + c u_x = 0}; the coefficient is {@code c}. */
        ADVECTION,
        /** {@code u_t + u u_x = nu u_xx}; the coefficient is {@code nu}. */
        BURGERS,
        /** {@code u_t = eps u_xx + 5 u - 5 u^3}; the coefficient is {@code eps}. */
        ALLEN_CAHN,
        /** {@code u_t + u u_x + c u_xxx = 0}; the coefficient is {@code c}. */
        KDV,
        /** {@code u_t + u u_x + u_xx + u_xxxx = 0}; the coefficient is unused. */
        KURAMOTO_SIVASHINSKY
    }

    /**
     * A solution sampled on a uniform periodic grid at increasing times.
     *
     * @param origin the left end of the domain
     * @param length the period
     * @param times the snapshot times
     * @param u the solution {@code [times][points]}
     */
    public record Solution(double origin, double length, double[] times, double[][] u) {

        /** @return the grid points per snapshot */
        public int points() {
            return u[0].length;
        }

        /**
         * @param j a grid index
         * @return its coordinate
         */
        public double x(int j) {
            return origin + j * length / points();
        }

        /** @return the last snapshot */
        public double[] last() {
            return u[u.length - 1];
        }

        /**
         * The solution at any point, by linear interpolation in time and periodic linear
         * interpolation in space.
         *
         * @param x the coordinate
         * @param t the time, within the snapshot times
         * @return the interpolated value
         */
        public double at(double x, double t) {
            int nt = times.length;
            double ft;
            int i0;
            if (nt == 1 || t <= times[0]) {
                i0 = 0;
                ft = 0;
            } else if (t >= times[nt - 1]) {
                i0 = nt - 2;
                ft = 1;
            } else {
                double dt = times[1] - times[0];
                i0 = Math.min(nt - 2, (int) ((t - times[0]) / dt));
                ft = (t - times[i0]) / dt;
            }
            double v0 = interpolate(u[i0], x);
            return nt == 1 ? v0 : v0 * (1 - ft) + interpolate(u[i0 + 1], x) * ft;
        }

        private double interpolate(double[] row, double x) {
            int n = row.length;
            double s = (x - origin) / length * n;
            s = ((s % n) + n) % n;
            int j = (int) Math.floor(s);
            double f = s - j;
            return row[j % n] * (1 - f) + row[(j + 1) % n] * f;
        }
    }

    /**
     * The wave numbers {@code 2 pi k / length} of an {@code n}-point periodic grid in FFT
     * order.
     *
     * @param n the points
     * @param length the period
     * @return the wave numbers
     */
    public static double[] waveNumbers(int n, double length) {
        double[] k = new double[n];
        for (int j = 0; j < n; j++) {
            int m = j <= n / 2 ? j : j - n;
            if (j == n / 2 && n % 2 == 0) {
                m = 0;   // Nyquist: drop it so odd derivatives stay real
            }
            k[j] = 2 * Math.PI * m / length;
        }
        return k;
    }

    /**
     * The forward FFT of a real row, as interleaved (real, imaginary) pairs.
     *
     * @param u the real values
     * @return the spectrum
     */
    public static double[] spectrum(double[] u) {
        int n = u.length;
        float[] d = new float[2 * n];
        for (int j = 0; j < n; j++) {
            d[2 * j] = (float) u[j];
        }
        Fft.transform(d, 1, n, false);
        double[] out = new double[2 * n];
        for (int j = 0; j < 2 * n; j++) {
            out[j] = d[j];
        }
        return out;
    }

    /**
     * The real part of the inverse FFT of interleaved pairs.
     *
     * @param spec the spectrum
     * @return the real values
     */
    public static double[] real(double[] spec) {
        int n = spec.length / 2;
        float[] d = new float[2 * n];
        for (int j = 0; j < 2 * n; j++) {
            d[j] = (float) spec[j];
        }
        Fft.transform(d, 1, n, true);
        double[] out = new double[n];
        for (int j = 0; j < n; j++) {
            out[j] = d[2 * j];
        }
        return out;
    }

    /**
     * The spectral derivative of a periodic function.
     *
     * @param u the values on a uniform periodic grid
     * @param length the period
     * @param order the derivative order
     * @return {@code d^order u / dx^order}
     */
    public static double[] derivative(double[] u, double length, int order) {
        int n = u.length;
        double[] k = waveNumbers(n, length);
        double[] s = spectrum(u);
        for (int j = 0; j < n; j++) {
            // (i k)^order
            double re = 1;
            double im = 0;
            for (int o = 0; o < order; o++) {
                double r2 = -im * k[j];
                double i2 = re * k[j];
                re = r2;
                im = i2;
            }
            double a = s[2 * j];
            double b = s[2 * j + 1];
            s[2 * j] = a * re - b * im;
            s[2 * j + 1] = a * im + b * re;
        }
        return real(s);
    }

    /**
     * A smooth random periodic function: {@code sum_k (a_k cos(2 pi k x) + b_k sin(2 pi k x))
     * / k^decay} with standard normal {@code a_k, b_k}, scaled to unit maximum.
     *
     * @param n the grid points
     * @param modes the highest mode
     * @param decay the amplitude decay exponent, for example 2
     * @param random the random source
     * @return the values
     */
    public static double[] randomField(int n, int modes, double decay, SplittableRandom random) {
        double[] u = new double[n];
        for (int k = 1; k <= modes; k++) {
            double a = random.nextGaussian() / Math.pow(k, decay);
            double b = random.nextGaussian() / Math.pow(k, decay);
            for (int j = 0; j < n; j++) {
                double phase = 2 * Math.PI * k * j / n;
                u[j] += a * Math.cos(phase) + b * Math.sin(phase);
            }
        }
        double max = 1e-12;
        for (double v : u) {
            max = Math.max(max, Math.abs(v));
        }
        for (int j = 0; j < n; j++) {
            u[j] /= max;
        }
        return u;
    }

    /**
     * Solves a periodic evolution equation.
     *
     * @param equation the equation
     * @param u0 the initial condition on a uniform grid of {@code [origin, origin + length)}
     * @param origin the left end of the domain
     * @param length the period
     * @param coefficient the equation's coefficient (see {@link Equation})
     * @param time the final time
     * @param snapshots the snapshots kept, at least 2: times {@code 0, time/(snapshots-1), ...}
     * @param maxStep the largest time step; the solver also limits it by an advective CFL bound
     * @return the solution
     */
    public static Solution solve(Equation equation, double[] u0, double origin, double length, double coefficient,
            double time, int snapshots, double maxStep) {
        int n = u0.length;
        double[] k = waveNumbers(n, length);
        double[] lr = new double[n];
        double[] li = new double[n];
        for (int j = 0; j < n; j++) {
            double k2 = k[j] * k[j];
            switch (equation) {
                case HEAT, BURGERS -> lr[j] = -coefficient * k2;
                case ADVECTION -> li[j] = -coefficient * k[j];
                case ALLEN_CAHN -> lr[j] = -coefficient * k2;
                case KDV -> li[j] = coefficient * k2 * k[j];
                case KURAMOTO_SIVASHINSKY -> lr[j] = k2 - k2 * k2;
                default -> throw new IllegalStateException();
            }
        }
        boolean[] keep = new boolean[n];
        for (int j = 0; j < n; j++) {
            int m = j <= n / 2 ? j : n - j;
            keep[j] = 3 * m < n;
        }
        double[][] out = new double[snapshots][];
        double[] times = new double[snapshots];
        out[0] = u0.clone();
        double[] s = spectrum(u0);
        double interval = time / (snapshots - 1);
        double dx = length / n;
        for (int snap = 1; snap < snapshots; snap++) {
            times[snap] = snap * interval;
            double max = 1e-9;
            for (double v : out[snap - 1]) {
                max = Math.max(max, Math.abs(v));
            }
            double limit = equation == Equation.HEAT || equation == Equation.ADVECTION ? interval
                    : Math.min(maxStep, 0.4 * dx / max);
            int steps = Math.max(1, (int) Math.ceil(interval / limit));
            double dt = interval / steps;
            double[][] e = factors(lr, li, dt);
            double[][] e2 = factors(lr, li, dt / 2);
            for (int step = 0; step < steps; step++) {
                s = equation == Equation.HEAT || equation == Equation.ADVECTION ? mul(e, s)
                        : rk4(s, e, e2, k, keep, equation, dt);
            }
            out[snap] = real(s);
        }
        return new Solution(origin, length, times, out);
    }

    private static double[][] factors(double[] lr, double[] li, double dt) {
        int n = lr.length;
        double[][] e = new double[2][n];
        for (int j = 0; j < n; j++) {
            double m = Math.exp(lr[j] * dt);
            e[0][j] = m * Math.cos(li[j] * dt);
            e[1][j] = m * Math.sin(li[j] * dt);
        }
        return e;
    }

    private static double[] mul(double[][] e, double[] s) {
        int n = e[0].length;
        double[] out = new double[2 * n];
        for (int j = 0; j < n; j++) {
            double a = s[2 * j];
            double b = s[2 * j + 1];
            out[2 * j] = a * e[0][j] - b * e[1][j];
            out[2 * j + 1] = a * e[1][j] + b * e[0][j];
        }
        return out;
    }

    private static double[] axpy(double[] x, double a, double[] y) {
        double[] out = new double[x.length];
        for (int j = 0; j < x.length; j++) {
            out[j] = x[j] + a * y[j];
        }
        return out;
    }

    private static double[] nonlinear(double[] s, double[] k, boolean[] keep, Equation equation, double dt) {
        int n = k.length;
        double[] u = real(s);
        double[] out;
        if (equation == Equation.ALLEN_CAHN) {
            double[] f = new double[n];
            for (int j = 0; j < n; j++) {
                f[j] = 5 * u[j] - 5 * u[j] * u[j] * u[j];
            }
            out = spectrum(f);
        } else {
            double[] sq = new double[n];
            for (int j = 0; j < n; j++) {
                sq[j] = u[j] * u[j];
            }
            double[] f = spectrum(sq);
            out = new double[2 * n];
            for (int j = 0; j < n; j++) {
                // -0.5 i k F(u^2)
                out[2 * j] = 0.5 * k[j] * f[2 * j + 1];
                out[2 * j + 1] = -0.5 * k[j] * f[2 * j];
            }
        }
        for (int j = 0; j < n; j++) {
            if (!keep[j]) {
                out[2 * j] = 0;
                out[2 * j + 1] = 0;
            }
            out[2 * j] *= dt;
            out[2 * j + 1] *= dt;
        }
        return out;
    }

    private static double[] rk4(double[] s, double[][] e, double[][] e2, double[] k, boolean[] keep, Equation eq,
            double dt) {
        double[] k1 = nonlinear(s, k, keep, eq, dt);
        double[] k2 = nonlinear(mul(e2, axpy(s, 0.5, k1)), k, keep, eq, dt);
        double[] k3 = nonlinear(axpy(mul(e2, s), 0.5, k2), k, keep, eq, dt);
        double[] k4 = nonlinear(axpy(mul(e, s), 1, mul(e2, k3)), k, keep, eq, dt);
        double[] sum = axpy(mul(e, k1), 2, mul(e2, axpy(k2, 1, k3)));
        sum = axpy(sum, 1, k4);
        return axpy(mul(e, s), 1.0 / 6, sum);
    }

    /**
     * Solves {@code -u'' = f} on {@code [0, 1]} with {@code u(0) = u(1) = 0} by second-order
     * finite differences.
     *
     * @param f the source at the {@code n} interior points {@code x_i = (i + 1) / (n + 1)}
     * @return {@code u} at the interior points
     */
    public static double[] poisson1d(double[] f) {
        double[] a = new double[f.length + 1];
        java.util.Arrays.fill(a, 1);
        return darcy1d(a, f);
    }

    /**
     * Solves the 1-D Darcy problem {@code -(a u')' = f} on {@code [0, 1]} with zero boundary
     * values, by finite differences with the coefficient at cell faces.
     *
     * @param a the coefficient at the {@code n + 1} faces {@code (i + 1/2) / (n + 1)}
     * @param f the source at the {@code n} interior points
     * @return {@code u} at the interior points
     */
    public static double[] darcy1d(double[] a, double[] f) {
        int n = f.length;
        if (a.length != n + 1) {
            throw new IllegalArgumentException("need n + 1 face coefficients");
        }
        double h2 = 1.0 / ((n + 1.0) * (n + 1.0));
        double[] diag = new double[n];
        double[] off = new double[n];
        double[] rhs = new double[n];
        for (int i = 0; i < n; i++) {
            diag[i] = (a[i] + a[i + 1]) / h2;
            off[i] = -a[i + 1] / h2;
            rhs[i] = f[i];
        }
        // Thomas algorithm on the symmetric tridiagonal system
        double[] c = new double[n];
        double[] d = new double[n];
        c[0] = off[0] / diag[0];
        d[0] = rhs[0] / diag[0];
        for (int i = 1; i < n; i++) {
            double m = diag[i] - off[i - 1] * c[i - 1];
            c[i] = off[i] / m;
            d[i] = (rhs[i] - off[i - 1] * d[i - 1]) / m;
        }
        double[] u = new double[n];
        u[n - 1] = d[n - 1];
        for (int i = n - 2; i >= 0; i--) {
            u[i] = d[i] - c[i] * u[i + 1];
        }
        return u;
    }

    /**
     * Solves {@code -lap u = f} on the unit square with zero boundary values by successive
     * over-relaxation on the five-point stencil.
     *
     * @param f the source at the {@code n x n} interior points {@code ((i+1)h, (j+1)h)},
     *     {@code h = 1/(n+1)}
     * @param tolerance the largest update at which to stop
     * @return {@code u} at the interior points
     */
    public static double[][] poisson2d(double[][] f, double tolerance) {
        int n = f.length;
        double h2 = 1.0 / ((n + 1.0) * (n + 1.0));
        double omega = 2 / (1 + Math.sin(Math.PI / (n + 1)));
        double[][] u = new double[n + 2][n + 2];
        for (int it = 0; it < 100_000; it++) {
            double change = 0;
            for (int i = 1; i <= n; i++) {
                for (int j = 1; j <= n; j++) {
                    double gs = 0.25 * (u[i - 1][j] + u[i + 1][j] + u[i][j - 1] + u[i][j + 1] + h2 * f[i - 1][j - 1]);
                    double d = omega * (gs - u[i][j]);
                    u[i][j] += d;
                    change = Math.max(change, Math.abs(d));
                }
            }
            if (change < tolerance) {
                break;
            }
        }
        double[][] out = new double[n][n];
        for (int i = 0; i < n; i++) {
            System.arraycopy(u[i + 1], 1, out[i], 0, n);
        }
        return out;
    }

    /**
     * The antiderivative operator {@code G(u)(y) = int_0^y u(s) ds} on a uniform grid of
     * {@code [0, 1)}, by the trapezoidal rule.
     *
     * @param u values at {@code j / n}
     * @return the integral up to each grid point
     */
    public static double[] antiderivative(double[] u) {
        int n = u.length;
        double h = 1.0 / n;
        double[] out = new double[n];
        for (int j = 1; j < n; j++) {
            out[j] = out[j - 1] + 0.5 * h * (u[j - 1] + u[j]);
        }
        return out;
    }

    /**
     * Operator-learning pairs for a periodic equation: random initial conditions on
     * {@code [0, length)} and the solution at {@code time}.
     *
     * @param equation the equation
     * @param count the number of pairs
     * @param points the grid points
     * @param length the period
     * @param coefficient the equation's coefficient
     * @param time the target time
     * @param modes the highest mode of the random initial conditions
     * @param seed the seed
     * @return {@code [count][2][points]}: input and output functions
     */
    public static double[][][] pairs(Equation equation, int count, int points, double length, double coefficient,
            double time, int modes, long seed) {
        SplittableRandom random = new SplittableRandom(seed);
        double[][][] out = new double[count][2][];
        for (int i = 0; i < count; i++) {
            double[] u0 = randomField(points, modes, 2, random);
            out[i][0] = u0;
            out[i][1] = solve(equation, u0, 0, length, coefficient, time, 2, 0.01).last();
        }
        return out;
    }

    /**
     * The relative L2 error {@code |a - b| / |b|}.
     *
     * @param prediction the prediction
     * @param reference the reference
     * @return the relative error
     */
    public static double relativeL2(double[] prediction, double[] reference) {
        double num = 0;
        double den = 1e-30;
        for (int i = 0; i < reference.length; i++) {
            double d = prediction[i] - reference[i];
            num += d * d;
            den += reference[i] * reference[i];
        }
        return Math.sqrt(num / den);
    }
}
