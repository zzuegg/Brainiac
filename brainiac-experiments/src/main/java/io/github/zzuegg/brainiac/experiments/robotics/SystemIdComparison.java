package io.github.zzuegg.brainiac.experiments.robotics;

import io.github.zzuegg.brainiac.experimental.geometry.GeometryLinalg;
import io.github.zzuegg.brainiac.experimental.geometry.SystemIdentification;
import io.github.zzuegg.brainiac.experimental.sciml.DynamicModeDecomposition;
import io.github.zzuegg.brainiac.experimental.sciml.Sindy;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * W12 12.3.17: linear system identification from a PRBS experiment on a two-mass
 * spring-damper chain (force on the first mass, position of the second measured; four states)
 * and a DC motor (voltage in, speed out; two states), both discretised exactly (zero-order
 * hold, matrix exponential). Output noise is white (10% of the output's standard deviation) or
 * coloured (the same power through {@code 1 / (1 - 0.9 q^-1)}). ARX, high-order ARX, ARMAX,
 * output error and N4SID see only the input and output; the baselines see the full noisy
 * state: DMDc (least squares {@code x' = A x + B u}) and SINDYc-style sparse regression of
 * the same map (STLSQ). Reported on fresh validation data: simulation fit (percent, against the noise-free output), one-step
 * RMSE against the noisy output, the mean distance from the true poles to the nearest estimated pole, and the order.
 */
public final class SystemIdComparison {

    /**
     * One method on one system and noise.
     *
     * @param system the system
     * @param noise the noise type
     * @param method the method
     * @param fit the validation simulation fit in percent
     * @param oneStep the validation one-step-ahead RMSE
     * @param poleError the pole error
     * @param order the model order
     */
    public record Row(String system, String noise, String method, double fit, double oneStep, double poleError,
            int order) {
    }

    private SystemIdComparison() {
    }

    /** A discrete plant with its true poles. */
    private record Plant(String name, double[][] a, double[] b, int output, double[][] poles) {
    }

    private static double[][] expm(double[][] m) {
        int n = m.length;
        double norm = 0;
        for (double[] r : m) {
            for (double v : r) {
                norm = Math.max(norm, Math.abs(v));
            }
        }
        int squarings = Math.max(0, (int) Math.ceil(Math.log(norm * n + 1e-300) / Math.log(2)) + 2);
        double scale = Math.pow(2, -squarings);
        double[][] a = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                a[i][j] = m[i][j] * scale;
            }
        }
        double[][] result = GeometryLinalg.identity(n);
        double[][] term = GeometryLinalg.identity(n);
        for (int k = 1; k <= 20; k++) {
            term = GeometryLinalg.multiply(term, a);
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    term[i][j] /= k;
                    result[i][j] += term[i][j];
                }
            }
        }
        for (int s = 0; s < squarings; s++) {
            result = GeometryLinalg.multiply(result, result);
        }
        return result;
    }

    private static Plant discretise(String name, double[][] a, double[] b, int output, double dt) {
        int n = a.length;
        double[][] m = new double[n + 1][n + 1];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                m[i][j] = a[i][j] * dt;
            }
            m[i][n] = b[i] * dt;
        }
        double[][] e = expm(m);
        double[][] ad = new double[n][n];
        double[] bd = new double[n];
        for (int i = 0; i < n; i++) {
            System.arraycopy(e[i], 0, ad[i], 0, n);
            bd[i] = e[i][n];
        }
        return new Plant(name, ad, bd, output, DynamicModeDecomposition.eigenvalues(ad));
    }

    private static List<Plant> plants() {
        double k = 1;
        double c = 0.2;
        double[][] chain = {
            {0, 0, 1, 0},
            {0, 0, 0, 1},
            {-2 * k, k, -c, 0},
            {k, -k, 0, -c}};
        double[][] motor = {{-10, 1}, {-0.02, -2}};
        return List.of(discretise("spring chain", chain, new double[] {0, 0, 1, 0}, 1, 0.3),
            discretise("DC motor", motor, new double[] {0, 2}, 0, 0.05));
    }

    private static double[] prbs(int n, double switchProbability, SplittableRandom r) {
        double[] u = new double[n];
        double v = 1;
        for (int t = 0; t < n; t++) {
            if (r.nextDouble() < switchProbability) {
                v = -v;
            }
            u[t] = v;
        }
        return u;
    }

    private static double[][] states(Plant p, double[] u) {
        int n = p.b().length;
        double[][] x = new double[u.length + 1][n];
        for (int t = 0; t < u.length; t++) {
            x[t + 1] = GeometryLinalg.multiply(p.a(), x[t]);
            for (int i = 0; i < n; i++) {
                x[t + 1][i] += p.b()[i] * u[t];
            }
        }
        return x;
    }

    private static double std(double[] v) {
        double m = 0;
        for (double x : v) {
            m += x;
        }
        m /= v.length;
        double s = 0;
        for (double x : v) {
            s += (x - m) * (x - m);
        }
        return Math.sqrt(s / v.length);
    }

    private static double[] noise(int n, boolean coloured, double sigma, SplittableRandom r) {
        double[] e = new double[n];
        double prev = 0;
        double gain = coloured ? Math.sqrt(1 - 0.81) : 1;
        for (int t = 0; t < n; t++) {
            double w = sigma * gain * r.nextGaussian();
            prev = coloured ? 0.9 * prev + w : w;
            e[t] = prev;
        }
        return e;
    }

    private static double rmse(double[] a, double[] b) {
        double s = 0;
        for (int t = 0; t < a.length; t++) {
            s += (a[t] - b[t]) * (a[t] - b[t]);
        }
        return Math.sqrt(s / a.length);
    }

    /**
     * Runs every method on both plants and both noise types.
     *
     * @param samples training and validation length
     * @param seed the seed
     * @return the rows
     */
    public static List<Row> run(int samples, long seed) {
        List<Row> rows = new ArrayList<>();
        SplittableRandom r = new SplittableRandom(seed);
        for (Plant p : plants()) {
            int n = p.b().length;
            double switchP = p.name().startsWith("DC") ? 0.05 : 0.15;
            for (boolean coloured : new boolean[] {false, true}) {
                String noise = coloured ? "coloured" : "white";
                double[] u = prbs(samples, switchP, r);
                double[] uv = prbs(samples, switchP, r);
                double[][] x = states(p, u);
                double[][] xv = states(p, uv);
                double[] y = new double[samples];
                double[] yv = new double[samples];
                for (int t = 0; t < samples; t++) {
                    y[t] = x[t][p.output()];
                    yv[t] = xv[t][p.output()];
                }
                double sigma = 0.1 * std(y);
                double[] e = noise(samples, coloured, sigma, r);
                double[] ev = noise(samples, coloured, sigma, r);
                double[] ym = new double[samples];
                double[] yvm = new double[samples];
                for (int t = 0; t < samples; t++) {
                    ym[t] = y[t] + e[t];
                    yvm[t] = yv[t] + ev[t];
                }
                SystemIdentification.Polynomial arx = SystemIdentification.arx(u, ym, n, n, 1);
                rows.add(row(p, noise, "ARX", arx, uv, yv, yvm, n));
                SystemIdentification.Polynomial arxHigh = SystemIdentification.arx(u, ym, 3 * n, 3 * n, 1);
                rows.add(row(p, noise, "ARX x3 order", arxHigh, uv, yv, yvm, 3 * n));
                SystemIdentification.Polynomial armax = SystemIdentification.armax(u, ym, n, n, n, 1, 15);
                rows.add(row(p, noise, "ARMAX", armax, uv, yv, yvm, n));
                SystemIdentification.Polynomial oe = SystemIdentification.outputError(u, ym, n, n, 1, 40);
                rows.add(row(p, noise, "OE", oe, uv, yv, yvm, n));
                SystemIdentification.StateSpace ss = SystemIdentification.n4sid(u, ym, 3 * n, 0, 2 * n);
                double[] sim = ss.simulate(uv);
                rows.add(new Row(p.name(), noise, "N4SID", SystemIdentification.fit(yv, sim), rmse(yvm, sim),
                    SystemIdentification.poleError(p.poles(), ss.poles()), ss.b().length));
                double[] sx = new double[n];
                double[] sxv = new double[n];
                for (int i = 0; i < n; i++) {
                    sx[i] = 0.1 * std(column(x, i));
                    sxv[i] = 0.1 * std(column(xv, i));
                }
                double[][] xm = new double[samples + 1][n];
                for (int t = 0; t <= samples; t++) {
                    for (int i = 0; i < n; i++) {
                        xm[t][i] = x[t][i] + sx[i] * r.nextGaussian();
                    }
                }
                for (boolean sparse : new boolean[] {false, true}) {
                    double[][] theta = new double[samples][n + 1];
                    double[][] next = new double[samples][];
                    for (int t = 0; t < samples; t++) {
                        System.arraycopy(xm[t], 0, theta[t], 0, n);
                        theta[t][n] = u[t];
                        next[t] = xm[t + 1];
                    }
                    double[][] coef = sparse ? Sindy.stlsq(theta, next, 0.02, 1e-9, 10)
                        : Sindy.lstsq(theta, next, 1e-9);
                    double[][] a = new double[n][n];
                    double[] b = new double[n];
                    for (int i = 0; i < n; i++) {
                        for (int j = 0; j < n; j++) {
                            a[i][j] = coef[j][i];
                        }
                        b[i] = coef[n][i];
                    }
                    double[] c = new double[n];
                    c[p.output()] = 1;
                    SystemIdentification.StateSpace m = new SystemIdentification.StateSpace(a, b, c, 0,
                        new double[0]);
                    double[] s = m.simulate(uv);
                    double[] one = new double[samples];
                    for (int t = 0; t < samples; t++) {
                        double v = b[p.output()] * (t > 0 ? uv[t - 1] : 0);
                        if (t > 0) {
                            for (int j = 0; j < n; j++) {
                                v += a[p.output()][j] * (xv[t - 1][j] + sxv[j] * r.nextGaussian());
                            }
                        }
                        one[t] = v;
                    }
                    rows.add(new Row(p.name(), noise, sparse ? "SINDYc (state)" : "DMDc (state)",
                        SystemIdentification.fit(yv, s), rmse(yvm, one),
                        SystemIdentification.poleError(p.poles(), m.poles()), n));
                }
            }
        }
        return rows;
    }

    private static double[] column(double[][] x, int i) {
        double[] c = new double[x.length];
        for (int t = 0; t < x.length; t++) {
            c[t] = x[t][i];
        }
        return c;
    }

    private static Row row(Plant p, String noise, String method, SystemIdentification.Polynomial m, double[] uv,
            double[] yv, double[] yvm, int order) {
        return new Row(p.name(), noise, method, SystemIdentification.fit(yv, m.simulate(uv)),
            rmse(yvm, m.predict(uv, yvm)), SystemIdentification.poleError(p.poles(), m.poles()), order);
    }
}
