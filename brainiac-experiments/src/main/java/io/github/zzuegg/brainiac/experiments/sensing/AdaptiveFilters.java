package io.github.zzuegg.brainiac.experiments.sensing;

import io.github.zzuegg.brainiac.data.LeastSquares;
import io.github.zzuegg.brainiac.experimental.data.ArraySignals;
import io.github.zzuegg.brainiac.experimental.sensing.AdaptiveFilters.Method;
import io.github.zzuegg.brainiac.experimental.sensing.AdaptiveFilters.Settings;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Adaptive filtering (wave 13, item 13.4.13): {@link #systemIdentification} compares LMS, NLMS,
 * affine projection, RLS and the Kalman FIR filter against offline least squares (core
 * {@link LeastSquares#ols}) and an AdaGrad online-gradient learner on white and coloured input
 * with fixed, switching and drifting systems; {@link #echoCancellation} measures echo-return-loss
 * enhancement with and without Geigel double-talk detection.
 */
public final class AdaptiveFilters {

    private AdaptiveFilters() {
    }

    /** Per-method settings: LMS at a stable step for unit-power input, the normalised rules at 0.5. */
    private static Settings settings(Method m) {
        double mu = m == Method.LMS ? 0.01 : 0.5;
        return new Settings(mu, 4, 1e-2, 0.995, 1e-5, 1e-2);
    }

    /**
     * System identification.
     *
     * @param taps the filter length
     * @param n the samples per record
     * @param noise the measurement noise
     * @param seed the seed
     * @return scenario / method to {samples to reach -20 dB misalignment (NaN if never), misalignment dB
     *     over the last 10%, mean misalignment dB over the 500 samples after the change (or the record), ms}
     */
    public static Map<String, double[]> systemIdentification(int taps, int n, double noise, long seed) {
        Map<String, double[]> out = new LinkedHashMap<>();
        Object[][] scenarios = {{"white", 0.0, ArraySignals.Change.NONE}, {"AR(1) 0.9", 0.9, ArraySignals.Change.NONE},
            {"AR(1) 0.9, abrupt change", 0.9, ArraySignals.Change.ABRUPT},
            {"AR(1) 0.9, drift", 0.9, ArraySignals.Change.DRIFT}};
        int s = 0;
        for (Object[] sc : scenarios) {
            ArraySignals.SystemRecord rec = ArraySignals.systemIdentification(taps, n, (Double) sc[1],
                    (ArraySignals.Change) sc[2], noise, new SplittableRandom(seed + s++));
            int from = sc[2] == ArraySignals.Change.ABRUPT ? n / 2 : 0;
            for (Method m : Method.values()) {
                long t0 = System.nanoTime();
                double[] mis = io.github.zzuegg.brainiac.experimental.sensing.AdaptiveFilters.run(m, settings(m),
                        taps, rec.input(), rec.desired(), null, rec.weights())[1];
                out.put(sc[0] + " / " + m, summary(mis, from, (System.nanoTime() - t0) / 1e6));
            }
            long t0 = System.nanoTime();
            out.put(sc[0] + " / OGD (AdaGrad)", summary(adagrad(taps, rec, 0.05), from,
                    (System.nanoTime() - t0) / 1e6));
            t0 = System.nanoTime();
            double[][] x = new double[n][taps];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < taps && j <= i; j++) {
                    x[i][j] = rec.input()[i - j];
                }
            }
            double[] w = LeastSquares.ols(x, rec.desired(), false).coefficients();
            double[] mis = new double[n];
            for (int i = 0; i < n; i++) {
                mis[i] = io.github.zzuegg.brainiac.experimental.sensing.AdaptiveFilters.misalignment(w,
                        rec.weights()[i]);
            }
            double[] row = summary(mis, from, (System.nanoTime() - t0) / 1e6);
            row[0] = Double.NaN;
            out.put(sc[0] + " / offline least squares (whole record)", row);
        }
        return out;
    }

    private static double[] adagrad(int taps, ArraySignals.SystemRecord rec, double lr) {
        int n = rec.input().length;
        double[] w = new double[taps];
        double[] g2 = new double[taps];
        double[] u = new double[taps];
        double[] mis = new double[n];
        for (int i = 0; i < n; i++) {
            System.arraycopy(u, 0, u, 1, taps - 1);
            u[0] = rec.input()[i];
            double e = rec.desired()[i];
            for (int j = 0; j < taps; j++) {
                e -= w[j] * u[j];
            }
            for (int j = 0; j < taps; j++) {
                double g = -e * u[j];
                g2[j] += g * g;
                w[j] -= lr * g / Math.sqrt(g2[j] + 1e-8);
            }
            mis[i] = io.github.zzuegg.brainiac.experimental.sensing.AdaptiveFilters.misalignment(w, rec.weights()[i]);
        }
        return mis;
    }

    private static double[] summary(double[] mis, int from, double ms) {
        int n = mis.length;
        double conv = Double.NaN;
        for (int i = from; i < n; i++) {
            if (mis[i] < 0.01) {
                conv = i - from;
                break;
            }
        }
        double tail = 0;
        int t0 = n - n / 10;
        for (int i = t0; i < n; i++) {
            tail += mis[i] / (n - t0);
        }
        double after = 0;
        int end = Math.min(n, from + 500);
        for (int i = from; i < end; i++) {
            after += mis[i] / (end - from);
        }
        return new double[] {conv, db(tail), db(after), ms};
    }

    private static double db(double v) {
        return 10 * Math.log10(Math.max(v, 1e-30));
    }

    /**
     * Echo cancellation with double-talk in the second half.
     *
     * @param taps the room impulse response and filter length
     * @param n the samples
     * @param seed the seed
     * @return method to {ERLE dB in single talk, ERLE dB in double talk, final misalignment dB, share of
     *     double-talk samples flagged by the detector, share of single-talk samples flagged}
     */
    public static Map<String, double[]> echoCancellation(int taps, int n, long seed) {
        ArraySignals.EchoRecord rec = ArraySignals.echo(taps, n, true, 0.003, new SplittableRandom(seed));
        boolean[] talk = io.github.zzuegg.brainiac.experimental.sensing.AdaptiveFilters.geigel(rec.far(), rec.mic(),
                taps, 0.5, 240);
        double flagged = 0;
        double falseAlarm = 0;
        for (int i = 0; i < n; i++) {
            if (i >= n / 2) {
                flagged += (talk[i] ? 1.0 : 0.0) / (n - n / 2);
            } else {
                falseAlarm += (talk[i] ? 1.0 : 0.0) / (n / 2);
            }
        }
        Map<String, double[]> out = new LinkedHashMap<>();
        Settings st = new Settings(0.5, 4, 1e-2, 0.9995, 1e-7, 1e-3);
        Object[][] runs = {{"NLMS", Method.NLMS, false}, {"NLMS + Geigel", Method.NLMS, true},
            {"APA (K=4)", Method.APA, false}, {"RLS", Method.RLS, false},
            {"Kalman FIR (adaptive noise)", Method.KALMAN_FIR, false},
            {"Kalman FIR + Geigel", Method.KALMAN_FIR, true}};
        for (Object[] run : runs) {
            io.github.zzuegg.brainiac.experimental.sensing.AdaptiveFilters.Filter f =
                    new io.github.zzuegg.brainiac.experimental.sensing.AdaptiveFilters.Filter((Method) run[1], taps,
                            st);
            double[] est = new double[n];
            for (int i = 0; i < n; i++) {
                double e = f.step(rec.far()[i], rec.mic()[i], !((Boolean) run[2] && talk[i]));
                est[i] = rec.mic()[i] - e;
            }
            double[] erle = new double[2];
            for (int h = 0; h < 2; h++) {
                double num = 0;
                double den = 0;
                for (int i = h * n / 2 + n / 8; i < (h + 1) * n / 2; i++) {
                    num += rec.echo()[i] * rec.echo()[i];
                    double res = rec.echo()[i] - est[i];
                    den += res * res;
                }
                erle[h] = 10 * Math.log10(num / den);
            }
            out.put((String) run[0], new double[] {erle[0], erle[1],
                db(io.github.zzuegg.brainiac.experimental.sensing.AdaptiveFilters.misalignment(f.weights(),
                    rec.path())), flagged, falseAlarm});
        }
        return out;
    }
}
