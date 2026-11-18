package io.github.zzuegg.brainiac.experiments.sensing;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.ArraySignals;
import io.github.zzuegg.brainiac.experimental.sensing.ArrayProcessing;
import io.github.zzuegg.brainiac.experimental.sensing.ArrayProcessing.Covariance;
import io.github.zzuegg.brainiac.experimental.sensing.ArrayProcessing.Method;
import io.github.zzuegg.brainiac.experimental.sensing.SubspaceNet;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Direction finding on a uniform linear array (wave 13, items 13.4.14 and 13.4.15):
 * {@link #classical} compares delay-and-sum, MVDR, MUSIC, root-MUSIC and ESPRIT with the FFT
 * spatial periodogram, a direct MLP regressor and the Cramer-Rao bound (RMSE against SNR,
 * resolution of close sources, coherent sources with and without spatial smoothing,
 * miscalibration, MDL/AIC source counting); {@link #learned} compares SubspaceNet with them.
 */
public final class DirectionFinding {

    private DirectionFinding() {
    }

    static double[] randomAngles(int k, double minSeparation, double span, SplittableRandom r) {
        while (true) {
            double[] a = new double[k];
            for (int i = 0; i < k; i++) {
                a[i] = span * (2 * r.nextDouble() - 1);
            }
            Arrays.sort(a);
            boolean ok = true;
            for (int i = 1; i < k; i++) {
                ok &= a[i] - a[i - 1] >= minSeparation;
            }
            if (ok) {
                return a;
            }
        }
    }

    /**
     * The mean squared error of sorted estimates against sorted truth; missing estimates repeat
     * the nearest found one (0 degrees when none).
     *
     * @param est the estimates, ascending
     * @param truth the truth, ascending
     * @return the mean squared error in degrees squared
     */
    static double squaredError(double[] est, double[] truth) {
        double[] e = fill(est, truth);
        double s = 0;
        for (int i = 0; i < truth.length; i++) {
            s += (e[i] - truth[i]) * (e[i] - truth[i]);
        }
        return s / truth.length;
    }

    private static double[] fill(double[] est, double[] truth) {
        int k = truth.length;
        if (est.length == k) {
            return est;
        }
        double[] e = new double[k];
        for (int i = 0; i < k; i++) {
            if (est.length == 0) {
                e[i] = 0;
            } else {
                double best = est[0];
                for (double v : est) {
                    if (Math.abs(v - truth[i]) < Math.abs(best - truth[i])) {
                        best = v;
                    }
                }
                e[i] = best;
            }
        }
        return e;
    }

    static boolean resolved(double[] est, double[] truth) {
        if (est.length < truth.length) {
            return false;
        }
        double sep = truth[1] - truth[0];
        for (int i = 0; i < truth.length; i++) {
            if (Math.abs(est[i] - truth[i]) > sep / 2) {
                return false;
            }
        }
        return true;
    }

    /** Covariance features: the upper triangle's real and imaginary parts, normalised by the trace. */
    static float[] features(Covariance c) {
        int m = c.size();
        double tr = 0;
        for (int i = 0; i < m; i++) {
            tr += c.re()[i][i];
        }
        float[] f = new float[m * m];
        int k = 0;
        for (int i = 0; i < m; i++) {
            for (int j = i; j < m; j++) {
                f[k++] = (float) (m * c.re()[i][j] / tr);
                if (j > i) {
                    f[k++] = (float) (m * c.im()[i][j] / tr);
                }
            }
        }
        return f;
    }

    /** The "learn everything" baseline: an MLP from covariance features to sorted angles. */
    static final class DirectMlp {
        private final Var w1;
        private final Var b1;
        private final Var w2;
        private final Var b2;
        private final Var w3;
        private final Var b3;
        private final int k;

        DirectMlp(int inputs, int hidden, int k, SplittableRandom r) {
            this.k = k;
            w1 = Var.param(Tensor.normal(Shape.of(inputs, hidden), 0, (float) Math.sqrt(1.0 / inputs), r));
            b1 = Var.param(Tensor.zeros(Shape.of(hidden)));
            w2 = Var.param(Tensor.normal(Shape.of(hidden, hidden), 0, (float) Math.sqrt(1.0 / hidden), r));
            b2 = Var.param(Tensor.zeros(Shape.of(hidden)));
            w3 = Var.param(Tensor.normal(Shape.of(hidden, k), 0, (float) Math.sqrt(1.0 / hidden), r));
            b3 = Var.param(Tensor.zeros(Shape.of(k)));
        }

        Var forward(Var x) {
            return x.matmul(w1).add(b1).tanh().matmul(w2).add(b2).tanh().matmul(w3).add(b3);
        }

        void fit(int sensors, int snapshots, double snrLo, double snrHi, double minSep, double span, boolean coherent,
                int steps, int batch, SplittableRandom r) {
            Optimizer opt = Optimizer.adam(3e-3f);
            int in = sensors * sensors;
            for (int s = 0; s < steps; s++) {
                float[] x = new float[batch * in];
                float[] y = new float[batch * k];
                for (int b = 0; b < batch; b++) {
                    double[] ang = randomAngles(k, minSep, span, r);
                    double snr = snrLo + (snrHi - snrLo) * r.nextDouble();
                    float[] f = features(ArrayProcessing.covariance(
                            ArraySignals.ula(sensors, ang, snapshots, snr, coherent, 0, r)));
                    System.arraycopy(f, 0, x, b * in, in);
                    for (int j = 0; j < k; j++) {
                        y[b * k + j] = (float) (ang[j] / 90);
                    }
                }
                Var loss = forward(Var.of(Tensor.of(Shape.of(batch, in), x)))
                        .sub(Var.of(Tensor.of(Shape.of(batch, k), y))).square().mean();
                loss.backward();
                opt.step(List.of(w1, b1, w2, b2, w3, b3));
            }
        }

        double[] estimate(Covariance c) {
            float[] f = features(c);
            float[] out = Var.noGrad(() -> forward(Var.of(Tensor.of(Shape.of(1, f.length), f)))).value().toArray();
            double[] a = new double[k];
            for (int j = 0; j < k; j++) {
                a[j] = 90 * out[j];
            }
            Arrays.sort(a);
            return a;
        }
    }

    /**
     * The classical table.
     *
     * @param sensors {@code M}
     * @param snapshots {@code T}
     * @param trials the trials per cell
     * @param mlpSteps the direct MLP's training steps
     * @param seed the seed
     * @return row to values: "rmse / method" over SNR {-5, 0, 10, 20} dB (degrees), "resolution / method" over
     *     separations {2, 4, 8, 16} degrees at 10 dB, "coherent / ..." and "miscalibrated / ..." RMSE at 10 dB,
     *     "count / MDL|AIC" accuracy over the SNRs
     */
    public static Map<String, double[]> classical(int sensors, int snapshots, int trials, int mlpSteps, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        Map<String, double[]> out = new LinkedHashMap<>();
        double[] snrs = {-5, 0, 10, 20};
        int k = 2;
        DirectMlp mlp = new DirectMlp(sensors * sensors, 64, k, r);
        mlp.fit(sensors, snapshots, 0, 20, 10, 50, false, mlpSteps, 32, r);
        String[] rmseNames = {"FFT periodogram", "DELAY_AND_SUM", "MVDR", "MUSIC", "ROOT_MUSIC", "ESPRIT",
            "direct MLP (trained 0..20 dB)", "CRB"};
        for (String n : rmseNames) {
            out.put("rmse / " + n, new double[snrs.length]);
        }
        out.put("count / MDL", new double[snrs.length]);
        out.put("count / AIC", new double[snrs.length]);
        for (int si = 0; si < snrs.length; si++) {
            for (int t = 0; t < trials; t++) {
                double[] ang = randomAngles(k, 10, 50, r);
                ArraySignals.Snapshots x = ArraySignals.ula(sensors, ang, snapshots, snrs[si], false, 0, r);
                Covariance c = ArrayProcessing.covariance(x);
                double[][] est = {ArrayProcessing.periodogram(x, 256, k),
                    ArrayProcessing.estimate(Method.DELAY_AND_SUM, c, k), ArrayProcessing.estimate(Method.MVDR, c, k),
                    ArrayProcessing.estimate(Method.MUSIC, c, k), ArrayProcessing.estimate(Method.ROOT_MUSIC, c, k),
                    ArrayProcessing.estimate(Method.ESPRIT, c, k), mlp.estimate(c)};
                for (int m = 0; m < est.length; m++) {
                    out.get("rmse / " + rmseNames[m])[si] += squaredError(est[m], ang) / trials;
                }
                double[] crb = ArrayProcessing.crb(sensors, ang, snrs[si], snapshots);
                out.get("rmse / CRB")[si] += (crb[0] * crb[0] + crb[1] * crb[1]) / 2 / trials;
                double[] eig = ArrayProcessing.eigenvalues(c);
                out.get("count / MDL")[si] += (ArrayProcessing.sourceCount(true, eig, snapshots) == k ? 1.0 : 0)
                        / trials;
                out.get("count / AIC")[si] += (ArrayProcessing.sourceCount(false, eig, snapshots) == k ? 1.0 : 0)
                        / trials;
            }
        }
        for (String n : rmseNames) {
            double[] row = out.get("rmse / " + n);
            for (int i = 0; i < row.length; i++) {
                row[i] = Math.sqrt(row[i]);
            }
        }
        double[] seps = {2, 4, 8, 16};
        Method[] res = {Method.DELAY_AND_SUM, Method.MVDR, Method.MUSIC, Method.ROOT_MUSIC, Method.ESPRIT};
        for (Method m : res) {
            out.put("resolution / " + m, new double[seps.length]);
        }
        for (int si = 0; si < seps.length; si++) {
            for (int t = 0; t < trials; t++) {
                double c0 = 30 * (2 * r.nextDouble() - 1);
                double[] ang = {c0 - seps[si] / 2, c0 + seps[si] / 2};
                Covariance c = ArrayProcessing.covariance(ArraySignals.ula(sensors, ang, snapshots, 10, false, 0, r));
                for (Method m : res) {
                    out.get("resolution / " + m)[si] += (resolved(ArrayProcessing.estimate(m, c, k), ang) ? 1.0 : 0)
                            / trials;
                }
            }
        }
        String[] special = {"coherent / MUSIC", "coherent / MUSIC + FB spatial smoothing",
            "coherent / ESPRIT + FB spatial smoothing", "miscalibrated 0.1 / MUSIC", "miscalibrated 0.1 / ESPRIT",
            "miscalibrated 0.1 / MVDR", "miscalibrated 0.1 / direct MLP"};
        double[] acc = new double[special.length];
        for (int t = 0; t < trials; t++) {
            double[] ang = randomAngles(k, 10, 50, r);
            Covariance coh = ArrayProcessing.covariance(ArraySignals.ula(sensors, ang, snapshots, 10, true, 0, r));
            Covariance smooth = ArrayProcessing.spatialSmoothing(coh, sensors - 2, true);
            Covariance mis = ArrayProcessing.covariance(ArraySignals.ula(sensors, ang, snapshots, 10, false, 0.1, r));
            double[][] est = {ArrayProcessing.estimate(Method.MUSIC, coh, k),
                ArrayProcessing.estimate(Method.MUSIC, smooth, k), ArrayProcessing.estimate(Method.ESPRIT, smooth, k),
                ArrayProcessing.estimate(Method.MUSIC, mis, k), ArrayProcessing.estimate(Method.ESPRIT, mis, k),
                ArrayProcessing.estimate(Method.MVDR, mis, k), mlp.estimate(mis)};
            for (int i = 0; i < est.length; i++) {
                acc[i] += squaredError(est[i], ang) / trials;
            }
        }
        for (int i = 0; i < special.length; i++) {
            out.put(special[i], new double[] {Math.sqrt(acc[i])});
        }
        return out;
    }

    /**
     * The learned table (13.4.15): SubspaceNet with coherent sources and few snapshots against
     * MUSIC on the sample covariance, forward-backward spatially smoothed MUSIC, root-MUSIC and
     * ESPRIT, the direct MLP trained on the same data and the (uncorrelated-source) CRB; and the
     * near-field DCD-MUSIC with a sample or learned covariance against 2-D MUSIC.
     *
     * @param sensors {@code M}
     * @param snapshots {@code T}
     * @param steps the training steps of each learned model
     * @param trials the test trials per cell
     * @param seed the seed
     * @return "far / method" to {RMSE deg at 10 dB, at 0 dB (outside the 5-15 dB training range), at 10 dB with
     *     0.1 miscalibration}; "near / method" to {angle RMSE deg, range RMSE wavelengths}; "parameters" to
     *     {SubspaceNet, direct MLP}
     */
    public static Map<String, double[]> learned(int sensors, int snapshots, int steps, int trials, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int k = 2;
        SubspaceNet net = new SubspaceNet(sensors, k, 3, 64, SubspaceNet.Geometry.FAR_FIELD, seed + 1);
        net.fit(rr -> {
            double[] ang = randomAngles(k, 10, 50, rr);
            return new SubspaceNet.Example(ArraySignals.ula(sensors, ang, snapshots, 5 + 10 * rr.nextDouble(), true, 0,
                    rr), ang, null);
        }, steps, 16, 3e-3f, seed + 2);
        DirectMlp mlp = new DirectMlp(sensors * sensors, 64, k, r);
        mlp.fit(sensors, snapshots, 5, 15, 10, 50, true, steps, 32, r);
        String[] names = {"SubspaceNet + MUSIC", "SubspaceNet + root-MUSIC", "MUSIC (sample covariance)",
            "MUSIC + FB spatial smoothing", "root-MUSIC + FB spatial smoothing", "ESPRIT + FB spatial smoothing",
            "direct MLP (same data)", "CRB (uncorrelated sources)"};
        Map<String, double[]> out = new LinkedHashMap<>();
        for (String n : names) {
            out.put("far / " + n, new double[3]);
        }
        double[][] conditions = {{10, 0}, {0, 0}, {10, 0.1}};
        for (int c = 0; c < conditions.length; c++) {
            for (int t = 0; t < trials; t++) {
                double[] ang = randomAngles(k, 10, 50, r);
                ArraySignals.Snapshots x = ArraySignals.ula(sensors, ang, snapshots, conditions[c][0], true,
                        conditions[c][1], r);
                Covariance sample = ArrayProcessing.covariance(x);
                Covariance smooth = ArrayProcessing.spatialSmoothing(sample, sensors - 2, true);
                Covariance learned = net.surrogate(x);
                double[][] est = {ArrayProcessing.estimate(Method.MUSIC, learned, k),
                    ArrayProcessing.estimate(Method.ROOT_MUSIC, learned, k),
                    ArrayProcessing.estimate(Method.MUSIC, sample, k), ArrayProcessing.estimate(Method.MUSIC, smooth, k),
                    ArrayProcessing.estimate(Method.ROOT_MUSIC, smooth, k),
                    ArrayProcessing.estimate(Method.ESPRIT, smooth, k), mlp.estimate(sample)};
                for (int m = 0; m < est.length; m++) {
                    out.get("far / " + names[m])[c] += squaredError(est[m], ang) / trials;
                }
                double[] crb = ArrayProcessing.crb(sensors, ang, conditions[c][0], snapshots);
                out.get("far / " + names[names.length - 1])[c] += (crb[0] * crb[0] + crb[1] * crb[1]) / 2 / trials;
            }
        }
        for (String n : names) {
            double[] row = out.get("far / " + n);
            for (int i = 0; i < row.length; i++) {
                row[i] = Math.sqrt(row[i]);
            }
        }
        double lo = 2;
        double hi = 8;
        SubspaceNet near = new SubspaceNet(sensors, k, 3, 64, SubspaceNet.Geometry.NEAR_FIELD, seed + 3);
        near.fit(rr -> {
            double[] ang = randomAngles(k, 10, 40, rr);
            double[] rng = {lo + (hi - lo) * rr.nextDouble(), lo + (hi - lo) * rr.nextDouble()};
            return new SubspaceNet.Example(ArraySignals.nearField(sensors, ang, rng, snapshots, 10, rr), ang, rng);
        }, steps, 16, 3e-3f, seed + 4);
        String[] nearNames = {"DCD-MUSIC (sample covariance)", "DCD-MUSIC (SubspaceNet covariance)",
            "2-D MUSIC (sample covariance)"};
        for (String n : nearNames) {
            out.put("near / " + n, new double[2]);
        }
        for (int t = 0; t < trials; t++) {
            double[] ang = randomAngles(k, 10, 40, r);
            double[] rng = {lo + (hi - lo) * r.nextDouble(), lo + (hi - lo) * r.nextDouble()};
            ArraySignals.Snapshots x = ArraySignals.nearField(sensors, ang, rng, snapshots, 10, r);
            Covariance sample = ArrayProcessing.covariance(x);
            double[][][] est = {SubspaceNet.dcdMusic(sample, k, lo, hi), SubspaceNet.dcdMusic(near.surrogate(x), k, lo,
                hi), SubspaceNet.music2d(sample, k, lo, hi)};
            for (int m = 0; m < est.length; m++) {
                double[] row = out.get("near / " + nearNames[m]);
                row[0] += squaredError(est[m][0], ang) / trials;
                double[] rt = est[m][1].length == k ? est[m][1] : new double[] {(lo + hi) / 2, (lo + hi) / 2};
                row[1] += ((rt[0] - rng[0]) * (rt[0] - rng[0]) + (rt[1] - rng[1]) * (rt[1] - rng[1])) / 2 / trials;
            }
        }
        for (String n : nearNames) {
            double[] row = out.get("near / " + n);
            row[0] = Math.sqrt(row[0]);
            row[1] = Math.sqrt(row[1]);
        }
        out.put("parameters", new double[] {net.parameterCount(), sensors * sensors * 64 + 64 * 64 + 64 * k + 128 + k});
        return out;
    }
}
