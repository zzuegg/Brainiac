package io.github.zzuegg.brainiac.experimental.data;

import java.util.SplittableRandom;

/**
 * Multichannel series on a fine time grid over {@code [0, 1]} with most values missing, so that
 * the observed points form an irregular sampling pattern:
 * <ul>
 *   <li>{@link Kind#SEASONAL}: per series, a sum of two sinusoids of random frequencies (1 to 4
 *       cycles per window) shared by all channels with channel-specific phases and amplitudes,
 *       plus a little noise; smooth dynamics;</li>
 *   <li>{@link Kind#CHAOTIC}: windows of the Lorenz attractor ({@link ChaoticSeries#lorenz}),
 *       channels {@code x, y, z};</li>
 *   <li>{@link Kind#LINEAR_GAUSSIAN}: independent damped stochastic oscillators (an AR(2) process
 *       per channel), the case a Kalman smoother is built for.</li>
 * </ul>
 * Every channel is standardised. Masks remove a target share of values
 * {@link Missing#MAR missing at random}, {@link Missing#MNAR not at random} (large values are
 * more likely to be missing) or in {@link Missing#BLOCK blocks} of consecutive steps.
 */
public final class IrregularSeries {

    /** The dynamics. */
    public enum Kind {
        /** Smooth periodic. */
        SEASONAL,
        /** Lorenz windows (3 channels). */
        CHAOTIC,
        /** Stochastic damped oscillators. */
        LINEAR_GAUSSIAN
    }

    /** The missingness mechanism. */
    public enum Missing {
        /** Independent of the values. */
        MAR,
        /** Depends on the missing value itself. */
        MNAR,
        /** Runs of 3 to 10 consecutive missing steps per channel. */
        BLOCK
    }

    private IrregularSeries() {
    }

    /**
     * The grid times.
     *
     * @param steps grid points
     * @return {@code steps} times evenly spread over {@code [0, 1]}
     */
    public static double[] times(int steps) {
        double[] t = new double[steps];
        for (int i = 0; i < steps; i++) {
            t[i] = steps == 1 ? 0 : (double) i / (steps - 1);
        }
        return t;
    }

    /**
     * Complete series.
     *
     * @param kind the dynamics
     * @param count series
     * @param steps grid points
     * @param channels channels (3 for {@link Kind#CHAOTIC})
     * @param seed the seed
     * @return values {@code [count][steps][channels]}
     */
    public static float[][][] generate(Kind kind, int count, int steps, int channels, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int c = kind == Kind.CHAOTIC ? 3 : channels;
        float[][][] out = new float[count][steps][c];
        double[][] lorenz = kind == Kind.CHAOTIC ? ChaoticSeries.lorenz(steps * 4 * 50 + steps * 4, 0.01, 500) : null;
        for (int i = 0; i < count; i++) {
            switch (kind) {
                case SEASONAL -> {
                    double f1 = 1 + 3 * r.nextDouble();
                    double f2 = 1 + 3 * r.nextDouble();
                    double w = r.nextDouble();
                    for (int ch = 0; ch < c; ch++) {
                        double p1 = r.nextDouble() * 2 * Math.PI;
                        double p2 = r.nextDouble() * 2 * Math.PI;
                        double amp = 0.5 + r.nextDouble();
                        for (int t = 0; t < steps; t++) {
                            double x = (double) t / (steps - 1);
                            out[i][t][ch] = (float) (amp * (Math.sin(2 * Math.PI * f1 * x + p1)
                                    + w * Math.sin(2 * Math.PI * f2 * x + p2)) + 0.05 * r.nextGaussian());
                        }
                    }
                }
                case CHAOTIC -> {
                    int start = r.nextInt(lorenz.length - steps * 4);
                    double[] scale = {8, 9, 9};
                    double[] shift = {0, 0, 24};
                    for (int t = 0; t < steps; t++) {
                        for (int ch = 0; ch < 3; ch++) {
                            out[i][t][ch] = (float) ((lorenz[start + 4 * t][ch] - shift[ch]) / scale[ch]);
                        }
                    }
                }
                default -> {
                    for (int ch = 0; ch < c; ch++) {
                        double rho = 0.9 + 0.08 * r.nextDouble();
                        double theta = 2 * Math.PI / (6 + 10 * r.nextDouble());
                        double a1 = 2 * rho * Math.cos(theta);
                        double a2 = -rho * rho;
                        double x1 = 0;
                        double x2 = 0;
                        double[] path = new double[steps + 50];
                        for (int t = 0; t < path.length; t++) {
                            double x = a1 * x1 + a2 * x2 + 0.3 * r.nextGaussian();
                            x2 = x1;
                            x1 = x;
                            path[t] = x;
                        }
                        for (int t = 0; t < steps; t++) {
                            out[i][t][ch] = (float) (path[t + 50] / 1.2 + 0.05 * r.nextGaussian());
                        }
                    }
                }
            }
        }
        return out;
    }

    /**
     * An observation mask.
     *
     * @param values complete values
     * @param missing the target share of missing values
     * @param mechanism how values go missing
     * @param seed the seed
     * @return {@code true} where observed
     */
    public static boolean[][][] mask(float[][][] values, double missing, Missing mechanism, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int n = values.length;
        int steps = values[0].length;
        int c = values[0][0].length;
        boolean[][][] obs = new boolean[n][steps][c];
        if (mechanism == Missing.BLOCK) {
            for (int i = 0; i < n; i++) {
                for (int ch = 0; ch < c; ch++) {
                    for (int t = 0; t < steps; t++) {
                        obs[i][t][ch] = true;
                    }
                    int removed = 0;
                    int guard = 0;
                    while (removed < missing * steps && guard++ < 1000) {
                        int len = 3 + r.nextInt(8);
                        int start = r.nextInt(steps);
                        for (int t = start; t < Math.min(steps, start + len); t++) {
                            if (obs[i][t][ch] && removed < missing * steps) {
                                obs[i][t][ch] = false;
                                removed++;
                            }
                        }
                    }
                }
            }
            return obs;
        }
        double bias = 0;
        if (mechanism == Missing.MNAR) {
            // calibrate sigmoid(bias + 1.5 x) to the target share
            double lo = -20;
            double hi = 20;
            for (int it = 0; it < 50; it++) {
                double mid = (lo + hi) / 2;
                double rate = 0;
                for (float[][] s : values) {
                    for (float[] row : s) {
                        for (float v : row) {
                            rate += 1 / (1 + Math.exp(-(mid + 1.5 * v)));
                        }
                    }
                }
                rate /= (double) n * steps * c;
                if (rate < missing) {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            bias = (lo + hi) / 2;
        }
        for (int i = 0; i < n; i++) {
            for (int t = 0; t < steps; t++) {
                for (int ch = 0; ch < c; ch++) {
                    double p = mechanism == Missing.MAR ? missing
                            : 1 / (1 + Math.exp(-(bias + 1.5 * values[i][t][ch])));
                    obs[i][t][ch] = r.nextDouble() >= p;
                }
            }
        }
        return obs;
    }
}
