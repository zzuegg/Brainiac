package io.github.zzuegg.brainiac.experimental.data;

import java.util.SplittableRandom;

/**
 * UCR-style series classification tasks beyond {@link SeriesClassificationTasks}' cylinder-bell-funnel,
 * shapelets and frequency bands. Each generator plants a different kind of class signal, so the
 * families of series classifiers (whole-series elastic distances, dictionary methods, interval
 * methods, convolutions) can each be shown winning and losing:
 * <ul>
 *   <li>{@link #twoPatterns}: two up-down or down-up steps at random positions; the class is
 *       their order (Geurts 2001);</li>
 *   <li>{@link #syntheticControl}: the six control-chart classes of Alcock and Manolopoulos
 *       (normal, cyclic, increasing and decreasing trend, upward and downward shift);</li>
 *   <li>{@link #trace}: four transient shapes (step, step with an overshoot, damped oscillation,
 *       ramp) at a random onset, after the nuclear-plant Trace data;</li>
 *   <li>{@link #phaseShifted}: one motif (sine burst, square pulse, triangle) at a random phase;</li>
 *   <li>{@link #warpedTemplates}: smooth class templates under random monotone time warps,
 *       where the elastic distances shine;</li>
 *   <li>{@link #dictionary}: two short motifs at random positions; the class is how often each
 *       occurs, the signal dictionary (bag-of-words) methods are built for;</li>
 *   <li>{@link #interval}: the class lives in a statistic (mean or variance) of one fixed
 *       interval, the signal interval methods are built for;</li>
 *   <li>{@link #multivariate}: two channels whose cross-lag decides the class.</li>
 * </ul>
 * Labels cycle through the classes ({@code label = i % classes}), so every prefix is balanced.
 * {@link #withNoise} and {@link #resample} add noise and length variation to any task.
 */
public final class SeriesArchive {

    private SeriesArchive() {
    }

    /**
     * A multivariate task.
     *
     * @param series {@code [n][channels][length]}
     * @param labels the classes
     * @param classes the number of classes
     */
    public record Multivariate(float[][][] series, int[] labels, int classes) {
        /** @return the number of series */
        public int size() {
            return labels.length;
        }
    }

    /**
     * Two patterns (four classes): an up-down ({@code U}) or down-up ({@code D}) pattern at two
     * random, non-overlapping positions; classes UU, UD, DU, DD.
     *
     * @param n series
     * @param length steps, at least 64
     * @param seed the seed
     * @return the task
     */
    public static SeriesClassificationTasks.Task twoPatterns(int n, int length, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        float[][] x = new float[n][length];
        int[] y = new int[n];
        int w = length / 8;
        for (int i = 0; i < n; i++) {
            int c = i % 4;
            y[i] = c;
            noise(x[i], 0.3, r);
            int a = r.nextInt(length / 2 - w);
            int b = length / 2 + r.nextInt(length / 2 - w);
            plantUpDown(x[i], a, w, (c & 2) == 0);
            plantUpDown(x[i], b, w, (c & 1) == 0);
        }
        return new SeriesClassificationTasks.Task(x, y, 4);
    }

    private static void plantUpDown(float[] x, int start, int w, boolean up) {
        for (int t = 0; t < w; t++) {
            double v = t < w / 2 ? 1 : -1;
            x[start + t] += (float) (up ? 2 * v : -2 * v);
        }
    }

    /**
     * Synthetic control charts (six classes).
     *
     * @param n series
     * @param length steps, at least 32
     * @param seed the seed
     * @return the task
     */
    public static SeriesClassificationTasks.Task syntheticControl(int n, int length, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        float[][] x = new float[n][length];
        int[] y = new int[n];
        for (int i = 0; i < n; i++) {
            int c = i % 6;
            y[i] = c;
            double amp = 1 + r.nextDouble();
            double period = 10 + 5 * r.nextDouble();
            double slope = (0.2 + 0.3 * r.nextDouble()) * 60.0 / length;
            int shift = length / 3 + r.nextInt(length / 3);
            double jump = 7.5 + 7.5 * r.nextDouble();
            for (int t = 0; t < length; t++) {
                double v = 30 + 2 * r.nextGaussian();
                v += switch (c) {
                    case 1 -> amp * 5 * Math.sin(2 * Math.PI * t / period);
                    case 2 -> slope * t;
                    case 3 -> -slope * t;
                    case 4 -> t >= shift ? jump : 0;
                    case 5 -> t >= shift ? -jump : 0;
                    default -> 0;
                };
                x[i][t] = (float) ((v - 30) / 5);
            }
        }
        return new SeriesClassificationTasks.Task(x, y, 6);
    }

    /**
     * Trace-like transients (four classes) at a random onset in the middle half.
     *
     * @param n series
     * @param length steps, at least 64
     * @param seed the seed
     * @return the task
     */
    public static SeriesClassificationTasks.Task trace(int n, int length, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        float[][] x = new float[n][length];
        int[] y = new int[n];
        for (int i = 0; i < n; i++) {
            int c = i % 4;
            y[i] = c;
            int onset = length / 4 + r.nextInt(length / 2);
            noise(x[i], 0.15, r);
            for (int t = onset; t < length; t++) {
                double u = t - onset;
                double v = switch (c) {
                    case 0 -> 1;
                    case 1 -> 1 + 1.5 * Math.exp(-u / 3);
                    case 2 -> Math.exp(-u / 12) * Math.sin(u / 2.0) * 1.5;
                    default -> Math.min(1, u / 20);
                };
                x[i][t] += (float) v;
            }
        }
        return new SeriesClassificationTasks.Task(x, y, 4);
    }

    /**
     * Phase-shifted motifs (three classes): a sine burst, a square pulse or a triangle of a fixed
     * width at a uniformly random position.
     *
     * @param n series
     * @param length steps, at least 48
     * @param seed the seed
     * @return the task
     */
    public static SeriesClassificationTasks.Task phaseShifted(int n, int length, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        float[][] x = new float[n][length];
        int[] y = new int[n];
        int w = Math.max(12, length / 6);
        for (int i = 0; i < n; i++) {
            int c = i % 3;
            y[i] = c;
            noise(x[i], 0.3, r);
            int s = r.nextInt(length - w);
            for (int t = 0; t < w; t++) {
                double f = t / (double) (w - 1);
                double v = switch (c) {
                    case 0 -> Math.sin(2 * Math.PI * f);
                    case 1 -> f < 0.5 ? 1 : -1;
                    default -> 1 - Math.abs(2 * f - 1) * 2;
                };
                x[i][s + t] += (float) (1.5 * v);
            }
        }
        return new SeriesClassificationTasks.Task(x, y, 3);
    }

    /**
     * Warped templates: each class has a template of three Gaussian bumps of heights +2, -1.5
     * and +1 at a class-specific permutation of the positions 0.25, 0.5 and 0.75 (so classes differ
     * only in the order of the same events, at most six classes). Every series is its class
     * template under a random monotone time warp
     * {@code u -> u + a1 sin(pi u) / pi + a2 sin(2 pi u) / (2 pi)} plus amplitude jitter and noise.
     * A warp moves bumps by several bump widths, which defeats lock-step distances, while the order
     * of events, which elastic distances follow, stays fixed.
     *
     * @param n series
     * @param length steps
     * @param classes classes, 2 to 6
     * @param warp the warp strength in {@code [0, 0.6)}, for example 0.45
     * @param seed the seed
     * @return the task
     */
    public static SeriesClassificationTasks.Task warpedTemplates(int n, int length, int classes, double warp,
            long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][][] bumps = bumps(classes);
        float[][] x = new float[n][length];
        int[] y = new int[n];
        for (int i = 0; i < n; i++) {
            int c = i % classes;
            y[i] = c;
            double a1 = warp * (2 * r.nextDouble() - 1);
            double a2 = warp * 0.5 * (2 * r.nextDouble() - 1);
            double scale = 1 + 0.1 * r.nextGaussian();
            for (int t = 0; t < length; t++) {
                double u = t / (double) (length - 1);
                double wu = u + a1 * Math.sin(Math.PI * u) / Math.PI + a2 * Math.sin(2 * Math.PI * u) / (2 * Math.PI);
                x[i][t] = (float) (scale * template(bumps[c], wu) + 0.2 * r.nextGaussian());
            }
        }
        return new SeriesClassificationTasks.Task(x, y, classes);
    }

    /**
     * The noise-free template of {@link #warpedTemplates}' class {@code c}, for scoring averages.
     *
     * @param length steps
     * @param classes classes (the same as for the task)
     * @param c the class
     * @return the template
     */
    public static double[] warpedTemplate(int length, int classes, int c) {
        double[][][] bumps = bumps(classes);
        double[] out = new double[length];
        for (int t = 0; t < length; t++) {
            out[t] = template(bumps[c], t / (double) (length - 1));
        }
        return out;
    }

    /** The class templates' bumps (centre, width, height): permutations of three fixed events. */
    private static double[][][] bumps(int classes) {
        int[][] perms = {{0, 1, 2}, {2, 1, 0}, {1, 0, 2}, {0, 2, 1}, {2, 0, 1}, {1, 2, 0}};
        double[] centres = {0.25, 0.5, 0.75};
        double[] heights = {2, -1.5, 1};
        double[][][] bumps = new double[classes][3][3];
        for (int c = 0; c < classes; c++) {
            for (int b = 0; b < 3; b++) {
                bumps[c][b][0] = centres[perms[c % perms.length][b]];
                bumps[c][b][1] = 0.04;
                bumps[c][b][2] = heights[b];
            }
        }
        return bumps;
    }

    private static double template(double[][] bumps, double u) {
        double v = 0;
        for (double[] b : bumps) {
            double z = (u - b[0]) / b[1];
            v += b[2] * Math.exp(-0.5 * z * z);
        }
        return v;
    }

    /**
     * Dictionary-type classes (three classes): eight short sawtooth motifs at random
     * non-overlapping positions, rising ramps with a sharp drop ({@code A}) or sharp rises with a
     * falling ramp ({@code B}, A reversed in time); class 0 has 6 A and 2 B, class 1 has 2 A and
     * 6 B, class 2 has 4 of each. The two motifs have the same values in a different order and
     * where they sit carries no information, so whole-series alignment (which can stretch a ramp
     * into a jump) fails and counts of local patterns succeed.
     *
     * @param n series
     * @param length steps, at least 96
     * @param seed the seed
     * @return the task
     */
    public static SeriesClassificationTasks.Task dictionary(int n, int length, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        float[][] x = new float[n][length];
        int[] y = new int[n];
        int w = 10;
        int slots = length / (w + 2);
        for (int i = 0; i < n; i++) {
            int c = i % 3;
            y[i] = c;
            noise(x[i], 0.3, r);
            int a = c == 0 ? 6 : c == 1 ? 2 : 4;
            int[] slot = permutation(slots, r);
            for (int m = 0; m < 8; m++) {
                int s = slot[m] * (w + 2) + r.nextInt(3);
                boolean up = m < a;
                for (int t = 0; t < w && s + t < length; t++) {
                    double ramp = t / (double) (w - 1) - 0.5;
                    x[i][s + t] += (float) (2.4 * (up ? ramp : -ramp));
                }
            }
        }
        return new SeriesClassificationTasks.Task(x, y, 3);
    }

    /**
     * Interval-type classes (three classes): unit noise everywhere and, in the fixed interval
     * {@code [length/3, length/3 + length/8)}, a mean shift of +0.8 (class 0), -0.8 (class 1) or
     * no shift but double the standard deviation (class 2). Elsewhere, distractor bumps at
     * random positions carry no information.
     *
     * @param n series
     * @param length steps, at least 64
     * @param seed the seed
     * @return the task
     */
    public static SeriesClassificationTasks.Task interval(int n, int length, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        float[][] x = new float[n][length];
        int[] y = new int[n];
        int from = length / 3;
        int to = from + length / 8;
        for (int i = 0; i < n; i++) {
            int c = i % 3;
            y[i] = c;
            noise(x[i], 1, r);
            for (int t = from; t < to; t++) {
                x[i][t] = (float) switch (c) {
                    case 0 -> x[i][t] + 0.8;
                    case 1 -> x[i][t] - 0.8;
                    default -> 2 * x[i][t];
                };
            }
            int d = to + r.nextInt(Math.max(1, length - to - 8));
            double sign = r.nextBoolean() ? 1 : -1;
            for (int t = d; t < Math.min(length, d + 8); t++) {
                x[i][t] += (float) (sign * 1.5);
            }
        }
        return new SeriesClassificationTasks.Task(x, y, 3);
    }

    /**
     * Two-channel series (three classes): channel 1 is channel 0 delayed by {@code +lag}, not
     * delayed, or advanced by {@code lag}; channel 0 is smoothed noise.
     *
     * @param n series
     * @param length steps
     * @param seed the seed
     * @return the task
     */
    public static Multivariate multivariate(int n, int length, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        float[][][] x = new float[n][2][length];
        int[] y = new int[n];
        int lag = 6;
        for (int i = 0; i < n; i++) {
            int c = i % 3;
            y[i] = c;
            double[] base = new double[length + 2 * lag];
            double s = 0;
            for (int t = 0; t < base.length; t++) {
                s = 0.8 * s + r.nextGaussian();
                base[t] = s / 1.7;
            }
            int shift = c == 0 ? lag : c == 1 ? 0 : -lag;
            for (int t = 0; t < length; t++) {
                x[i][0][t] = (float) base[t + lag];
                x[i][1][t] = (float) (base[t + lag - shift] + 0.3 * r.nextGaussian());
            }
        }
        return new Multivariate(x, y, 3);
    }

    /**
     * Adds Gaussian noise to a task.
     *
     * @param task the task
     * @param sigma the noise standard deviation
     * @param seed the seed
     * @return a noisy copy
     */
    public static SeriesClassificationTasks.Task withNoise(SeriesClassificationTasks.Task task, double sigma,
            long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        float[][] x = new float[task.size()][];
        for (int i = 0; i < x.length; i++) {
            x[i] = task.series()[i].clone();
            for (int t = 0; t < x[i].length; t++) {
                x[i][t] += (float) (sigma * r.nextGaussian());
            }
        }
        return new SeriesClassificationTasks.Task(x, task.labels().clone(), task.classes());
    }

    /**
     * Resamples every series to a new length by linear interpolation (length variation).
     *
     * @param task the task
     * @param length the new length
     * @return the resampled copy
     */
    public static SeriesClassificationTasks.Task resample(SeriesClassificationTasks.Task task, int length) {
        float[][] x = new float[task.size()][length];
        for (int i = 0; i < x.length; i++) {
            float[] s = task.series()[i];
            for (int t = 0; t < length; t++) {
                double u = t * (s.length - 1) / (double) Math.max(1, length - 1);
                int a = (int) Math.floor(u);
                int b = Math.min(s.length - 1, a + 1);
                double f = u - a;
                x[i][t] = (float) ((1 - f) * s[a] + f * s[b]);
            }
        }
        return new SeriesClassificationTasks.Task(x, task.labels().clone(), task.classes());
    }

    /**
     * A task by name: "two-patterns", "control", "trace", "phase", "warped", "dictionary",
     * "interval", or one of {@link SeriesClassificationTasks}' "CBF" and "shapelets".
     *
     * @param name the task name
     * @param n series
     * @param length steps
     * @param seed the seed
     * @return the task
     */
    public static SeriesClassificationTasks.Task named(String name, int n, int length, long seed) {
        return switch (name) {
            case "two-patterns" -> twoPatterns(n, length, seed);
            case "control" -> syntheticControl(n, length, seed);
            case "trace" -> trace(n, length, seed);
            case "phase" -> phaseShifted(n, length, seed);
            case "warped" -> warpedTemplates(n, length, 6, 0.45, seed);
            case "dictionary" -> dictionary(n, length, seed);
            case "interval" -> interval(n, length, seed);
            case "CBF" -> SeriesClassificationTasks.cylinderBellFunnel(n, length, seed);
            case "shapelets" -> SeriesClassificationTasks.shapelets(n, length, seed);
            default -> throw new IllegalArgumentException("unknown task " + name);
        };
    }

    private static void noise(float[] x, double sigma, SplittableRandom r) {
        for (int t = 0; t < x.length; t++) {
            x[t] = (float) (sigma * r.nextGaussian());
        }
    }

    private static int[] permutation(int n, SplittableRandom r) {
        int[] p = new int[n];
        for (int i = 0; i < n; i++) {
            p[i] = i;
        }
        for (int i = n - 1; i > 0; i--) {
            int j = r.nextInt(i + 1);
            int t = p[i];
            p[i] = p[j];
            p[j] = t;
        }
        return p;
    }
}
