package io.github.zzuegg.brainiac.experimental.ml;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Randomised-weight networks: the hidden weights are drawn once and frozen, and only the read-out
 * is solved, by ridge regression in closed form.
 * <ul>
 *   <li>{@link Kind#ELM} (Huang, Zhu and Siew, Neurocomputing 2006): one random hidden layer
 *       {@code H = g(x W + b)}, read-out on {@code [H, 1]}.</li>
 *   <li>{@link Kind#RVFL} (Pao, Park and Sobajic, Neurocomputing 1994): the same hidden layer plus
 *       direct input-output links, read-out on {@code [x, H, 1]}.</li>
 *   <li>{@link Kind#DEEP_RVFL} (Katuwal and Suganthan, 2019): {@code L} random layers, each fed
 *       with the input and the previous layer ({@code H_l = g([x, H_{l-1}] W_l + b_l)}), one
 *       read-out on {@code [x, H_1, ..., H_L, 1]}.</li>
 *   <li>{@link Kind#ENSEMBLE_DEEP_RVFL}: the same stack with one read-out per layer on
 *       {@code [x, H_l, 1]}; the outputs are averaged.</li>
 * </ul>
 * The inputs are standardised with the training statistics; the hidden features of the deep
 * variants are standardised layer by layer before they feed the next layer, as in edRVFL. With
 * {@code lambda <= 0} the ridge penalty is chosen from a grid on a 20% hold-out of the training
 * rows, then the read-out is refitted on all rows.
 */
public final class RandomWeightNetworks {

    /** The architecture. */
    public enum Kind {
        /** Extreme learning machine: one random layer, no direct links. */
        ELM,
        /** Random vector functional link: one random layer plus direct links. */
        RVFL,
        /** Deep RVFL: stacked random layers, all of them and the input feeding one read-out. */
        DEEP_RVFL,
        /** Ensemble deep RVFL: one read-out per layer, averaged. */
        ENSEMBLE_DEEP_RVFL
    }

    /** The hidden activation. */
    public enum Activation {
        /** The logistic sigmoid. */
        SIGMOID,
        /** The hyperbolic tangent. */
        TANH,
        /** The rectifier. */
        RELU
    }

    /**
     * The settings.
     *
     * @param kind the architecture
     * @param hidden the width of each random layer
     * @param layers the random layers (1 for ELM and RVFL)
     * @param activation the activation
     * @param scale the random weights are uniform with standard deviation {@code scale / sqrt(fan-in)}, so
     *     that standardised inputs give pre-activations of standard deviation about {@code scale}; biases
     *     are uniform on {@code [0, scale]} for the sigmoid, else on {@code [-scale, scale]}
     * @param lambda the ridge penalty, or {@code <= 0} to choose it on a hold-out
     * @param seed the seed of the random weights
     */
    public record Settings(Kind kind, int hidden, int layers, Activation activation, double scale, double lambda,
            long seed) {

        /**
         * Validates.
         *
         * @param kind the architecture
         * @param hidden the width
         * @param layers the layers
         * @param activation the activation
         * @param scale the weight scale
         * @param lambda the ridge penalty
         * @param seed the seed
         */
        public Settings {
            if (hidden < 1 || layers < 1 || scale <= 0) {
                throw new IllegalArgumentException("hidden, layers and scale must be positive");
            }
        }

        /**
         * Defaults: tanh, scale 1, ridge chosen on a hold-out.
         *
         * @param kind the architecture
         * @param hidden the width
         * @param layers the layers (forced to 1 for ELM and RVFL)
         * @param seed the seed
         * @return the settings
         */
        public static Settings of(Kind kind, int hidden, int layers, long seed) {
            int l = kind == Kind.ELM || kind == Kind.RVFL ? 1 : layers;
            return new Settings(kind, hidden, l, Activation.TANH, 1, 0, seed);
        }
    }

    /** A frozen random layer with the standardisation of its outputs. */
    private record RandomLayer(double[][] w, double[] b, double[] mean, double[] std) {
    }

    private static final double[] LAMBDAS = {1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100};

    private final Settings settings;
    private final double[] inMean;
    private final double[] inStd;
    private final List<RandomLayer> layers;
    private final List<double[][]> readouts;
    private final double lambda;

    private RandomWeightNetworks(Settings settings, double[] inMean, double[] inStd, List<RandomLayer> layers,
            List<double[][]> readouts, double lambda) {
        this.settings = settings;
        this.inMean = inMean;
        this.inStd = inStd;
        this.layers = layers;
        this.readouts = readouts;
        this.lambda = lambda;
    }

    /**
     * Fits a classifier on one-hot targets.
     *
     * @param x the inputs {@code [n][d]}
     * @param labels the classes in {@code [0, classes)}
     * @param classes the number of classes
     * @param settings the settings
     * @return the network
     */
    public static RandomWeightNetworks classifier(double[][] x, int[] labels, int classes, Settings settings) {
        double[][] y = new double[x.length][classes];
        for (int i = 0; i < x.length; i++) {
            y[i][labels[i]] = 1;
        }
        return fit(x, y, settings);
    }

    /**
     * Fits a regressor.
     *
     * @param x the inputs {@code [n][d]}
     * @param y the targets
     * @param settings the settings
     * @return the network
     */
    public static RandomWeightNetworks regressor(double[][] x, double[] y, Settings settings) {
        double[][] t = new double[y.length][1];
        for (int i = 0; i < y.length; i++) {
            t[i][0] = y[i];
        }
        return fit(x, t, settings);
    }

    /**
     * Fits on multi-output targets.
     *
     * @param x the inputs {@code [n][d]}
     * @param y the targets {@code [n][k]}
     * @param settings the settings
     * @return the network
     */
    public static RandomWeightNetworks fit(double[][] x, double[][] y, Settings settings) {
        int n = x.length;
        int d = x[0].length;
        double[] mean = new double[d];
        double[] std = new double[d];
        standardisation(x, mean, std);
        double[][] z = apply(x, mean, std);
        SplittableRandom r = new SplittableRandom(settings.seed());
        List<RandomLayer> layers = new ArrayList<>();
        List<double[][]> hidden = new ArrayList<>();
        double[][] prev = null;
        for (int l = 0; l < settings.layers(); l++) {
            double[][] in = prev == null ? z : concat(z, prev);
            RandomLayer raw = draw(in[0].length, settings, r);
            double[][] h = activate(in, raw, settings.activation());
            double[] hm = new double[settings.hidden()];
            double[] hs = new double[settings.hidden()];
            boolean deep = settings.kind() == Kind.DEEP_RVFL || settings.kind() == Kind.ENSEMBLE_DEEP_RVFL;
            if (deep) {
                standardisation(h, hm, hs);
                h = apply(h, hm, hs);
            } else {
                java.util.Arrays.fill(hs, 1);
            }
            layers.add(new RandomLayer(raw.w(), raw.b(), hm, hs));
            hidden.add(h);
            prev = h;
        }
        List<double[][]> designs = designs(z, hidden, settings.kind());
        double chosen = settings.lambda();
        if (chosen <= 0) {
            chosen = choose(designs, y, settings.seed());
        }
        List<double[][]> readouts = new ArrayList<>();
        for (double[][] design : designs) {
            readouts.add(ridge(design, y, chosen));
        }
        return new RandomWeightNetworks(settings, mean, std, layers, readouts, chosen);
    }

    private static RandomLayer draw(int in, Settings s, SplittableRandom r) {
        double[][] w = new double[in][s.hidden()];
        double[] b = new double[s.hidden()];
        double a = s.scale() * Math.sqrt(3.0 / in);
        for (int i = 0; i < in; i++) {
            for (int j = 0; j < s.hidden(); j++) {
                w[i][j] = a * (2 * r.nextDouble() - 1);
            }
        }
        double c = s.scale();
        for (int j = 0; j < s.hidden(); j++) {
            b[j] = s.activation() == Activation.SIGMOID ? c * r.nextDouble() : c * (2 * r.nextDouble() - 1);
        }
        return new RandomLayer(w, b, null, null);
    }

    private static double[][] activate(double[][] in, RandomLayer layer, Activation g) {
        int n = in.length;
        int h = layer.b().length;
        double[][] out = new double[n][h];
        for (int i = 0; i < n; i++) {
            double[] row = in[i];
            double[] o = out[i];
            System.arraycopy(layer.b(), 0, o, 0, h);
            for (int k = 0; k < row.length; k++) {
                double v = row[k];
                if (v == 0) {
                    continue;
                }
                double[] wk = layer.w()[k];
                for (int j = 0; j < h; j++) {
                    o[j] += v * wk[j];
                }
            }
            for (int j = 0; j < h; j++) {
                o[j] = switch (g) {
                    case SIGMOID -> 1 / (1 + Math.exp(-o[j]));
                    case TANH -> Math.tanh(o[j]);
                    case RELU -> Math.max(0, o[j]);
                };
            }
        }
        return out;
    }

    private static List<double[][]> designs(double[][] z, List<double[][]> hidden, Kind kind) {
        List<double[][]> out = new ArrayList<>();
        switch (kind) {
            case ELM -> out.add(withBias(hidden.get(0)));
            case RVFL -> out.add(withBias(concat(z, hidden.get(0))));
            case DEEP_RVFL -> {
                double[][] all = z;
                for (double[][] h : hidden) {
                    all = concat(all, h);
                }
                out.add(withBias(all));
            }
            case ENSEMBLE_DEEP_RVFL -> {
                for (double[][] h : hidden) {
                    out.add(withBias(concat(z, h)));
                }
            }
            default -> throw new IllegalStateException(kind.toString());
        }
        return out;
    }

    /** Picks the ridge penalty with the smallest hold-out squared error, summed over the read-outs. */
    private static double choose(List<double[][]> designs, double[][] y, long seed) {
        int n = y.length;
        SplittableRandom r = new SplittableRandom(seed ^ 0x5DEECE66DL);
        boolean[] hold = new boolean[n];
        int m = 0;
        for (int i = 0; i < n; i++) {
            hold[i] = r.nextDouble() < 0.2;
            m += hold[i] ? 1 : 0;
        }
        if (m < 2 || m > n - 2) {
            return 1e-2;
        }
        double best = Double.POSITIVE_INFINITY;
        double chosen = 1e-2;
        double[] err = new double[LAMBDAS.length];
        for (double[][] design : designs) {
            double[][] trX = rows(design, hold, false);
            double[][] trY = rows(y, hold, false);
            double[][] teX = rows(design, hold, true);
            double[][] teY = rows(y, hold, true);
            double[][] gram = gram(trX);
            double[][] xty = crossProduct(trX, trY);
            for (int a = 0; a < LAMBDAS.length; a++) {
                double[][] beta = solve(gram, xty, LAMBDAS[a]);
                double[][] p = multiply(teX, beta);
                for (int i = 0; i < p.length; i++) {
                    for (int k = 0; k < p[i].length; k++) {
                        double e = p[i][k] - teY[i][k];
                        err[a] += e * e;
                    }
                }
            }
        }
        for (int a = 0; a < LAMBDAS.length; a++) {
            if (err[a] < best) {
                best = err[a];
                chosen = LAMBDAS[a];
            }
        }
        return chosen;
    }

    /**
     * A ridge read-out on given features, the solver the networks use: a bias column is appended,
     * and with {@code lambda <= 0} the penalty is chosen on a 20% hold-out.
     *
     * @param features the design {@code [n][p]}
     * @param y the targets {@code [n][k]}
     * @param lambda the penalty relative to the mean Gram diagonal, or {@code <= 0} to choose it
     * @param seed the hold-out seed
     * @return the coefficients {@code [p + 1][k]}, bias last
     */
    public static double[][] ridgeReadout(double[][] features, double[][] y, double lambda, long seed) {
        double[][] design = withBias(features);
        double chosen = lambda > 0 ? lambda : choose(List.<double[][]>of(design), y, seed);
        return ridge(design, y, chosen);
    }

    /**
     * Applies a read-out from {@link #ridgeReadout}.
     *
     * @param features the design {@code [n][p]}
     * @param beta the coefficients {@code [p + 1][k]}
     * @return {@code [n][k]}
     */
    public static double[][] applyReadout(double[][] features, double[][] beta) {
        return multiply(withBias(features), beta);
    }

    private static double[][] rows(double[][] x, boolean[] mask, boolean keep) {
        List<double[]> out = new ArrayList<>();
        for (int i = 0; i < x.length; i++) {
            if (mask[i] == keep) {
                out.add(x[i]);
            }
        }
        return out.toArray(new double[0][]);
    }

    /** The ridge read-out {@code (X^T X + lambda n I)^-1 X^T Y} (the bias column is penalised too, lightly). */
    private static double[][] ridge(double[][] x, double[][] y, double lambda) {
        return solve(gram(x), crossProduct(x, y), lambda);
    }

    private static double[][] solve(double[][] gram, double[][] xty, double lambda) {
        int p = gram.length;
        double[][] a = new double[p][];
        double trace = 0;
        for (int i = 0; i < p; i++) {
            trace += gram[i][i];
        }
        double ridge = lambda * Math.max(1e-12, trace / p);
        for (int i = 0; i < p; i++) {
            a[i] = gram[i].clone();
            a[i][i] += ridge;
        }
        double[][] l = MlMatrices.cholesky(a);
        return MlMatrices.choleskySolve(l, xty);
    }

    private static double[][] gram(double[][] x) {
        int p = x[0].length;
        double[][] g = new double[p][p];
        for (double[] row : x) {
            for (int i = 0; i < p; i++) {
                double v = row[i];
                if (v == 0) {
                    continue;
                }
                double[] gi = g[i];
                for (int j = i; j < p; j++) {
                    gi[j] += v * row[j];
                }
            }
        }
        for (int i = 0; i < p; i++) {
            for (int j = 0; j < i; j++) {
                g[i][j] = g[j][i];
            }
        }
        return g;
    }

    private static double[][] crossProduct(double[][] x, double[][] y) {
        int p = x[0].length;
        int k = y[0].length;
        double[][] out = new double[p][k];
        for (int i = 0; i < x.length; i++) {
            for (int a = 0; a < p; a++) {
                double v = x[i][a];
                for (int c = 0; c < k; c++) {
                    out[a][c] += v * y[i][c];
                }
            }
        }
        return out;
    }

    private static double[][] multiply(double[][] x, double[][] beta) {
        int k = beta[0].length;
        double[][] out = new double[x.length][k];
        for (int i = 0; i < x.length; i++) {
            for (int a = 0; a < beta.length; a++) {
                double v = x[i][a];
                if (v == 0) {
                    continue;
                }
                for (int c = 0; c < k; c++) {
                    out[i][c] += v * beta[a][c];
                }
            }
        }
        return out;
    }

    private static void standardisation(double[][] x, double[] mean, double[] std) {
        int n = x.length;
        for (double[] row : x) {
            for (int j = 0; j < mean.length; j++) {
                mean[j] += row[j] / n;
            }
        }
        for (double[] row : x) {
            for (int j = 0; j < mean.length; j++) {
                std[j] += (row[j] - mean[j]) * (row[j] - mean[j]) / n;
            }
        }
        for (int j = 0; j < std.length; j++) {
            std[j] = Math.sqrt(std[j]) + 1e-8;
        }
    }

    private static double[][] apply(double[][] x, double[] mean, double[] std) {
        double[][] out = new double[x.length][mean.length];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < mean.length; j++) {
                out[i][j] = (x[i][j] - mean[j]) / std[j];
            }
        }
        return out;
    }

    private static double[][] concat(double[][] a, double[][] b) {
        double[][] out = new double[a.length][a[0].length + b[0].length];
        for (int i = 0; i < a.length; i++) {
            System.arraycopy(a[i], 0, out[i], 0, a[i].length);
            System.arraycopy(b[i], 0, out[i], a[i].length, b[i].length);
        }
        return out;
    }

    private static double[][] withBias(double[][] a) {
        double[][] out = new double[a.length][a[0].length + 1];
        for (int i = 0; i < a.length; i++) {
            System.arraycopy(a[i], 0, out[i], 0, a[i].length);
            out[i][a[i].length] = 1;
        }
        return out;
    }

    /**
     * The raw outputs (class scores or regression values), averaged over the read-outs of the
     * ensemble variant.
     *
     * @param x the inputs {@code [n][d]}
     * @return {@code [n][k]}
     */
    public double[][] outputs(double[][] x) {
        double[][] z = apply(x, inMean, inStd);
        List<double[][]> hidden = new ArrayList<>();
        double[][] prev = null;
        for (RandomLayer layer : layers) {
            double[][] in = prev == null ? z : concat(z, prev);
            double[][] h = activate(in, layer, settings.activation());
            h = apply(h, layer.mean(), layer.std());
            hidden.add(h);
            prev = h;
        }
        List<double[][]> designs = designs(z, hidden, settings.kind());
        double[][] out = null;
        for (int m = 0; m < designs.size(); m++) {
            double[][] p = multiply(designs.get(m), readouts.get(m));
            if (out == null) {
                out = p;
            } else {
                for (int i = 0; i < p.length; i++) {
                    for (int k = 0; k < p[i].length; k++) {
                        out[i][k] += p[i][k];
                    }
                }
            }
        }
        double share = 1.0 / designs.size();
        for (double[] row : out) {
            for (int k = 0; k < row.length; k++) {
                row[k] *= share;
            }
        }
        return out;
    }

    /**
     * Predicts classes.
     *
     * @param x the inputs
     * @return the argmax of the outputs
     */
    public int[] predict(double[][] x) {
        double[][] o = outputs(x);
        int[] out = new int[o.length];
        for (int i = 0; i < o.length; i++) {
            int best = 0;
            for (int k = 1; k < o[i].length; k++) {
                best = o[i][k] > o[i][best] ? k : best;
            }
            out[i] = best;
        }
        return out;
    }

    /**
     * Predicts a single regression output.
     *
     * @param x the inputs
     * @return the first output column
     */
    public double[] predictValue(double[][] x) {
        double[][] o = outputs(x);
        double[] out = new double[o.length];
        for (int i = 0; i < o.length; i++) {
            out[i] = o[i][0];
        }
        return out;
    }

    /** @return the ridge penalty used (relative to the mean diagonal of the Gram matrix) */
    public double lambda() {
        return lambda;
    }

    /** @return the settings */
    public Settings settings() {
        return settings;
    }

    /** @return the number of read-out weights */
    public int readoutWeights() {
        int s = 0;
        for (double[][] b : readouts) {
            s += b.length * b[0].length;
        }
        return s;
    }
}
