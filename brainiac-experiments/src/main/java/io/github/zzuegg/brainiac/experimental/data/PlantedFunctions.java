package io.github.zzuegg.brainiac.experimental.data;

import java.util.Arrays;
import java.util.SplittableRandom;

/**
 * Tabular data with a known generating function, the ground truth that explanation methods are
 * scored against.
 *
 * <ul>
 *   <li>{@link #additive}: five additive shapes (linear, quadratic, step, sine, saturating) on
 *       {@code x0..x4} and a planted {@code x5 x6} interaction, with Toeplitz-correlated
 *       Gaussian inputs ({@code corr(x_i, x_j) = rho^|i-j|}) so that partial dependence has to
 *       extrapolate off the data at large {@code rho}; {@link #shape} gives each planted
 *       {@code g_j}.</li>
 *   <li>{@link #interactions}: additive terms plus the planted pairs {@code (0,1)}, {@code (2,3)}
 *       and the three-way term {@code (4,5,6)}, padded with noise features.</li>
 *   <li>{@link #rule}: a binary label that follows {@code x1 > 0.5 and x3 < 0}.</li>
 *   <li>{@link #ruleList}, {@link #xorTree}, {@link #redundant}: binary features with a planted
 *       rule list, an XOR-like depth-3 tree (which greedy splitting cannot find) and noisy
 *       copies of the relevant features (many equally good models).</li>
 *   <li>{@link #credit}: credit-style rows with immutable, increase-only and decrease-only
 *       features and their per-unit costs, labelled by a known linear rule, for recourse.</li>
 *   <li>{@link #score}: binary features whose true log-odds is a sparse integer score.</li>
 *   <li>{@link #varyingCoefficients}: {@code y = sum_j beta_j(z) x_j} with known local
 *       coefficients.</li>
 *   <li>{@link #leaky}: a classification task with a feature that leaks the label in training
 *       but not at test time.</li>
 * </ul>
 */
public final class PlantedFunctions {

    private PlantedFunctions() {
    }

    /**
     * A generated table.
     *
     * @param x the rows {@code [n][p]}
     * @param y one target per row (0/1 for classification)
     */
    public record Table(double[][] x, double[] y) {

        /** @return the number of rows */
        public int rows() {
            return x.length;
        }

        /** @return the number of features */
        public int features() {
            return x[0].length;
        }

        /**
         * @param from the first row
         * @param to one past the last row
         * @return those rows
         */
        public Table slice(int from, int to) {
            return new Table(Arrays.copyOfRange(x, from, to), Arrays.copyOfRange(y, from, to));
        }
    }

    /**
     * Standard normal rows with Toeplitz correlation {@code rho^|i-j|} (an AR(1) chain over the
     * columns).
     *
     * @param n rows
     * @param p columns
     * @param rho the neighbour correlation
     * @param random the random source
     * @return the rows
     */
    public static double[][] gaussian(int n, int p, double rho, SplittableRandom random) {
        double[][] x = new double[n][p];
        double s = Math.sqrt(1 - rho * rho);
        for (double[] row : x) {
            row[0] = random.nextGaussian();
            for (int j = 1; j < p; j++) {
                row[j] = rho * row[j - 1] + s * random.nextGaussian();
            }
        }
        return x;
    }

    /**
     * The planted additive shape of feature {@code j} in {@link #additive}, each with mean 0
     * under a standard normal input.
     *
     * @param j the feature, 0 to 4
     * @param v its value
     * @return {@code v}, {@code v^2 - 1}, {@code sign(v)}, {@code sin(2v)} or {@code tanh(2v)}
     */
    public static double shape(int j, double v) {
        return switch (j) {
            case 0 -> v;
            case 1 -> v * v - 1;
            case 2 -> v > 0 ? 1 : -1;
            case 3 -> Math.sin(2 * v);
            case 4 -> Math.tanh(2 * v);
            default -> 0;
        };
    }

    /**
     * The noiseless function of {@link #additive}.
     *
     * @param x a row of at least 7 features
     * @return {@code sum_{j<5} g_j(x_j) + x5 x6}
     */
    public static double additiveTruth(double[] x) {
        double s = x[5] * x[6];
        for (int j = 0; j < 5; j++) {
            s += shape(j, x[j]);
        }
        return s;
    }

    /**
     * Five additive shapes on {@code x0..x4} and a planted {@code x5 x6} interaction, plus
     * Gaussian noise of standard deviation 0.1; 7 features.
     *
     * @param n rows
     * @param rho the neighbour correlation of the inputs, such as 0, 0.5 or 0.9
     * @param seed the seed
     * @return the table
     */
    public static Table additive(int n, double rho, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][] x = gaussian(n, 7, rho, r);
        double[] y = new double[n];
        for (int i = 0; i < n; i++) {
            y[i] = additiveTruth(x[i]) + 0.1 * r.nextGaussian();
        }
        return new Table(x, y);
    }

    /** The planted interacting pairs of {@link #interactions}, including the pairs of its triple. */
    public static final int[][] PLANTED_PAIRS = {{0, 1}, {2, 3}, {4, 5}, {4, 6}, {5, 6}};

    /**
     * The noiseless function of {@link #interactions}.
     *
     * @param x a row of at least 7 features
     * @return {@code 0.5 x0 + 0.5 x2 + x0 x1 + x2 x3 + 2 x4 x5 x6}
     */
    public static double interactionTruth(double[] x) {
        return 0.5 * x[0] + 0.5 * x[2] + x[0] * x[1] + x[2] * x[3] + 2 * x[4] * x[5] * x[6];
    }

    /**
     * Uniform inputs on {@code [-1, 1]}, the target {@link #interactionTruth} plus noise of
     * standard deviation 0.05; features 7 and up are noise.
     *
     * @param n rows
     * @param p features, at least 7
     * @param seed the seed
     * @return the table
     */
    public static Table interactions(int n, int p, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][] x = new double[n][p];
        double[] y = new double[n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < p; j++) {
                x[i][j] = 2 * r.nextDouble() - 1;
            }
            y[i] = interactionTruth(x[i]) + 0.05 * r.nextGaussian();
        }
        return new Table(x, y);
    }

    /**
     * A binary label that follows {@code x1 > 0.5 and x3 < 0} on standard normal inputs, with
     * the label flipped with probability {@code noise}.
     *
     * @param n rows
     * @param p features, at least 4
     * @param noise the label-flip rate
     * @param seed the seed
     * @return the table
     */
    public static Table rule(int n, int p, double noise, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][] x = gaussian(n, p, 0, r);
        double[] y = new double[n];
        for (int i = 0; i < n; i++) {
            boolean v = x[i][1] > 0.5 && x[i][3] < 0;
            y[i] = (v ^ r.nextDouble() < noise) ? 1 : 0;
        }
        return new Table(x, y);
    }

    private static double[][] bits(int n, int p, SplittableRandom r) {
        double[][] x = new double[n][p];
        for (double[] row : x) {
            for (int j = 0; j < p; j++) {
                row[j] = r.nextBoolean() ? 1 : 0;
            }
        }
        return x;
    }

    /**
     * The planted rule list of {@link #ruleList}: if {@code b0 and b1} then 1, else if
     * {@code b2 and not b3} then 0, else if {@code b4} then 1, else 0.
     *
     * @param x a row of binary features
     * @return the label
     */
    public static double ruleListTruth(double[] x) {
        if (x[0] > 0.5 && x[1] > 0.5) {
            return 1;
        }
        if (x[2] > 0.5 && x[3] < 0.5) {
            return 0;
        }
        return x[4] > 0.5 ? 1 : 0;
    }

    /**
     * Uniform binary features labelled by {@link #ruleListTruth}, flipped with probability
     * {@code noise}.
     *
     * @param n rows
     * @param p binary features, at least 5
     * @param noise the label-flip rate
     * @param seed the seed
     * @return the table
     */
    public static Table ruleList(int n, int p, double noise, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][] x = bits(n, p, r);
        double[] y = new double[n];
        for (int i = 0; i < n; i++) {
            y[i] = ruleListTruth(x[i]) != 0 ^ r.nextDouble() < noise ? 1 : 0;
        }
        return new Table(x, y);
    }

    /**
     * The planted depth-3 tree of {@link #xorTree}: {@code b0 xor b1}, and where that is true
     * {@code b2}, else {@code not b3}. Its root split alone has no marginal signal, so greedy
     * splitting misses it.
     *
     * @param x a row of binary features
     * @return the label
     */
    public static double xorTreeTruth(double[] x) {
        boolean a = x[0] > 0.5 ^ x[1] > 0.5;
        return (a ? x[2] > 0.5 : x[3] < 0.5) ? 1 : 0;
    }

    /**
     * Binary features labelled by {@link #xorTreeTruth} with label noise, where the noise
     * features are weakly correlated with the label (each agrees with it with probability
     * {@code 0.5 + lure}) so that a greedy splitter is lured away from the XOR root.
     *
     * @param n rows
     * @param p binary features, at least 6
     * @param noise the label-flip rate
     * @param lure the extra agreement of the decoy features (4 and up) with the label
     * @param seed the seed
     * @return the table
     */
    public static Table xorTree(int n, int p, double noise, double lure, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][] x = bits(n, p, r);
        double[] y = new double[n];
        for (int i = 0; i < n; i++) {
            y[i] = xorTreeTruth(x[i]) != 0 ^ r.nextDouble() < noise ? 1 : 0;
            for (int j = 4; j < p; j++) {
                x[i][j] = r.nextDouble() < 0.5 + lure ? y[i] : 1 - y[i];
            }
        }
        return new Table(x, y);
    }

    /**
     * Binary features where the label is {@code b0 and b1} (with noise) and features 2 and 3 are
     * noisy copies of 0 and 1 (each bit flipped with probability {@code copyNoise}), so several
     * near-equally good trees exist; the rest are noise.
     *
     * @param n rows
     * @param p binary features, at least 5
     * @param noise the label-flip rate
     * @param copyNoise the flip rate of the copies
     * @param seed the seed
     * @return the table
     */
    public static Table redundant(int n, int p, double noise, double copyNoise, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][] x = bits(n, p, r);
        double[] y = new double[n];
        for (int i = 0; i < n; i++) {
            x[i][2] = r.nextDouble() < copyNoise ? 1 - x[i][0] : x[i][0];
            x[i][3] = r.nextDouble() < copyNoise ? 1 - x[i][1] : x[i][1];
            boolean v = x[i][0] > 0.5 && x[i][1] > 0.5;
            y[i] = v ^ r.nextDouble() < noise ? 1 : 0;
        }
        return new Table(x, y);
    }

    /** How a credit feature may change in a recourse. */
    public enum Actionability {
        /** May not change (age-like). */
        IMMUTABLE,
        /** May only increase. */
        INCREASE,
        /** May only decrease. */
        DECREASE,
        /** May change either way. */
        FREE
    }

    /** The credit features' actionability: age, income, debt, savings, education, tenure. */
    public static final Actionability[] CREDIT_ACTIONS = {Actionability.IMMUTABLE, Actionability.INCREASE,
        Actionability.DECREASE, Actionability.FREE, Actionability.INCREASE, Actionability.IMMUTABLE};

    /** The credit features' cost per standard deviation of change. */
    public static final double[] CREDIT_COSTS = {Double.POSITIVE_INFINITY, 1.0, 1.0, 1.5, 3.0,
        Double.POSITIVE_INFINITY};

    /** The weights of the credit rule {@code w x + b > 0}. */
    public static final double[] CREDIT_WEIGHTS = {1.2, 1.0, -1.0, 0.6, 0.5, 0.8};

    /**
     * Credit-style rows: six correlated standardized features (age, income, debt, savings,
     * education, tenure) approved when {@code w x - 0.5 > 0} with logistic label noise of scale
     * {@code 0.3}. Age carries the largest weight, so an unconstrained counterfactual changes it.
     *
     * @param n rows
     * @param seed the seed
     * @return the table
     */
    public static Table credit(int n, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][] x = gaussian(n, 6, 0.3, r);
        double[] y = new double[n];
        for (int i = 0; i < n; i++) {
            double z = -0.5;
            for (int j = 0; j < 6; j++) {
                z += CREDIT_WEIGHTS[j] * x[i][j];
            }
            double u = r.nextDouble();
            z += 0.3 * Math.log(u / (1 - u));
            y[i] = z > 0 ? 1 : 0;
        }
        return new Table(x, y);
    }

    /** The planted integer points of {@link #score}, on features 0 to 4; the rest score 0. */
    public static final int[] SCORE_POINTS = {3, -2, 2, 1, -3};

    /** The planted intercept of {@link #score}. */
    public static final int SCORE_INTERCEPT = -1;

    /**
     * Binary features whose log-odds is {@code SCORE_INTERCEPT + sum_j SCORE_POINTS[j] b_j}; with
     * {@code nonlinear} the log-odds gets an extra {@code 2 b5 b6} term no additive score fits.
     *
     * @param n rows
     * @param p binary features, at least 7
     * @param nonlinear whether to add the interaction
     * @param seed the seed
     * @return the table
     */
    public static Table score(int n, int p, boolean nonlinear, long seed) {
        return score(n, p, nonlinear, 1, seed);
    }

    /**
     * As {@link #score(int, int, boolean, long)} with the log-odds multiplied by {@code scale}, so
     * that at a scale other than 1 the best coefficients are not integers and rounding them loses.
     *
     * @param n rows
     * @param p binary features, at least 7
     * @param nonlinear whether to add the interaction
     * @param scale the factor on the log-odds
     * @param seed the seed
     * @return the table
     */
    public static Table score(int n, int p, boolean nonlinear, double scale, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][] x = new double[n][p];
        double[] y = new double[n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < p; j++) {
                x[i][j] = r.nextDouble() < 0.4 ? 1 : 0;
            }
            double z = SCORE_INTERCEPT;
            for (int j = 0; j < SCORE_POINTS.length; j++) {
                z += SCORE_POINTS[j] * x[i][j];
            }
            if (nonlinear) {
                z += 2 * x[i][5] * x[i][6];
            }
            y[i] = r.nextDouble() < 1 / (1 + Math.exp(-scale * z)) ? 1 : 0;
        }
        return new Table(x, y);
    }

    /**
     * The local coefficients of {@link #varyingCoefficients}: {@code beta_0 = 1 + z},
     * {@code beta_1 = sin(2 z)}, {@code beta_2 = -1}, {@code beta_3 = z^2 - 0.5}.
     *
     * @param z the modifier
     * @return the four coefficients
     */
    public static double[] coefficients(double z) {
        return new double[] {1 + z, Math.sin(2 * z), -1, z * z - 0.5};
    }

    /**
     * A varying-coefficient model {@code y = sum_j beta_j(z) x_j + 0.05 noise}: the first column
     * is the modifier {@code z} uniform on {@code [-1, 1]}, the next four are standard normal
     * {@code x_j}.
     *
     * @param n rows
     * @param seed the seed
     * @return the table, 5 features
     */
    public static Table varyingCoefficients(int n, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][] x = new double[n][5];
        double[] y = new double[n];
        for (int i = 0; i < n; i++) {
            x[i][0] = 2 * r.nextDouble() - 1;
            double[] b = coefficients(x[i][0]);
            double s = 0;
            for (int j = 0; j < 4; j++) {
                x[i][j + 1] = r.nextGaussian();
                s += b[j] * x[i][j + 1];
            }
            y[i] = s + 0.05 * r.nextGaussian();
        }
        return new Table(x, y);
    }

    /**
     * Classification with a leaky feature: the label is {@code x0 + x1 - x2 > 0} (with 5%
     * flips), and the last feature equals {@code 2 y - 1} plus small noise with probability
     * {@code leak} and is pure noise otherwise.
     *
     * @param n rows
     * @param p features, at least 4; the last one leaks
     * @param leak how often the leaky feature tells the label, such as 0.95 in training, 0 at test
     * @param seed the seed
     * @return the table
     */
    public static Table leaky(int n, int p, double leak, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][] x = gaussian(n, p, 0, r);
        double[] y = new double[n];
        for (int i = 0; i < n; i++) {
            boolean v = x[i][0] + x[i][1] - x[i][2] > 0;
            y[i] = v ^ r.nextDouble() < 0.05 ? 1 : 0;
            x[i][p - 1] = r.nextDouble() < leak ? 2 * y[i] - 1 + 0.1 * r.nextGaussian() : r.nextGaussian();
        }
        return new Table(x, y);
    }
}
