package io.github.zzuegg.brainiac.experimental.ml;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * The multiclass Tsetlin machine (Granmo, 2018,
 * <a href="https://arxiv.org/abs/1804.01508">arXiv:1804.01508</a>) and the fuzzy-pattern
 * variant (Hnilov, 2025, <a href="https://arxiv.org/abs/2508.08350">arXiv:2508.08350</a>).
 *
 * <p>Each class owns {@code clauses} conjunctions over the {@code 2f} literals {@code x_k} and
 * {@code ¬x_k}; even clauses vote for the class and odd clauses against it. Every literal of every
 * clause has a Tsetlin automaton with {@code 2N} states, and the literal is included when its state
 * exceeds {@code N}. Literals and include masks are bit-packed 64 to a {@code long}, so a clause
 * evaluates as {@code popcount(include & ~literals) == 0}.
 *
 * <p>Training follows the paper: the target class's clauses get feedback with probability
 * {@code (T - v) / 2T} (type I for positive clauses, type II for negative), one random other class's
 * with {@code (T + v) / 2T} (the reverse), where {@code v} is the class vote clipped to
 * {@code [-T, T]}. Type I strengthens literals that are 1 in a firing clause with probability
 * {@code (s-1)/s} (type Ia) and forgets the others with {@code 1/s} (type Ib); type II includes
 * the 0-literals of a firing clause, so it stops firing on that false positive.
 *
 * <p>{@link Mode#FUZZY_PATTERN}: a clause with {@code m} included literals of which {@code f}
 * fail scores {@code max(0, v0 - f)} with {@code v0 = m} for {@code 0 < m <= LF} and {@code LF}
 * otherwise (an empty clause scores {@code LF}), so partly matching patterns still vote; a clause
 * "fires" for feedback when its score is positive, type Ia is deterministic while the clause has
 * fewer than {@code L} literals (and off beyond it), and type Ib keeps its {@code 1/s} (the paper
 * uses a much larger {@code s}, about the square of a standard machine's). The paper does not
 * spell out type II; here a firing fuzzy clause includes only as many of its excluded 0-literals
 * as it scores (just enough to silence it on that input), because the standard rule (all of them)
 * empties a fuzzy clause after one false positive. With {@code L > 0},
 * the standard mode honours the same literal budget.
 */
public final class TsetlinMachine {

    /** How a clause scores. */
    public enum Mode {
        /** a clause is 1 when every included literal holds */
        STANDARD,
        /** a clause scores {@code max(0, min(m, LF) - failed)} */
        FUZZY_PATTERN
    }

    /**
     * Settings.
     *
     * @param clauses clauses per class (even: half positive, half negative)
     * @param threshold the vote threshold {@code T}
     * @param specificity {@code s}
     * @param states automaton states per action {@code N}
     * @param mode the clause scoring
     * @param literalLimit {@code L}: type Ia adds no literal to a clause with this many (0 = no limit)
     * @param fuzzyLimit {@code LF}: the most a fuzzy clause can score
     * @param seed the seed
     */
    public record Settings(int clauses, int threshold, double specificity, int states, Mode mode, int literalLimit,
            int fuzzyLimit, long seed) {

        /**
         * The standard machine with 100 states per action and no literal budget.
         *
         * @param clauses clauses per class
         * @param threshold {@code T}
         * @param specificity {@code s}
         * @param seed the seed
         * @return the settings
         */
        public static Settings standard(int clauses, int threshold, double specificity, long seed) {
            return new Settings(clauses, threshold, specificity, 100, Mode.STANDARD, 0, 1, seed);
        }

        /**
         * The fuzzy-pattern machine.
         *
         * @param clauses clauses per class
         * @param threshold {@code T}, in units of clause scores
         * @param specificity {@code s}
         * @param literalLimit {@code L}
         * @param fuzzyLimit {@code LF}
         * @param seed the seed
         * @return the settings
         */
        public static Settings fuzzy(int clauses, int threshold, double specificity, int literalLimit,
                int fuzzyLimit, long seed) {
            return new Settings(clauses, threshold, specificity, 100, Mode.FUZZY_PATTERN, literalLimit, fuzzyLimit,
                    seed);
        }
    }

    private final int features;
    private final int classes;
    private final int literals;
    private final int words;
    private final Settings settings;
    private final short[][][] state;
    private final long[][][] include;
    private final int[][] count;
    private final SplittableRandom random;

    private TsetlinMachine(int features, int classes, Settings settings) {
        if (settings.clauses() < 2 || settings.clauses() % 2 != 0) {
            throw new IllegalArgumentException("clauses must be even and at least 2");
        }
        this.features = features;
        this.classes = classes;
        this.literals = 2 * features;
        this.words = (literals + 63) / 64;
        this.settings = settings;
        this.state = new short[classes][settings.clauses()][literals];
        this.include = new long[classes][settings.clauses()][words];
        this.count = new int[classes][settings.clauses()];
        this.random = new SplittableRandom(settings.seed());
        for (short[][] c : state) {
            for (short[] clause : c) {
                // every automaton starts on the exclude side of the boundary
                Arrays.fill(clause, (short) settings.states());
            }
        }
    }

    /**
     * Creates an untrained machine.
     *
     * @param features Boolean input features
     * @param classes the number of classes (2 or more)
     * @param settings the settings
     * @return the machine
     */
    public static TsetlinMachine create(int features, int classes, Settings settings) {
        return new TsetlinMachine(features, classes, settings);
    }

    /**
     * Packs {@code x} and its negation as literals.
     *
     * @param x Boolean features
     * @return the packed literals: bit {@code k} is {@code x_k}, bit {@code f + k} is {@code ¬x_k}
     */
    public long[] literals(boolean[] x) {
        long[] p = new long[words];
        for (int k = 0; k < features; k++) {
            int bit = x[k] ? k : features + k;
            p[bit >> 6] |= 1L << (bit & 63);
        }
        return p;
    }

    private int failed(long[] mask, long[] lit) {
        int f = 0;
        for (int j = 0; j < words; j++) {
            f += Long.bitCount(mask[j] & ~lit[j]);
        }
        return f;
    }

    /** The clause's score; {@code training} makes empty clauses fire. */
    private int score(int c, int j, long[] lit, boolean training) {
        int m = count[c][j];
        if (settings.mode() == Mode.STANDARD) {
            if (m == 0) {
                return training ? 1 : 0;
            }
            return failed(include[c][j], lit) == 0 ? 1 : 0;
        }
        int lf = settings.fuzzyLimit();
        if (m == 0) {
            return lf;
        }
        return Math.max(0, Math.min(m, lf) - failed(include[c][j], lit));
    }

    private int vote(int c, long[] lit, boolean training) {
        int v = 0;
        for (int j = 0; j < settings.clauses(); j++) {
            int s = score(c, j, lit, training);
            v += (j & 1) == 0 ? s : -s;
        }
        return v;
    }

    /**
     * The class votes (unclipped).
     *
     * @param x Boolean features
     * @return the vote per class
     */
    public int[] votes(boolean[] x) {
        long[] lit = literals(x);
        int[] v = new int[classes];
        for (int c = 0; c < classes; c++) {
            v[c] = vote(c, lit, false);
        }
        return v;
    }

    /**
     * The predicted class.
     *
     * @param x Boolean features
     * @return the class with the largest vote
     */
    public int predict(boolean[] x) {
        int[] v = votes(x);
        int best = 0;
        for (int c = 1; c < classes; c++) {
            if (v[c] > v[best]) {
                best = c;
            }
        }
        return best;
    }

    /**
     * The accuracy on labelled rows.
     *
     * @param x Boolean rows
     * @param y labels
     * @return the share predicted correctly
     */
    public double accuracy(boolean[][] x, int[] y) {
        int ok = 0;
        for (int i = 0; i < x.length; i++) {
            ok += predict(x[i]) == y[i] ? 1 : 0;
        }
        return ok / (double) x.length;
    }

    /**
     * Trains for some epochs over shuffled rows.
     *
     * @param x Boolean rows
     * @param y labels
     * @param epochs passes over the data
     */
    public void fit(boolean[][] x, int[] y, int epochs) {
        long[][] lit = new long[x.length][];
        for (int i = 0; i < x.length; i++) {
            lit[i] = literals(x[i]);
        }
        int[] order = new int[x.length];
        for (int i = 0; i < order.length; i++) {
            order[i] = i;
        }
        for (int e = 0; e < epochs; e++) {
            for (int i = order.length - 1; i > 0; i--) {
                int k = random.nextInt(i + 1);
                int t = order[i];
                order[i] = order[k];
                order[k] = t;
            }
            for (int i : order) {
                update(lit[i], y[i]);
            }
        }
    }

    private void update(long[] lit, int target) {
        int t = settings.threshold();
        int v = Math.clamp(vote(target, lit, true), -t, t);
        feedback(target, lit, (t - v) / (2.0 * t), true);
        int other = random.nextInt(classes - 1);
        if (other >= target) {
            other++;
        }
        int w = Math.clamp(vote(other, lit, true), -t, t);
        feedback(other, lit, (t + w) / (2.0 * t), false);
    }

    private void feedback(int c, long[] lit, double probability, boolean positiveClass) {
        for (int j = 0; j < settings.clauses(); j++) {
            if (random.nextDouble() >= probability) {
                continue;
            }
            boolean positiveClause = (j & 1) == 0;
            boolean fires = score(c, j, lit, true) > 0;
            if (positiveClause == positiveClass) {
                typeOne(c, j, lit, fires);
            } else if (fires) {
                typeTwo(c, j, lit);
            }
        }
    }

    private boolean bit(long[] p, int k) {
        return (p[k >> 6] >>> (k & 63) & 1L) != 0;
    }

    private void typeOne(int c, int j, long[] lit, boolean fires) {
        double forget = 1 / settings.specificity();
        double strengthen = 1 - forget;
        boolean fuzzy = settings.mode() == Mode.FUZZY_PATTERN;
        int limit = settings.literalLimit();
        if (!fires) {
            // type Ib on every literal: skip ahead geometrically between the 1/s events
            for (int k = skip(forget, -1); k < literals; k = skip(forget, k)) {
                step(c, j, k, -1);
            }
            return;
        }
        // start at a random literal so a literal budget is not always spent on the lowest indices
        int offset = random.nextInt(literals);
        for (int n = 0; n < literals; n++) {
            int k = offset + n < literals ? offset + n : offset + n - literals;
            if (bit(lit, k)) {
                boolean under = limit <= 0 || count[c][j] < limit;
                boolean grow = fuzzy ? under : under && random.nextDouble() < strengthen;
                if (grow) {
                    step(c, j, k, +1);
                }
            } else if (random.nextDouble() < forget) {
                step(c, j, k, -1);
            }
        }
    }

    private int skip(double p, int from) {
        if (p >= 1) {
            return from + 1;
        }
        double u = random.nextDouble();
        return from + 1 + (int) Math.floor(Math.log1p(-u) / Math.log1p(-p));
    }

    private void typeTwo(int c, int j, long[] lit) {
        // standard: every excluded 0-literal moves toward include; fuzzy: only as many as the
        // clause scores, which is just enough to silence it on this false positive
        int budget = settings.mode() == Mode.STANDARD ? literals : score(c, j, lit, true);
        int offset = random.nextInt(literals);
        for (int n = 0; n < literals && budget > 0; n++) {
            int k = offset + n < literals ? offset + n : offset + n - literals;
            if (!bit(lit, k) && !bit(include[c][j], k)) {
                step(c, j, k, +1);
                budget--;
            }
        }
    }

    private void step(int c, int j, int k, int delta) {
        int n = settings.states();
        int before = state[c][j][k];
        int after = Math.clamp(before + delta, 1, 2 * n);
        if (after == before) {
            return;
        }
        state[c][j][k] = (short) after;
        boolean was = before > n;
        boolean is = after > n;
        if (was != is) {
            include[c][j][k >> 6] ^= 1L << (k & 63);
            count[c][j] += is ? 1 : -1;
        }
    }

    /**
     * The literals a clause includes.
     *
     * @param c the class
     * @param j the clause (even: votes for {@code c})
     * @return literal indices: {@code k < f} is {@code x_k}, {@code f + k} is {@code ¬x_k}
     */
    public int[] included(int c, int j) {
        int[] out = new int[count[c][j]];
        int n = 0;
        for (int k = 0; k < literals; k++) {
            if (bit(include[c][j], k)) {
                out[n++] = k;
            }
        }
        return out;
    }

    /**
     * A clause as a readable rule.
     *
     * @param c the class
     * @param j the clause
     * @return such as {@code "+ x0 & !x3"}, or {@code "+ true"} when empty
     */
    public String describe(int c, int j) {
        List<String> parts = new ArrayList<>();
        for (int k : included(c, j)) {
            parts.add(k < features ? "x" + k : "!x" + (k - features));
        }
        return ((j & 1) == 0 ? "+ " : "- ") + (parts.isEmpty() ? "true" : String.join(" & ", parts));
    }

    /** @return clauses per class */
    public int clauses() {
        return settings.clauses();
    }

    /** @return the number of classes */
    public int classes() {
        return classes;
    }

    /** @return the number of Boolean features */
    public int features() {
        return features;
    }

    /**
     * Thermometer encoding of continuous features: feature {@code d} becomes {@code levels} bits
     * {@code x_d > q_l} for training-set quantiles {@code q_1 < ... < q_levels}.
     */
    public static final class Thermometer {

        private final double[][] thresholds;

        private Thermometer(double[][] thresholds) {
            this.thresholds = thresholds;
        }

        /**
         * Fits per-feature quantile thresholds.
         *
         * @param x continuous rows
         * @param levels bits per feature
         * @return the encoder
         */
        public static Thermometer fit(double[][] x, int levels) {
            int d = x[0].length;
            double[][] q = new double[d][levels];
            double[] col = new double[x.length];
            for (int f = 0; f < d; f++) {
                for (int i = 0; i < x.length; i++) {
                    col[i] = x[i][f];
                }
                Arrays.sort(col);
                for (int l = 0; l < levels; l++) {
                    q[f][l] = col[(int) ((l + 1) * (long) (x.length - 1) / (levels + 1))];
                }
            }
            return new Thermometer(q);
        }

        /** @return Boolean features per row */
        public int bits() {
            return thresholds.length * thresholds[0].length;
        }

        /**
         * @param row a continuous row
         * @return its thermometer bits
         */
        public boolean[] encode(double[] row) {
            int levels = thresholds[0].length;
            boolean[] out = new boolean[bits()];
            for (int f = 0; f < thresholds.length; f++) {
                for (int l = 0; l < levels; l++) {
                    out[f * levels + l] = row[f] > thresholds[f][l];
                }
            }
            return out;
        }

        /**
         * @param x continuous rows
         * @return their thermometer bits
         */
        public boolean[][] encode(double[][] x) {
            boolean[][] out = new boolean[x.length][];
            for (int i = 0; i < x.length; i++) {
                out[i] = encode(x[i]);
            }
            return out;
        }
    }
}
