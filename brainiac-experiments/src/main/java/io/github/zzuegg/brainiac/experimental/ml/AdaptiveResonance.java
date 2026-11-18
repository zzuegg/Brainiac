package io.github.zzuegg.brainiac.experimental.ml;

import java.util.ArrayList;
import java.util.List;

/**
 * Fuzzy adaptive resonance theory (Carpenter, Grossberg and Rosen, Neural Networks 1991) and Fuzzy
 * ARTMAP (Carpenter et al., IEEE TNN 1992): online category learning without catastrophic
 * forgetting.
 *
 * <p>Inputs in {@code [0, 1]^d} are complement coded, {@code I = [a, 1 - a]}. A category {@code j}
 * with weight {@code w_j} has the choice value {@code T_j = |I ^ w_j| / (alpha + |w_j|)} ({@code ^}
 * the element-wise minimum, {@code |.|} the L1 norm). Categories are tried in order of {@code T_j};
 * the first whose match {@code |I ^ w_j| / |I|} reaches the vigilance {@code rho} resonates and
 * learns {@code w_j <- beta (I ^ w_j) + (1 - beta) w_j} ({@code beta = 1}: fast learning); if none
 * resonates, a new category {@code w = I} is created. Learning only shrinks a category's box, so what
 * was learnt is never overwritten by later inputs elsewhere.
 *
 * <p>{@link Mode#FUZZY_ARTMAP} adds a label to every category (the map field). When the resonating
 * category predicts the wrong label, match tracking raises the vigilance to just above that
 * category's match ({@code + epsilon}) and the search continues, so the network creates finer
 * categories only where classes collide. Training is one pass and incremental:
 * {@link #learn(double[], int)} can be called at any time, with new classes too.
 */
public final class AdaptiveResonance {

    /** The network. */
    public enum Mode {
        /** Unsupervised Fuzzy ART: {@link #learn(double[], int)} ignores the label. */
        FUZZY_ART,
        /** Supervised Fuzzy ARTMAP with match tracking. */
        FUZZY_ARTMAP
    }

    private final Mode mode;
    private final int inputs;
    private final double vigilance;
    private final double choice;
    private final double learningRate;
    private final double epsilon;
    private final List<double[]> weights = new ArrayList<>();
    private final List<double[]> norms = new ArrayList<>();
    private final List<Integer> labels = new ArrayList<>();
    private int resets;

    /**
     * Creates an empty network.
     *
     * @param mode the network
     * @param inputs the input dimension {@code d} (before complement coding)
     * @param vigilance the baseline vigilance {@code rho} in {@code [0, 1]}
     * @param choice the choice parameter {@code alpha}, such as 0.001
     * @param learningRate {@code beta} in {@code (0, 1]}; 1 is fast learning
     * @param epsilon the match-tracking increment, such as 0.001
     */
    public AdaptiveResonance(Mode mode, int inputs, double vigilance, double choice, double learningRate,
            double epsilon) {
        if (vigilance < 0 || vigilance > 1 || learningRate <= 0 || learningRate > 1) {
            throw new IllegalArgumentException("vigilance in [0, 1], learning rate in (0, 1]");
        }
        this.mode = mode;
        this.inputs = inputs;
        this.vigilance = vigilance;
        this.choice = choice;
        this.learningRate = learningRate;
        this.epsilon = epsilon;
    }

    /**
     * Complement codes a point.
     *
     * @param a a point in {@code [0, 1]^d} (clipped)
     * @return {@code [a, 1 - a]}
     */
    public static double[] complement(double[] a) {
        double[] out = new double[2 * a.length];
        for (int i = 0; i < a.length; i++) {
            double v = Math.min(1, Math.max(0, a[i]));
            out[i] = v;
            out[a.length + i] = 1 - v;
        }
        return out;
    }

    /**
     * A min-max scaler to {@code [0, 1]} fitted on some rows (ART needs bounded inputs).
     *
     * @param x rows
     * @return {@code {min, max}} per column
     */
    public static double[][] range(double[][] x) {
        int d = x[0].length;
        double[] lo = new double[d];
        double[] hi = new double[d];
        java.util.Arrays.fill(lo, Double.POSITIVE_INFINITY);
        java.util.Arrays.fill(hi, Double.NEGATIVE_INFINITY);
        for (double[] row : x) {
            for (int j = 0; j < d; j++) {
                lo[j] = Math.min(lo[j], row[j]);
                hi[j] = Math.max(hi[j], row[j]);
            }
        }
        return new double[][] {lo, hi};
    }

    /**
     * Scales a row with a range from {@link #range}.
     *
     * @param row a row
     * @param range the range
     * @return the row in {@code [0, 1]} (clipped)
     */
    public static double[] scale(double[] row, double[][] range) {
        double[] out = new double[row.length];
        for (int j = 0; j < row.length; j++) {
            double w = range[1][j] - range[0][j];
            out[j] = w <= 0 ? 0.5 : Math.min(1, Math.max(0, (row[j] - range[0][j]) / w));
        }
        return out;
    }

    private static double minNorm(double[] a, double[] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) {
            s += Math.min(a[i], b[i]);
        }
        return s;
    }

    /**
     * Presents one input and learns from it.
     *
     * @param x a point in {@code [0, 1]^d}
     * @param label its class (ignored by Fuzzy ART)
     * @return the category that resonated (possibly a new one)
     */
    public int learn(double[] x, int label) {
        if (x.length != inputs) {
            throw new IllegalArgumentException("expected " + inputs + " inputs");
        }
        double[] in = complement(x);
        double size = inputs; // |I| = d under complement coding
        int c = weights.size();
        double[] t = new double[c];
        double[] match = new double[c];
        Integer[] order = new Integer[c];
        for (int j = 0; j < c; j++) {
            double m = minNorm(in, weights.get(j));
            t[j] = m / (choice + norms.get(j)[0]);
            match[j] = m / size;
            order[j] = j;
        }
        java.util.Arrays.sort(order, (a, b) -> Double.compare(t[b], t[a]));
        double rho = vigilance;
        for (int j : order) {
            if (match[j] < rho) {
                continue;
            }
            if (mode == Mode.FUZZY_ARTMAP && labels.get(j) != label) {
                resets++;
                rho = match[j] + epsilon;
                continue;
            }
            double[] w = weights.get(j);
            double norm = 0;
            for (int i = 0; i < w.length; i++) {
                w[i] = learningRate * Math.min(in[i], w[i]) + (1 - learningRate) * w[i];
                norm += w[i];
            }
            norms.get(j)[0] = norm;
            return j;
        }
        weights.add(in);
        norms.add(new double[] {size});
        labels.add(label);
        return c;
    }

    /**
     * The best-matching category without learning (no vigilance test, as in ARTMAP prediction).
     *
     * @param x a point in {@code [0, 1]^d}
     * @return the category, or {@code -1} if there is none
     */
    public int category(double[] x) {
        double[] in = complement(x);
        int best = -1;
        double bt = Double.NEGATIVE_INFINITY;
        for (int j = 0; j < weights.size(); j++) {
            double tj = minNorm(in, weights.get(j)) / (choice + norms.get(j)[0]);
            if (tj > bt) {
                bt = tj;
                best = j;
            }
        }
        return best;
    }

    /**
     * Predicts a class (Fuzzy ARTMAP) or the category's first label (Fuzzy ART).
     *
     * @param x a point in {@code [0, 1]^d}
     * @return the label, or {@code -1} before any learning
     */
    public int predict(double[] x) {
        int j = category(x);
        return j < 0 ? -1 : labels.get(j);
    }

    /**
     * Predicts many points.
     *
     * @param x points in {@code [0, 1]^d}
     * @return the labels
     */
    public int[] predict(double[][] x) {
        int[] out = new int[x.length];
        for (int i = 0; i < x.length; i++) {
            out[i] = predict(x[i]);
        }
        return out;
    }

    /** @return the number of categories */
    public int categories() {
        return weights.size();
    }

    /** @return how often match tracking reset a category with the wrong label */
    public int resets() {
        return resets;
    }

    /**
     * The category boxes as complement-coded weights (a copy).
     *
     * @return {@code [categories][2d]}
     */
    public double[][] weights() {
        double[][] out = new double[weights.size()][];
        for (int j = 0; j < out.length; j++) {
            out[j] = weights.get(j).clone();
        }
        return out;
    }

    /** @return the label of each category */
    public int[] categoryLabels() {
        return labels.stream().mapToInt(Integer::intValue).toArray();
    }
}
