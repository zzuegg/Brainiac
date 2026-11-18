package io.github.zzuegg.brainiac.experimental.explain;

import java.util.Arrays;

/**
 * Exact Shapley values of tree ensembles (Lundberg et al., "From local explanations to global
 * understanding with explainable AI for trees", Nature Machine Intelligence 2020,
 * <a href="https://arxiv.org/abs/1905.04610">arXiv:1905.04610</a>).
 *
 * <ul>
 *   <li>{@link Mode#PATH_DEPENDENT}: the game {@code v(S) = E[f(x) | x_S]} where the
 *       conditional expectation follows the trees' covers (the share of training rows down each
 *       branch). All subsets are pushed down each tree at once by tracking, along the decision
 *       path, the proportion of subsets of every size that flow down it ({@code extend} and
 *       {@code unwind} of the path weights), in {@code O(T L D^2)}.</li>
 *   <li>{@link Mode#INTERVENTIONAL}: the game {@code v(S) = mean_z f(x_S, z_{-S})} over
 *       background rows {@code z}. For one reference row, a leaf is reached by a coalition iff
 *       every split where {@code x} and {@code z} disagree is decided by the right player, so
 *       the leaf's value is split among the features that sent the path the {@code x} way
 *       ({@code +v |A|-1)! |B|! / (|A|+|B|)!}) and those that sent it the {@code z} way (the
 *       negative share), in {@code O(T L B)}.</li>
 *   <li>{@link #interactionValues}: the SHAP interaction matrix, path-dependent, from TreeSHAP
 *       runs conditioned on each feature being present or absent:
 *       {@code Phi_ij = (phi_i | j on - phi_i | j off) / 2}, the diagonal the remainder.</li>
 * </ul>
 * Every result satisfies local accuracy: the values sum to {@code f(x) - expectedValue}.
 */
public final class TreeShap {

    /** Which conditional expectation defines the game. */
    public enum Mode {
        /** Tree covers as the conditional distribution. */
        PATH_DEPENDENT,
        /** Averages over background rows (marginal, interventional). */
        INTERVENTIONAL
    }

    private final TreeView model;

    /**
     * Creates the explainer.
     *
     * @param model the ensemble, with covers for the path-dependent mode
     */
    public TreeShap(TreeView model) {
        this.model = model;
    }

    // ------------------------------------------------------------------ path-dependent

    /** The proportions of subsets of each size flowing down a unique path. */
    private static final class Path {
        final int[] d;
        final double[] z;
        final double[] o;
        final double[] w;
        int length;

        Path(int capacity) {
            d = new int[capacity];
            z = new double[capacity];
            o = new double[capacity];
            w = new double[capacity];
        }

        Path copy() {
            Path p = new Path(d.length);
            System.arraycopy(d, 0, p.d, 0, length);
            System.arraycopy(z, 0, p.z, 0, length);
            System.arraycopy(o, 0, p.o, 0, length);
            System.arraycopy(w, 0, p.w, 0, length);
            p.length = length;
            return p;
        }

        void extend(double pz, double po, int pi) {
            int l = length;
            d[l] = pi;
            z[l] = pz;
            o[l] = po;
            w[l] = l == 0 ? 1 : 0;
            for (int i = l - 1; i >= 0; i--) {
                w[i + 1] += po * w[i] * (i + 1) / (l + 1);
                w[i] = pz * w[i] * (l - i) / (l + 1);
            }
            length = l + 1;
        }

        void unwind(int i) {
            int l = length - 1;
            double n = w[l];
            double oi = o[i];
            double zi = z[i];
            for (int j = l - 1; j >= 0; j--) {
                if (oi != 0) {
                    double t = w[j];
                    w[j] = n * (l + 1) / ((j + 1) * oi);
                    n = t - w[j] * zi * (l - j) / (l + 1);
                } else {
                    w[j] = w[j] * (l + 1) / (zi * (l - j));
                }
            }
            for (int j = i; j < l; j++) {
                d[j] = d[j + 1];
                z[j] = z[j + 1];
                o[j] = o[j + 1];
            }
            length = l;
        }

        /** The total weight of the path with element {@code i} unwound, without changing it. */
        double unwoundSum(int i) {
            Path p = copy();
            p.unwind(i);
            double s = 0;
            for (int j = 0; j < p.length; j++) {
                s += p.w[j];
            }
            return s;
        }
    }

    /**
     * Shapley values.
     *
     * @param mode the game
     * @param x the row to explain
     * @param background the reference rows (interventional mode only; may be null otherwise)
     * @return one value per feature
     */
    public double[] values(Mode mode, double[] x, double[][] background) {
        return mode == Mode.PATH_DEPENDENT ? pathDependent(x, 0, -1) : interventional(x, background);
    }

    /**
     * The expected value the Shapley values are measured from.
     *
     * @param mode the game
     * @param background the reference rows (interventional mode)
     * @return the cover-weighted mean leaf value, or the mean prediction on the background
     */
    public double expectedValue(Mode mode, double[][] background) {
        if (mode == Mode.INTERVENTIONAL) {
            return Arrays.stream(model.predict(background)).average().orElse(0);
        }
        double s = 0;
        for (TreeView.Tree t : model.trees()) {
            for (int k = 0; k < t.nodes(); k++) {
                if (t.feature()[k] < 0) {
                    s += t.value()[k] * t.cover()[k] / t.cover()[0];
                }
            }
        }
        return model.base() + model.scale() * s;
    }

    private double[] pathDependent(double[] x, int condition, int conditionFeature) {
        double[] phi = new double[model.features()];
        for (TreeView.Tree t : model.trees()) {
            int cap = t.depth() + 2;
            recurse(t, x, phi, 0, new Path(cap), 1, 1, -1, condition, conditionFeature, 1);
        }
        for (int j = 0; j < phi.length; j++) {
            phi[j] *= model.scale();
        }
        return phi;
    }

    private static void recurse(TreeView.Tree t, double[] x, double[] phi, int node, Path parent, double pz,
            double po, int pi, int condition, int conditionFeature, double conditionFraction) {
        if (conditionFraction == 0) {
            return;
        }
        Path m = parent.copy();
        if (condition == 0 || conditionFeature != pi) {
            m.extend(pz, po, pi);
        }
        int split = t.feature()[node];
        if (split < 0) {
            for (int i = 1; i < m.length; i++) {
                double w = m.unwoundSum(i);
                phi[m.d[i]] += w * (m.o[i] - m.z[i]) * t.value()[node] * conditionFraction;
            }
            return;
        }
        int hot = x[split] <= t.threshold()[node] ? t.left()[node] : t.right()[node];
        int cold = hot == t.left()[node] ? t.right()[node] : t.left()[node];
        double cover = t.cover()[node];
        double hotZero = cover > 0 ? t.cover()[hot] / cover : 0.5;
        double coldZero = cover > 0 ? t.cover()[cold] / cover : 0.5;
        double iz = 1;
        double io = 1;
        for (int k = 1; k < m.length; k++) {
            if (m.d[k] == split) {
                iz = m.z[k];
                io = m.o[k];
                m.unwind(k);
                break;
            }
        }
        double hotFraction = conditionFraction;
        double coldFraction = conditionFraction;
        if (condition > 0 && split == conditionFeature) {
            coldFraction = 0;
        } else if (condition < 0 && split == conditionFeature) {
            hotFraction *= hotZero;
            coldFraction *= coldZero;
        }
        recurse(t, x, phi, hot, m, iz * hotZero, io, split, condition, conditionFeature, hotFraction);
        recurse(t, x, phi, cold, m, iz * coldZero, 0, split, condition, conditionFeature, coldFraction);
    }

    /**
     * The path-dependent SHAP interaction values.
     *
     * @param x the row
     * @return {@code [p][p]} symmetric; each row sums to the feature's Shapley value
     */
    public double[][] interactionValues(double[] x) {
        int p = model.features();
        double[] phi = pathDependent(x, 0, -1);
        double[][] out = new double[p][p];
        for (int j = 0; j < p; j++) {
            double[] on = pathDependent(x, 1, j);
            double[] off = pathDependent(x, -1, j);
            for (int i = 0; i < p; i++) {
                if (i != j) {
                    out[i][j] = (on[i] - off[i]) / 2;
                }
            }
        }
        for (int i = 0; i < p; i++) {
            double s = 0;
            for (int j = 0; j < p; j++) {
                if (j != i) {
                    s += out[i][j];
                }
            }
            out[i][i] = phi[i] - s;
        }
        return out;
    }

    // ------------------------------------------------------------------ interventional

    private double[] interventional(double[] x, double[][] background) {
        int p = model.features();
        double[] phi = new double[p];
        byte[] state = new byte[p];
        double[] factorial = factorials(p + 1);
        for (TreeView.Tree t : model.trees()) {
            for (double[] z : background) {
                walk(t, x, z, 0, state, 0, 0, phi, factorial);
            }
        }
        for (int j = 0; j < p; j++) {
            phi[j] *= model.scale() / background.length;
        }
        return phi;
    }

    private static double[] factorials(int n) {
        double[] f = new double[n + 1];
        f[0] = 1;
        for (int i = 1; i <= n; i++) {
            f[i] = f[i - 1] * i;
        }
        return f;
    }

    /** {@code state}: 0 undecided, 1 follows x (feature in S), 2 follows z (feature not in S). */
    private static void walk(TreeView.Tree t, double[] x, double[] z, int node, byte[] state, int nx, int nz,
            double[] phi, double[] factorial) {
        int f = t.feature()[node];
        if (f < 0) {
            double v = t.value()[node];
            int total = nx + nz;
            if (total == 0) {
                return;
            }
            double plus = nx > 0 ? v * factorial[nx - 1] * factorial[nz] / factorial[total] : 0;
            double minus = nz > 0 ? v * factorial[nx] * factorial[nz - 1] / factorial[total] : 0;
            for (int j = 0; j < state.length; j++) {
                if (state[j] == 1) {
                    phi[j] += plus;
                } else if (state[j] == 2) {
                    phi[j] -= minus;
                }
            }
            return;
        }
        double thr = t.threshold()[node];
        int xChild = x[f] <= thr ? t.left()[node] : t.right()[node];
        int zChild = z[f] <= thr ? t.left()[node] : t.right()[node];
        if (xChild == zChild) {
            walk(t, x, z, xChild, state, nx, nz, phi, factorial);
        } else if (state[f] == 1) {
            walk(t, x, z, xChild, state, nx, nz, phi, factorial);
        } else if (state[f] == 2) {
            walk(t, x, z, zChild, state, nx, nz, phi, factorial);
        } else {
            state[f] = 1;
            walk(t, x, z, xChild, state, nx + 1, nz, phi, factorial);
            state[f] = 2;
            walk(t, x, z, zChild, state, nx, nz + 1, phi, factorial);
            state[f] = 0;
        }
    }

    // ------------------------------------------------------------------ references

    /**
     * The path-dependent game: {@code E[f | x_S]} with the covers as the distribution (Algorithm
     * 1 of the paper, exponential in nothing but called once per coalition).
     *
     * @param x the row
     * @param present which features are in the coalition
     * @return the conditional expectation
     */
    public double conditionalExpectation(double[] x, boolean[] present) {
        double s = 0;
        for (TreeView.Tree t : model.trees()) {
            s += expect(t, x, present, 0);
        }
        return model.base() + model.scale() * s;
    }

    private static double expect(TreeView.Tree t, double[] x, boolean[] present, int node) {
        int f = t.feature()[node];
        if (f < 0) {
            return t.value()[node];
        }
        int l = t.left()[node];
        int r = t.right()[node];
        if (present[f]) {
            return expect(t, x, present, x[f] <= t.threshold()[node] ? l : r);
        }
        double c = t.cover()[node];
        if (c == 0) {
            return 0.5 * (expect(t, x, present, l) + expect(t, x, present, r));
        }
        return (t.cover()[l] * expect(t, x, present, l) + t.cover()[r] * expect(t, x, present, r)) / c;
    }

    /**
     * Brute-force Shapley values by enumerating all {@code 2^p} coalitions of a game, the
     * reference for small {@code p}.
     *
     * @param p the number of players
     * @param game the value of each coalition, given as a bit mask
     * @return the Shapley values
     */
    public static double[] bruteForce(int p, java.util.function.LongToDoubleFunction game) {
        long full = 1L << p;
        double[] v = new double[(int) full];
        for (long s = 0; s < full; s++) {
            v[(int) s] = game.applyAsDouble(s);
        }
        double[] factorial = factorials(p);
        double[] phi = new double[p];
        for (long s = 0; s < full; s++) {
            int size = Long.bitCount(s);
            for (int i = 0; i < p; i++) {
                if ((s & (1L << i)) == 0) {
                    double w = factorial[size] * factorial[p - size - 1] / factorial[p];
                    phi[i] += w * (v[(int) (s | (1L << i))] - v[(int) s]);
                }
            }
        }
        return phi;
    }

    /**
     * The interventional game of a black box: {@code v(S) = mean_z f(x_S, z_{-S})}.
     *
     * @param f the model
     * @param x the row
     * @param background the reference rows
     * @param mask the coalition as bits
     * @return its value
     */
    public static double interventionalGame(java.util.function.ToDoubleFunction<double[]> f, double[] x,
            double[][] background, long mask) {
        double s = 0;
        for (double[] z : background) {
            double[] row = z.clone();
            for (int j = 0; j < x.length; j++) {
                if ((mask & (1L << j)) != 0) {
                    row[j] = x[j];
                }
            }
            s += f.applyAsDouble(row);
        }
        return s / background.length;
    }
}
