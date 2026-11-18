package io.github.zzuegg.brainiac.experimental.anomaly;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * The matrix profile by STOMP (Yeh et al., ICDM 2016; Zhu et al., ICDM 2016; see the
 * <a href="https://www.cs.ucr.edu/~eamonn/MatrixProfile.html">UCR Matrix Profile page</a>):
 * for every subsequence of length {@code m}, the z-normalised Euclidean distance to its
 * nearest non-trivial match (outside an exclusion zone of {@code m / 4} around itself).
 * STOMP computes it exactly in {@code O(n^2)} time and {@code O(n)} memory: the dot products
 * of row {@code i} follow from row {@code i - 1} by one subtraction and one addition each
 * ({@code QT_i,j = QT_i-1,j-1 - T_i-1 T_j-1 + T_i+m-1 T_j+m-1}). Maxima are <b>discords</b>
 * (subsequences unlike anything else: anomalies) and minima are <b>motifs</b> (repeated
 * patterns). The only parameter is {@code m}, and no training is involved.
 */
public final class MatrixProfile {

    private final double[] profile;
    private final int[] index;
    private final int window;

    private MatrixProfile(double[] profile, int[] index, int window) {
        this.profile = profile;
        this.index = index;
        this.window = window;
    }

    /**
     * Computes the self-join matrix profile.
     *
     * @param series the series
     * @param window the subsequence length {@code m}, at least 4
     * @return the profile
     */
    public static MatrixProfile stomp(double[] series, int window) {
        return join(series, series, window, true);
    }

    /**
     * Computes the AB-join: for each subsequence of {@code query}, the distance to its nearest
     * neighbour in {@code reference} (for example a clean training stretch), which turns
     * the profile into a novelty score.
     *
     * @param query the series to score
     * @param reference the reference series
     * @param window the subsequence length
     * @return the profile over {@code query}
     */
    public static MatrixProfile abJoin(double[] query, double[] reference, int window) {
        return join(query, reference, window, false);
    }

    private static MatrixProfile join(double[] a, double[] b, int m, boolean self) {
        int la = a.length - m + 1;
        int lb = b.length - m + 1;
        double[] muA = new double[la];
        double[] sdA = new double[la];
        double[] muB = new double[lb];
        double[] sdB = new double[lb];
        stats(a, m, muA, sdA);
        stats(b, m, muB, sdB);
        double[] profile = new double[la];
        int[] index = new int[la];
        Arrays.fill(profile, Double.POSITIVE_INFINITY);
        int exclusion = self ? Math.max(1, m / 4) : -1;
        double[] qt = new double[lb];
        double[] first = new double[la];   // dot products of b's first subsequence with every a subsequence
        for (int j = 0; j < la; j++) {
            double s = 0;
            for (int k = 0; k < m; k++) {
                s += a[j + k] * b[k];
            }
            first[j] = s;
        }
        for (int i = 0; i < la; i++) {
            if (i == 0) {
                for (int j = 0; j < lb; j++) {
                    double s = 0;
                    for (int k = 0; k < m; k++) {
                        s += a[k] * b[j + k];
                    }
                    qt[j] = s;
                }
            } else {
                for (int j = lb - 1; j >= 1; j--) {
                    qt[j] = qt[j - 1] - a[i - 1] * b[j - 1] + a[i + m - 1] * b[j + m - 1];
                }
                qt[0] = first[i];
            }
            for (int j = 0; j < lb; j++) {
                if (self && Math.abs(i - j) <= exclusion) {
                    continue;
                }
                double corr = (qt[j] - m * muA[i] * muB[j]) / (m * sdA[i] * sdB[j]);
                double d = Math.sqrt(Math.max(0, 2 * m * (1 - Math.min(1, corr))));
                if (d < profile[i]) {
                    profile[i] = d;
                    index[i] = j;
                }
            }
        }
        return new MatrixProfile(profile, index, m);
    }

    private static void stats(double[] t, int m, double[] mu, double[] sd) {
        double s = 0;
        double s2 = 0;
        for (int k = 0; k < m; k++) {
            s += t[k];
            s2 += t[k] * t[k];
        }
        for (int i = 0; i < mu.length; i++) {
            if (i > 0) {
                s += t[i + m - 1] - t[i - 1];
                s2 += t[i + m - 1] * t[i + m - 1] - t[i - 1] * t[i - 1];
            }
            mu[i] = s / m;
            sd[i] = Math.sqrt(Math.max(1e-8, s2 / m - mu[i] * mu[i]));
        }
    }

    /** @return the profile, one distance per subsequence start */
    public double[] profile() {
        return profile.clone();
    }

    /** @return the nearest neighbour's start for each subsequence */
    public int[] index() {
        return index.clone();
    }

    /** @return the subsequence length */
    public int window() {
        return window;
    }

    /**
     * The top discords: profile maxima, each at least {@code m} steps from the others.
     *
     * @param k how many
     * @return their start indices, most unusual first
     */
    public int[] discords(int k) {
        return extremes(k, true);
    }

    /**
     * The top motifs: profile minima (the start of each pair's first member), each at least
     * {@code m} steps from the others.
     *
     * @param k how many
     * @return their start indices, best match first
     */
    public int[] motifs(int k) {
        return extremes(k, false);
    }

    private int[] extremes(int k, boolean max) {
        Integer[] order = new Integer[profile.length];
        for (int i = 0; i < order.length; i++) {
            order[i] = i;
        }
        Arrays.sort(order, (x, y) -> max ? Double.compare(profile[y], profile[x])
                : Double.compare(profile[x], profile[y]));
        List<Integer> chosen = new ArrayList<>();
        for (int i : order) {
            if (chosen.size() == k) {
                break;
            }
            if (!Double.isFinite(profile[i])) {
                continue;
            }
            boolean far = true;
            for (int c : chosen) {
                far &= Math.abs(c - i) >= window;
            }
            if (far) {
                chosen.add(i);
            }
        }
        return chosen.stream().mapToInt(Integer::intValue).toArray();
    }

    /**
     * Per-step anomaly scores: each step takes the largest profile value of the subsequences
     * covering it.
     *
     * @param length the series length
     * @return per-step scores
     */
    public double[] perStep(int length) {
        return SeriesAnomalyMetrics.perStep(profile, window, length);
    }
}
