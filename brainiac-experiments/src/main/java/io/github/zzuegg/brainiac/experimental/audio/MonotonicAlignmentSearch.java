package io.github.zzuegg.brainiac.experimental.audio;

/**
 * Monotonic alignment search (Kim et al., Glow-TTS, NeurIPS 2020,
 * <a href="https://arxiv.org/abs/2005.11129">arXiv:2005.11129</a>), as Matcha-TTS uses it: the
 * most likely monotonic, surjective alignment of {@code F} frames to {@code S} text symbols, by a
 * Viterbi-like dynamic programme {@code Q[i][j] = max(Q[i][j-1], Q[i-1][j-1]) + logLik[i][j]}
 * that starts at the first symbol on the first frame, ends at the last symbol on the last frame,
 * and lets each frame either stay on the current symbol or advance by one. Every symbol gets at
 * least one frame.
 */
public final class MonotonicAlignmentSearch {

    private MonotonicAlignmentSearch() {
    }

    /**
     * The best alignment.
     *
     * @param logLik {@code [symbols][frames]} log-likelihood of each frame under each symbol;
     *     needs {@code symbols <= frames}
     * @return the symbol of every frame, non-decreasing from 0 to {@code symbols - 1} in steps of
     *     at most one
     */
    public static int[] align(double[][] logLik) {
        int s = logLik.length;
        int f = logLik[0].length;
        if (s > f) {
            throw new IllegalArgumentException(s + " symbols cannot align to " + f + " frames");
        }
        double[][] q = new double[s][f];
        for (double[] row : q) {
            java.util.Arrays.fill(row, Double.NEGATIVE_INFINITY);
        }
        q[0][0] = logLik[0][0];
        for (int j = 1; j < f; j++) {
            for (int i = Math.max(0, s - (f - j)); i <= Math.min(s - 1, j); i++) {
                double stay = q[i][j - 1];
                double move = i > 0 ? q[i - 1][j - 1] : Double.NEGATIVE_INFINITY;
                q[i][j] = Math.max(stay, move) + logLik[i][j];
            }
        }
        int[] path = new int[f];
        int i = s - 1;
        for (int j = f - 1; j >= 0; j--) {
            path[j] = i;
            if (j > 0 && i > 0 && (i == j || q[i - 1][j - 1] > q[i][j - 1])) {
                i--;
            }
        }
        return path;
    }

    /**
     * Frames per symbol of an alignment.
     *
     * @param path the symbol of every frame
     * @param symbols the number of symbols
     * @return the durations
     */
    public static int[] durations(int[] path, int symbols) {
        int[] d = new int[symbols];
        for (int p : path) {
            d[p]++;
        }
        return d;
    }

    /**
     * The alignment that durations describe.
     *
     * @param durations frames per symbol
     * @param frames the number of frames to fill (the last symbol is extended or cut)
     * @return the symbol of every frame
     */
    public static int[] path(int[] durations, int frames) {
        int[] p = new int[frames];
        int j = 0;
        for (int i = 0; i < durations.length && j < frames; i++) {
            for (int k = 0; k < durations[i] && j < frames; k++) {
                p[j++] = i;
            }
        }
        while (j < frames) {
            p[j] = durations.length - 1;
            j++;
        }
        return p;
    }

    /**
     * The mean absolute error of symbol boundaries, in frames, between two duration lists of the
     * same symbols.
     *
     * @param a durations
     * @param b durations
     * @return the error
     */
    public static double boundaryError(int[] a, int[] b) {
        double e = 0;
        int ca = 0;
        int cb = 0;
        int n = Math.min(a.length, b.length) - 1;
        for (int i = 0; i < n; i++) {
            ca += a[i];
            cb += b[i];
            e += Math.abs(ca - cb);
        }
        return n <= 0 ? 0 : e / n;
    }
}
