package io.github.zzuegg.brainiac.experimental.nn;

/**
 * SnapKV prompt compression (Li et al. 2024, <a href="https://arxiv.org/abs/2404.14469">arXiv:2404.14469</a>):
 * the last {@code window} tokens of a prompt (the "observation window", usually the
 * question) already attend to what generation will need, so their attention votes, once, for
 * which earlier tokens stay. Votes are summed over the window's queries and pooled over
 * neighbouring positions (so a selected token keeps its context), and the prefix tokens with
 * the most votes are kept together with the whole window.
 *
 * <p>The paper votes per head; {@link EvictingKvCache} keeps one set of positions for all
 * heads, so votes arrive averaged over heads.
 */
public final class SnapKv {

    private final int window;
    private final int kernel;

    private SnapKv(int window, int kernel) {
        if (window < 1 || kernel < 1 || kernel % 2 == 0) {
            throw new IllegalArgumentException("window must be positive and kernel a positive odd number");
        }
        this.window = window;
        this.kernel = kernel;
    }

    /**
     * Creates the rule.
     *
     * @param window the observation window, such as 16 or 32
     * @param kernel the pooling width over positions, odd, such as 5 or 7 (1 for no pooling)
     * @return the rule
     */
    public static SnapKv of(int window, int kernel) {
        return new SnapKv(window, kernel);
    }

    /** @return the observation window */
    public int window() {
        return window;
    }

    /**
     * Chooses the tokens to keep.
     *
     * @param votes per window query, its attention over all {@code n} cached tokens, oldest
     *     first; the window is the last {@code votes.length} tokens
     * @param keep the tokens kept, window included
     * @return per token whether it stays
     */
    public boolean[] select(float[][] votes, int keep) {
        int n = votes.length == 0 ? 0 : votes[0].length;
        boolean[] kept = new boolean[n];
        int w = Math.min(votes.length, n);
        int prefix = n - w;
        for (int i = prefix; i < n; i++) {
            kept[i] = true;
        }
        int room = keep - w;
        if (room >= prefix) {
            java.util.Arrays.fill(kept, true);
            return kept;
        }
        if (room <= 0) {
            return kept;
        }
        double[] score = new double[prefix];
        for (float[] v : votes) {
            for (int j = 0; j < prefix; j++) {
                score[j] += v[j];
            }
        }
        double[] pooled = new double[prefix];
        int r = kernel / 2;
        for (int j = 0; j < prefix; j++) {
            double m = 0;
            for (int d = -r; d <= r; d++) {
                int at = j + d;
                if (at >= 0 && at < prefix) {
                    m = Math.max(m, score[at]);
                }
            }
            pooled[j] = m + 1e-9 * score[j];   // ties go to the token voted for itself
        }
        Integer[] order = new Integer[prefix];
        for (int j = 0; j < prefix; j++) {
            order[j] = j;
        }
        java.util.Arrays.sort(order, (a, b) -> Double.compare(pooled[b], pooled[a]));
        for (int i = 0; i < room; i++) {
            kept[order[i]] = true;
        }
        return kept;
    }

    @Override
    public String toString() {
        return "SnapKv(window " + window + ", kernel " + kernel + ")";
    }
}
