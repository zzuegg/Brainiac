package io.github.zzuegg.brainiac.experimental.serving;

import java.util.Arrays;

/**
 * Draft-based approximate inference (Galim et al., "Draft-based Approximate Inference for
 * LLMs", ICLR 2026, <a href="https://arxiv.org/abs/2506.08373">arXiv:2506.08373</a>). KV
 * eviction rules such as SnapKV score prompt tokens by the attention the prompt's own last
 * tokens pay them, which cannot see what the answer will need. A small draft model first writes
 * a short draft answer; its tokens then act as extra queries that "look ahead".
 * <ul>
 * <li>{@link #specKv}: the draft tokens are run through the <em>target</em> model, and the
 *     target's queries of the last {@code window} prompt tokens and of the draft tokens attend
 *     to the earlier prompt keys; each key's score is the maximum attention any of those
 *     queries gives it, average-pooled over {@code kernel} neighbours. The top
 *     {@code keep - window} keys and the whole window stay in the cache (Algorithm 1 of the
 *     paper; per head, call it once per head).</li>
 * <li>{@link #specPc}: prompt compression before the target runs at all. The <em>draft</em>
 *     model's own attention rows (the window's and the draft tokens' queries over the prompt)
 *     are reweighted so later window rows count more, max-reduced, average-pooled over
 *     {@code kernel} and max-pooled over {@code neighbours} (so a key and its value stay
 *     together); the top tokens plus the window form the compressed prompt (Algorithm 2).</li>
 * </ul>
 * Both work on plain attention scores, so they plug into any KV store or prompt path: the
 * caller passes queries and keys (SpecKV) or the draft's attention matrix (SpecPC).
 */
public final class DraftGuidedKvSelector {

    private final boolean promptCompression;
    private final int window;
    private final int kernel;
    private final int neighbours;

    private DraftGuidedKvSelector(boolean promptCompression, int window, int kernel, int neighbours) {
        if (window < 0 || kernel < 1 || kernel % 2 == 0 || neighbours < 1 || neighbours % 2 == 0) {
            throw new IllegalArgumentException("window must be at least 0, kernel and neighbours positive and odd");
        }
        this.promptCompression = promptCompression;
        this.window = window;
        this.kernel = kernel;
        this.neighbours = neighbours;
    }

    /**
     * SpecKV: KV selection by the target's attention from the window and the draft tokens.
     *
     * @param window the last prompt tokens that are always kept and also vote
     * @param kernel the average-pooling width, odd (1 for none)
     * @return the selector
     */
    public static DraftGuidedKvSelector specKv(int window, int kernel) {
        return new DraftGuidedKvSelector(false, window, kernel, 1);
    }

    /**
     * SpecPC: prompt compression by the draft model's attention.
     *
     * @param window the last prompt tokens that are always kept and also vote
     * @param kernel the average-pooling width, odd
     * @param neighbours the max-pooling width that keeps spans together, odd
     * @return the selector
     */
    public static DraftGuidedKvSelector specPc(int window, int kernel, int neighbours) {
        return new DraftGuidedKvSelector(true, window, kernel, neighbours);
    }

    /** @return whether this is SpecPC (prompt compression) rather than SpecKV */
    public boolean promptCompression() {
        return promptCompression;
    }

    /** @return the always-kept window */
    public int window() {
        return window;
    }

    /**
     * SpecKV selection from the target's queries and keys of one head.
     *
     * @param queries the target queries of the last {@code window} prompt tokens followed by
     *     those of the draft tokens, {@code [q][d]}
     * @param keys the target keys of every prompt token, oldest first, {@code [n][d]}
     * @param scale the softmax scale, such as {@code 1 / sqrt(d)}
     * @param keep the tokens kept, window included
     * @return per prompt token whether it stays
     */
    public boolean[] select(float[][] queries, float[][] keys, float scale, int keep) {
        int n = keys.length;
        int prefix = Math.max(0, n - window);
        float[][] attention = new float[queries.length][prefix];
        for (int i = 0; i < queries.length; i++) {
            double max = Double.NEGATIVE_INFINITY;
            double[] s = new double[prefix];
            for (int j = 0; j < prefix; j++) {
                double dot = 0;
                for (int c = 0; c < queries[i].length; c++) {
                    dot += queries[i][c] * keys[j][c];
                }
                s[j] = dot * scale;
                max = Math.max(max, s[j]);
            }
            double z = 0;
            for (int j = 0; j < prefix; j++) {
                s[j] = Math.exp(s[j] - max);
                z += s[j];
            }
            for (int j = 0; j < prefix; j++) {
                attention[i][j] = (float) (s[j] / z);
            }
        }
        return selectFromAttention(attention, n, keep);
    }

    /**
     * Selection from attention rows already computed: the target's (SpecKV) or the draft's
     * (SpecPC), each row a query's softmax attention over the {@code n - window} prompt tokens
     * before the window (longer rows are cut). Rows are ordered window rows first, then draft
     * rows; with SpecPC, window row {@code j} of {@code w} gets weight {@code (j + 1) / w}.
     *
     * @param attention the attention rows, {@code [q][>= n - window]}
     * @param n the prompt length
     * @param keep the tokens kept, window included
     * @return per prompt token whether it stays
     */
    public boolean[] selectFromAttention(float[][] attention, int n, int keep) {
        boolean[] kept = new boolean[n];
        int w = Math.min(window, n);
        int prefix = n - w;
        for (int i = prefix; i < n; i++) {
            kept[i] = true;
        }
        int room = keep - w;
        if (room >= prefix) {
            Arrays.fill(kept, true);
            return kept;
        }
        if (room <= 0) {
            return kept;
        }
        double[] score = scores(attention, prefix);
        Integer[] order = new Integer[prefix];
        for (int j = 0; j < prefix; j++) {
            order[j] = j;
        }
        Arrays.sort(order, (a, b) -> Double.compare(score[b], score[a]));
        for (int i = 0; i < room; i++) {
            kept[order[i]] = true;
        }
        return kept;
    }

    /**
     * The pooled importance of every prompt token before the window.
     *
     * @param attention the attention rows
     * @param prefix the tokens scored
     * @return the scores
     */
    public double[] scores(float[][] attention, int prefix) {
        double[] s = new double[prefix];
        int windowRows = Math.min(window, attention.length);
        for (int i = 0; i < attention.length; i++) {
            double weight = promptCompression && i < windowRows ? (i + 1) / (double) windowRows : 1;
            for (int j = 0; j < prefix && j < attention[i].length; j++) {
                s[j] = Math.max(s[j], weight * attention[i][j]);
            }
        }
        double[] pooled = pool(s, kernel, false);
        if (promptCompression && neighbours > 1) {
            pooled = pool(pooled, neighbours, true);
        }
        for (int j = 0; j < prefix; j++) {
            pooled[j] += 1e-9 * s[j];   // ties go to the token that was attended itself
        }
        return pooled;
    }

    private static double[] pool(double[] s, int width, boolean max) {
        if (width == 1) {
            return s.clone();
        }
        int r = width / 2;
        double[] out = new double[s.length];
        for (int j = 0; j < s.length; j++) {
            double acc = max ? Double.NEGATIVE_INFINITY : 0;
            int count = 0;
            for (int d = -r; d <= r; d++) {
                int at = j + d;
                if (at >= 0 && at < s.length) {
                    acc = max ? Math.max(acc, s[at]) : acc + s[at];
                    count++;
                }
            }
            out[j] = max ? acc : acc / count;
        }
        return out;
    }

    @Override
    public String toString() {
        return (promptCompression ? "SpecPC" : "SpecKV") + "(window " + window + ", kernel " + kernel
                + (promptCompression ? ", neighbours " + neighbours : "") + ")";
    }
}
