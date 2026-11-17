package io.github.zzuegg.brainiac.experimental.llm;

/**
 * A cached context as KV compression policies see it: the keys of every token in every
 * layer and head, plus the queries that score them (the observation window of SnapKV, the
 * queries of a context-reconstruction pass for KVzip, or calibration queries). Policies such
 * as {@link KvBudgetPolicy} read it and decide which entries each head keeps.
 *
 * <p>Head-level granularity matters: heads differ in how many tokens they need (retrieval
 * heads read a few far-back tokens, streaming heads only recent ones, some heads spread
 * attention widely), which a cache that keeps one set of positions for all heads cannot use.
 *
 * @param keys {@code [layer][head][token][dim]}
 * @param queries {@code [layer][head][query][dim]} the scoring queries; may be empty per head
 * @param scale the attention score scale, usually {@code 1 / sqrt(dim)}
 */
public record KvSnapshot(float[][][][] keys, float[][][][] queries, float scale) {

    /** Checks the shapes. */
    public KvSnapshot {
        if (keys.length == 0 || keys.length != queries.length) {
            throw new IllegalArgumentException("need keys and queries for the same layers");
        }
    }

    /** @return the layers */
    public int layers() {
        return keys.length;
    }

    /** @return the heads per layer */
    public int heads() {
        return keys[0].length;
    }

    /** @return the cached tokens */
    public int length() {
        return keys[0][0].length;
    }

    /**
     * The softmax attention of each scoring query of one head over every cached token.
     *
     * @param layer the layer
     * @param head the head
     * @return {@code [query][token]}
     */
    public float[][] attention(int layer, int head) {
        float[][] q = queries[layer][head];
        float[][] k = keys[layer][head];
        float[][] out = new float[q.length][];
        for (int i = 0; i < q.length; i++) {
            out[i] = softmax(q[i], k, null, scale);
        }
        return out;
    }

    /**
     * Softmax attention weights of one query over keys, optionally restricted to kept keys.
     *
     * @param q the query
     * @param keys the keys {@code [token][dim]}
     * @param keep which keys count, or {@code null} for all
     * @param scale the score scale
     * @return the weights; zero for keys not kept
     */
    public static float[] softmax(float[] q, float[][] keys, boolean[] keep, float scale) {
        int n = keys.length;
        double[] s = new double[n];
        double max = Double.NEGATIVE_INFINITY;
        for (int j = 0; j < n; j++) {
            if (keep != null && !keep[j]) {
                continue;
            }
            double d = 0;
            for (int c = 0; c < q.length; c++) {
                d += q[c] * keys[j][c];
            }
            s[j] = d * scale;
            max = Math.max(max, s[j]);
        }
        double sum = 0;
        for (int j = 0; j < n; j++) {
            if (keep == null || keep[j]) {
                s[j] = Math.exp(s[j] - max);
                sum += s[j];
            }
        }
        float[] w = new float[n];
        for (int j = 0; j < n; j++) {
            if (keep == null || keep[j]) {
                w[j] = (float) (s[j] / sum);
            }
        }
        return w;
    }
}
