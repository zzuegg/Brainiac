package io.github.zzuegg.brainiac.experimental.llm;

import java.util.Arrays;

/**
 * KVzip: query-agnostic KV eviction. After the context is cached, the model is prompted to
 * repeat the context ("Repeat the previous context:" followed by the context itself) and the
 * attention its reconstruction queries pay to each cached entry is recorded. An entry that no
 * reconstruction step attends to carries no information the model would need to rebuild the
 * context, so it can go; the entries kept serve <em>any</em> later question. Query-aware
 * methods such as SnapKV score with the question at the end of the prompt and lose what a
 * different, later question needs.
 *
 * <p>An entry's score is its <em>largest</em> attention over all reconstruction queries of
 * its head (one strong read is enough to need it); the budget goes to the best scores over
 * every layer and head together, so heads get non-uniform budgets. The snapshot's queries are
 * the reconstruction pass's queries; any {@link KvBudgetPolicy} harness can supply them.
 *
 * <p>Kim et al., "KVzip: Query-Agnostic KV Cache Compression with Context Reconstruction"
 * (2025), <a href="https://arxiv.org/abs/2505.23416">arXiv:2505.23416</a>.
 */
public final class ReconstructionKvScorer implements KvBudgetPolicy {

    private final int recent;

    /**
     * @param recent the most recent tokens every head keeps regardless of score
     */
    public ReconstructionKvScorer(int recent) {
        if (recent < 0) {
            throw new IllegalArgumentException("recent must be at least 0");
        }
        this.recent = recent;
    }

    /**
     * The score of every entry: its largest attention over the head's reconstruction queries.
     *
     * @param snapshot keys and reconstruction queries
     * @return {@code [layer][head][token]}
     */
    public static float[][][] scores(KvSnapshot snapshot) {
        float[][][] out = new float[snapshot.layers()][snapshot.heads()][snapshot.length()];
        for (int l = 0; l < snapshot.layers(); l++) {
            for (int h = 0; h < snapshot.heads(); h++) {
                for (float[] row : snapshot.attention(l, h)) {
                    for (int t = 0; t < row.length; t++) {
                        out[l][h][t] = Math.max(out[l][h][t], row[t]);
                    }
                }
            }
        }
        return out;
    }

    @Override
    public boolean[][][] select(KvSnapshot snapshot, int budget) {
        int layers = snapshot.layers();
        int heads = snapshot.heads();
        int n = snapshot.length();
        float[][][] score = scores(snapshot);
        boolean[][][] keep = new boolean[layers][heads][n];
        int used = 0;
        int r = Math.min(recent, n);
        for (int l = 0; l < layers; l++) {
            for (int h = 0; h < heads; h++) {
                for (int t = n - r; t < n; t++) {
                    keep[l][h][t] = true;
                    used++;
                }
            }
        }
        int candidates = layers * heads * (n - r);
        Integer[] order = new Integer[candidates];
        float[] flat = new float[candidates];
        int i = 0;
        for (int l = 0; l < layers; l++) {
            for (int h = 0; h < heads; h++) {
                for (int t = 0; t < n - r; t++) {
                    flat[i] = score[l][h][t];
                    order[i] = i;
                    i++;
                }
            }
        }
        Arrays.sort(order, (a, b) -> Float.compare(flat[b], flat[a]));
        int per = n - r;
        for (int k = 0; k < candidates && used < budget; k++) {
            int at = order[k];
            int lh = at / Math.max(1, per);
            keep[lh / heads][lh % heads][at % per] = true;
            used++;
        }
        return keep;
    }

    @Override
    public String toString() {
        return "ReconstructionKvScorer(recent " + recent + ")";
    }
}
