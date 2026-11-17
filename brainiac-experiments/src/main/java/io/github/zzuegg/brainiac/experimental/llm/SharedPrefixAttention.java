package io.github.zzuegg.brainiac.experimental.llm;

import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;

/**
 * Hydragen: exact attention for many sequences that share a long prefix (many samples of one
 * prompt, a shared system prompt or few-shot block). Softmax attention over the concatenation
 * of a prefix and a suffix splits into the two parts, combined by their log-sum-exps:
 *
 * <pre>
 * out = (e^lse_p o_p + e^lse_s o_s) / (e^lse_p + e^lse_s)
 * </pre>
 *
 * <p>The prefix part of all sequences is then <em>one</em> attention call with the
 * sequences' queries stacked as rows ({@code [1, heads, batch, dim]} against the prefix keys
 * read once), a matrix-matrix product instead of {@code batch} matrix-vector products that each
 * re-read the same prefix keys; only the short suffixes are attended per sequence. With paged
 * blocks shared copy-on-write ({@link PrefixCache}) the memory is shared already, but plain
 * per-sequence attention still reads the prefix {@code batch} times; this decomposition reads
 * it once. Results equal per-sequence attention up to rounding.
 *
 * <p>This is the attention kernel only, on any backend with fused attention
 * ({@link Tensor#attention}, which returns the log-sum-exp needed for the merge); wiring it into
 * {@link PagedLlama}'s decode path would need the store to expose shared-prefix block ranges.
 *
 * <p>Juravsky et al., "Hydragen: High-Throughput LLM Inference with Shared Prefixes" (2024),
 * <a href="https://arxiv.org/abs/2402.05099">arXiv:2402.05099</a>.
 */
public final class SharedPrefixAttention {

    private SharedPrefixAttention() {
    }

    /**
     * Decode attention of many sequences over a shared prefix plus each one's own suffix.
     *
     * @param queries one query per sequence {@code [batch, heads, dim]}
     * @param prefixKeys the shared prefix's keys {@code [prefix, kvHeads, dim]}
     * @param prefixValues its values {@code [prefix, kvHeads, dim]}
     * @param suffixKeys per sequence its own keys {@code [suffix, kvHeads, dim]} (at least one row)
     * @param suffixValues per sequence its own values
     * @param scale the score scale
     * @return {@code [batch, heads, dim]}
     */
    public static Tensor attend(Tensor queries, Tensor prefixKeys, Tensor prefixValues, List<Tensor> suffixKeys,
            List<Tensor> suffixValues, float scale) {
        int batch = queries.shape().dim(0);
        int heads = queries.shape().dim(1);
        int dim = queries.shape().dim(2);
        Tensor q = queries.permute(1, 0, 2).reshape(1, heads, batch, dim);
        Tensor[] prefix = Tensor.attention(q, heads4(prefixKeys), heads4(prefixValues), scale, false);
        Tensor[] outs = new Tensor[batch];
        Tensor[] lses = new Tensor[batch];
        for (int b = 0; b < batch; b++) {
            Tensor qb = queries.slice(0, b, b + 1).reshape(1, heads, 1, dim);
            Tensor[] s = Tensor.attention(qb, heads4(suffixKeys.get(b)), heads4(suffixValues.get(b)), scale, false);
            outs[b] = s[0];
            lses[b] = s[1];
        }
        Tensor suffixOut = batch == 1 ? outs[0] : Tensor.concat(2, outs);   // [1, heads, batch, dv]
        Tensor suffixLse = batch == 1 ? lses[0] : Tensor.concat(2, lses);   // [1, heads, batch]
        Tensor merged = merge(prefix[0], prefix[1], suffixOut, suffixLse);
        int dv = merged.shape().dim(3);
        return merged.reshape(heads, batch, dv).permute(1, 0, 2);
    }

    /**
     * Combines two partial attentions over disjoint key sets into attention over their union.
     *
     * @param o1 the first part's output {@code [..., rows, dv]}
     * @param lse1 its log-sum-exp {@code [..., rows]}
     * @param o2 the second part's output
     * @param lse2 its log-sum-exp
     * @return the combined output
     */
    public static Tensor merge(Tensor o1, Tensor lse1, Tensor o2, Tensor lse2) {
        int[] dims = lse1.shape().toArray();
        int[] col = java.util.Arrays.copyOf(dims, dims.length + 1);
        col[dims.length] = 1;
        Tensor m = lse1.maximum(lse2);
        Tensor w1 = lse1.sub(m).exp().reshape(col);
        Tensor w2 = lse2.sub(m).exp().reshape(col);
        return o1.mul(w1).add(o2.mul(w2)).div(w1.add(w2));
    }

    /**
     * The baseline: each sequence attends to its prefix and suffix as one key list, as
     * per-sequence attention over shared paged blocks does (the prefix is read once per
     * sequence).
     *
     * @param queries one query per sequence {@code [batch, heads, dim]}
     * @param prefixKeys the shared prefix's keys {@code [prefix, kvHeads, dim]}
     * @param prefixValues its values
     * @param suffixKeys per sequence its own keys
     * @param suffixValues per sequence its own values
     * @param scale the score scale
     * @return {@code [batch, heads, dim]}
     */
    public static Tensor perSequence(Tensor queries, Tensor prefixKeys, Tensor prefixValues, List<Tensor> suffixKeys,
            List<Tensor> suffixValues, float scale) {
        int batch = queries.shape().dim(0);
        int heads = queries.shape().dim(1);
        int dim = queries.shape().dim(2);
        Tensor[] outs = new Tensor[batch];
        for (int b = 0; b < batch; b++) {
            Tensor k = Tensor.concat(0, prefixKeys, suffixKeys.get(b));
            Tensor v = Tensor.concat(0, prefixValues, suffixValues.get(b));
            Tensor qb = queries.slice(0, b, b + 1).reshape(1, heads, 1, dim);
            Tensor o = Tensor.attention(qb, heads4(k), heads4(v), scale, false)[0];
            outs[b] = o.reshape(1, heads, o.shape().dim(3));
        }
        return batch == 1 ? outs[0] : Tensor.concat(0, outs);
    }

    /** {@code [n, kvHeads, dim]} to {@code [1, kvHeads, n, dim]}. */
    private static Tensor heads4(Tensor rows) {
        int n = rows.shape().dim(0);
        int kv = rows.shape().dim(1);
        int d = rows.shape().dim(2);
        return rows.permute(1, 0, 2).reshape(1, kv, n, d);
    }
}
