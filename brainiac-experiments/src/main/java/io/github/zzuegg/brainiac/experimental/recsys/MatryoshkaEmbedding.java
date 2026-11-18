package io.github.zzuegg.brainiac.experimental.recsys;

import io.github.zzuegg.brainiac.autograd.Var;

import java.util.Arrays;
import java.util.function.BinaryOperator;

/**
 * Matryoshka representation learning (Kusupati et al., 2022, <a href="https://arxiv.org/abs/2205.13147">
 * arXiv:2205.13147</a>): a loss wrapper that trains one embedding so that every nested prefix
 * (for example the first {@code d/8}, {@code d/4}, {@code d/2} and all {@code d} dimensions) is
 * itself a good embedding, by summing the base loss over the prefixes. At serving time each
 * system picks the size its budget allows, or uses <em>adaptive retrieval</em>
 * ({@link #adaptiveSearch}): a short prefix shortlists candidates cheaply and the full vectors
 * re-rank the shortlist. It wraps any pair loss of two embedding batches, such as
 * {@code Functional::infoNce} with a temperature.
 */
public final class MatryoshkaEmbedding {

    private final int[] sizes;
    private final double[] weights;
    private final BinaryOperator<Var> loss;

    private MatryoshkaEmbedding(int[] sizes, double[] weights, BinaryOperator<Var> loss) {
        this.sizes = sizes;
        this.weights = weights;
        this.loss = loss;
    }

    /**
     * Equal weights for all prefixes.
     *
     * @param sizes the nested prefix sizes, increasing; the last is usually the full width
     * @param loss the base loss of two {@code [batch, size]} embedding batches
     * @return the wrapper
     */
    public static MatryoshkaEmbedding of(int[] sizes, BinaryOperator<Var> loss) {
        double[] w = new double[sizes.length];
        Arrays.fill(w, 1.0 / sizes.length);
        return new MatryoshkaEmbedding(sizes.clone(), w, loss);
    }

    /** @return the nested prefix sizes */
    public int[] sizes() {
        return sizes.clone();
    }

    /**
     * The weighted sum of the base loss over the nested prefixes.
     *
     * @param a {@code [batch, d]}
     * @param b {@code [batch, d]}
     * @return a scalar
     */
    public Var compute(Var a, Var b) {
        Var total = null;
        for (int i = 0; i < sizes.length; i++) {
            Var l = loss.apply(a.slice(1, 0, sizes[i]), b.slice(1, 0, sizes[i])).mul((float) weights[i]);
            total = total == null ? l : total.add(l);
        }
        return total;
    }

    /**
     * Cosine similarity of the first {@code size} dimensions.
     *
     * @param q a query vector
     * @param d a document vector
     * @param size the prefix length
     * @return the cosine of the prefixes
     */
    public static double prefixCosine(float[] q, float[] d, int size) {
        double dot = 0;
        double nq = 0;
        double nd = 0;
        for (int i = 0; i < size; i++) {
            dot += q[i] * d[i];
            nq += q[i] * q[i];
            nd += d[i] * d[i];
        }
        return dot / Math.sqrt(nq * nd + 1e-12);
    }

    /**
     * Adaptive retrieval: shortlist by the cosine of a short prefix, then re-rank the shortlist
     * by the full vectors.
     *
     * @param query the query vector
     * @param documents the document vectors
     * @param shortSize the shortlisting prefix length
     * @param shortlist how many candidates to keep
     * @param k how many results
     * @return the best {@code k} documents, best first
     */
    public static int[] adaptiveSearch(float[] query, float[][] documents, int shortSize, int shortlist, int k) {
        float[] coarse = new float[documents.length];
        for (int i = 0; i < documents.length; i++) {
            coarse[i] = (float) prefixCosine(query, documents[i], shortSize);
        }
        int[] candidates = RankingMetrics.topK(coarse, shortlist, null);
        float[] fine = new float[documents.length];
        Arrays.fill(fine, Float.NEGATIVE_INFINITY);
        for (int c : candidates) {
            fine[c] = (float) prefixCosine(query, documents[c], query.length);
        }
        return RankingMetrics.topK(fine, k, null);
    }
}
