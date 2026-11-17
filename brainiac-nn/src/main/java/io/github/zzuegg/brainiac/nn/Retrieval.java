package io.github.zzuegg.brainiac.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * Two-tower retrieval: a query tower and an item tower embed into one space, trained with
 * an in-batch softmax, and items are found by the largest dot product. The search is exact,
 * a matrix product and a top-k on the device, in blocks of queries.
 */
public final class Retrieval {

    private Retrieval() {
    }

    /**
     * The in-batch softmax of two-tower models (Yi et al., 2019): query {@code i}'s positive
     * is item {@code i}, the other items of the batch are its negatives, and each item's
     * logit is lowered by the log of how often it is sampled so popular items are not
     * over-penalized.
     *
     * @param queries {@code [batch, dim]}
     * @param items {@code [batch, dim]}
     * @param logSampling each item's log sampling probability {@code [batch]}, or {@code null}
     *     for no correction
     * @param temperature divides the scores
     * @param normalize whether to compare by cosine rather than dot product
     * @return the mean cross-entropy, a scalar
     */
    public static Var inBatchSoftmax(Var queries, Var items, Tensor logSampling, float temperature,
            boolean normalize) {
        if (!queries.shape().equals(items.shape()) || queries.shape().rank() != 2) {
            throw new IllegalArgumentException("two [batch, dim] inputs needed, got " + queries.shape() + " and "
                    + items.shape());
        }
        Var q = normalize ? unit(queries) : queries;
        Var it = normalize ? unit(items) : items;
        int n = queries.shape().dim(0);
        Var logits = q.matmul(it.transpose()).div(temperature);
        if (logSampling != null) {
            logits = logits.sub(logSampling.reshape(Shape.of(1, n)).to(queries.value().device()));
        }
        return Functional.crossEntropy(logits, Tensor.arange(n).to(queries.value().device()));
    }

    private static Var unit(Var x) {
        return x.div(x.square().sum(1, true).add(1e-12f).sqrt());
    }

    /**
     * What a search found.
     *
     * @param ids per query, the item rows, best first
     * @param scores their scores
     */
    public record Hits(int[][] ids, float[][] scores) {
    }

    /** Item embeddings to search. */
    public static final class Index {
        private final Tensor items;
        private final boolean cosine;
        private final int block;

        /**
         * @param items {@code [n, dim]}
         * @param cosine whether to score by cosine rather than dot product
         */
        public Index(Tensor items, boolean cosine) {
            this(items, cosine, 256);
        }

        /**
         * @param items {@code [n, dim]}
         * @param cosine whether to score by cosine rather than dot product
         * @param block queries scored at once, bounding the {@code [block, n]} score matrix
         */
        public Index(Tensor items, boolean cosine, int block) {
            if (items.shape().rank() != 2) {
                throw new IllegalArgumentException("items need [n, dim], got " + items.shape());
            }
            this.items = cosine ? unitRows(items) : items;
            this.cosine = cosine;
            this.block = Math.max(1, block);
        }

        /** @return how many items */
        public int size() {
            return items.shape().dim(0);
        }

        /**
         * @param queries {@code [q, dim]}
         * @param k how many items per query
         * @return the best {@code k} items of each query
         */
        public Hits search(Tensor queries, int k) {
            Tensor q = cosine ? unitRows(queries) : queries;
            int rows = q.shape().dim(0);
            int kk = Math.min(k, size());
            int[][] ids = new int[rows][];
            float[][] scores = new float[rows][];
            Tensor t = items.transpose();
            for (int start = 0; start < rows; start += block) {
                int end = Math.min(rows, start + block);
                Tensor[] top = q.slice(0, start, end).to(items.device()).matmul(t).topk(kk);
                float[] v = top[0].toArray();
                float[] at = top[1].toArray();
                for (int r = start; r < end; r++) {
                    ids[r] = new int[kk];
                    scores[r] = new float[kk];
                    for (int j = 0; j < kk; j++) {
                        ids[r][j] = (int) at[(r - start) * kk + j];
                        scores[r][j] = v[(r - start) * kk + j];
                    }
                }
            }
            return new Hits(ids, scores);
        }

        private static Tensor unitRows(Tensor x) {
            return x.div(x.square().sum(1, true).add(1e-12f).sqrt());
        }
    }

    /**
     * @param hits a search's result
     * @param relevant each query's one relevant item
     * @return the share of queries whose relevant item was found
     */
    public static double recall(Hits hits, int[] relevant) {
        int found = 0;
        for (int q = 0; q < relevant.length; q++) {
            for (int id : hits.ids()[q]) {
                if (id == relevant[q]) {
                    found++;
                    break;
                }
            }
        }
        return relevant.length == 0 ? 0 : (double) found / relevant.length;
    }

    /**
     * @param hits a search's result
     * @param relevant each query's one relevant item
     * @return the mean of {@code 1 / rank} of the relevant item, 0 where it was not found
     */
    public static double meanReciprocalRank(Hits hits, int[] relevant) {
        double sum = 0;
        for (int q = 0; q < relevant.length; q++) {
            int[] ids = hits.ids()[q];
            for (int j = 0; j < ids.length; j++) {
                if (ids[j] == relevant[q]) {
                    sum += 1.0 / (j + 1);
                    break;
                }
            }
        }
        return relevant.length == 0 ? 0 : sum / relevant.length;
    }
}
