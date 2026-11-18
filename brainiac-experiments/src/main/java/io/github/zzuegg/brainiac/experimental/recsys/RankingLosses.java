package io.github.zzuegg.brainiac.experimental.recsys;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Loss;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * Learning-to-rank losses as core {@link Loss}es. Predictions are scores {@code [queries, n]} of
 * {@code n} candidate documents per query; targets are their graded relevance labels, same shape.
 * <ul>
 *   <li>{@link #rankNet} (Burges et al., ICML 2005): a pairwise logistic loss on score
 *       differences, {@code log(1 + exp(-(s_i - s_j)))} for every pair with {@code y_i > y_j};</li>
 *   <li>{@link #lambdaRank} (Burges, 2010, "From RankNet to LambdaRank to LambdaMART",
 *       MSR-TR-2010-82): the same pairs, each weighted by {@code |delta NDCG@k|}, the change of
 *       NDCG@k if the two were swapped in the current ranking, so the gradient concentrates on the
 *       pairs that move the metric (see {@code LambdaMart} for the boosted-tree version);</li>
 *   <li>{@link #listNet} (Cao et al., ICML 2007): the cross-entropy between the top-one
 *       probabilities {@code softmax(y)} of the labels and {@code softmax(s)} of the scores.</li>
 * </ul>
 */
public final class RankingLosses {

    private RankingLosses() {
    }

    /** @return the RankNet loss */
    public static Loss rankNet() {
        return (scores, labels) -> pairwise(scores, pairs(labels, null, 0));
    }

    /**
     * @param k the NDCG cut-off whose changes weight the pairs
     * @return the LambdaRank loss
     */
    public static Loss lambdaRank(int k) {
        return (scores, labels) -> pairwise(scores, pairs(labels, scores.value(), k));
    }

    /** @return the ListNet loss */
    public static Loss listNet() {
        return (scores, labels) -> {
            Tensor p = Activation.SOFTMAX.forward(Var.of(labels)).value();
            return Activation.LOG_SOFTMAX.forward(scores).mul(p).sum(1, false).mean().neg();
        };
    }

    private static Var pairwise(Var scores, Tensor weights) {
        int q = scores.shape().dim(0);
        int n = scores.shape().dim(1);
        Var diff = scores.reshape(q, n, 1).sub(scores.reshape(q, 1, n));
        float total = Math.max(1e-6f, weights.sum().item());
        return diff.neg().softplus().mul(weights).sum().div(total);
    }

    /**
     * Pair weights {@code [q, n, n]}: 1 (or {@code |delta NDCG@k|} when {@code scores} is given)
     * where {@code y_i > y_j}, else 0.
     */
    private static Tensor pairs(Tensor labels, Tensor scores, int k) {
        int q = labels.shape().dim(0);
        int n = labels.shape().dim(1);
        float[] y = labels.toArray();
        float[] s = scores == null ? null : scores.toArray();
        float[] w = new float[q * n * n];
        for (int a = 0; a < q; a++) {
            double[] discount = new double[n];
            double idcg = 1;
            if (s != null) {
                float[] row = java.util.Arrays.copyOfRange(s, a * n, (a + 1) * n);
                int[] order = RankingMetrics.topK(row, n, null);
                for (int r = 0; r < order.length; r++) {
                    discount[order[r]] = r < k ? 1 / Math.log(r + 2) * Math.log(2) : 0;
                }
                float[] sorted = java.util.Arrays.copyOfRange(y, a * n, (a + 1) * n);
                java.util.Arrays.sort(sorted);
                idcg = 0;
                for (int r = 0; r < Math.min(k, n); r++) {
                    idcg += (Math.pow(2, sorted[n - 1 - r]) - 1) / (Math.log(r + 2) / Math.log(2));
                }
                idcg = Math.max(idcg, 1e-9);
            }
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    if (y[a * n + i] > y[a * n + j]) {
                        w[(a * n + i) * n + j] = s == null ? 1 : (float) Math.abs(
                                (Math.pow(2, y[a * n + i]) - Math.pow(2, y[a * n + j]))
                                        * (discount[i] - discount[j]) / idcg);
                    }
                }
            }
        }
        return Tensor.of(Shape.of(q, n, n), w);
    }
}
