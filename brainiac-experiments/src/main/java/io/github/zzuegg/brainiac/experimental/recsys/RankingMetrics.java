package io.github.zzuegg.brainiac.experimental.recsys;

import java.util.Arrays;
import java.util.Set;

/**
 * Top-k ranking metrics for recommendation and retrieval: the top-k list of a score vector,
 * recall@k, hit rate, NDCG@k with binary or graded relevance, and the reciprocal rank.
 */
public final class RankingMetrics {

    private RankingMetrics() {
    }

    /**
     * The indices of the {@code k} highest scores, best first.
     *
     * @param scores one score per candidate; {@code NaN} and negative infinity are never picked
     * @param k how many
     * @param excluded candidates to skip (such as already-seen items), or {@code null}
     * @return up to {@code k} indices
     */
    public static int[] topK(float[] scores, int k, Set<Integer> excluded) {
        int[] best = new int[k];
        float[] val = new float[k];
        Arrays.fill(val, Float.NEGATIVE_INFINITY);
        int n = 0;
        for (int i = 0; i < scores.length; i++) {
            float s = scores[i];
            if (Float.isNaN(s) || s == Float.NEGATIVE_INFINITY || (excluded != null && excluded.contains(i))) {
                continue;
            }
            if (n < k || s > val[n - 1]) {
                int pos = n < k ? n++ : n - 1;
                while (pos > 0 && val[pos - 1] < s) {
                    val[pos] = val[pos - 1];
                    best[pos] = best[pos - 1];
                    pos--;
                }
                val[pos] = s;
                best[pos] = i;
            }
        }
        return Arrays.copyOf(best, n);
    }

    /**
     * @param ranked a ranked list
     * @param relevant the relevant candidates
     * @param k the cut-off
     * @return the share of relevant candidates in the first {@code k}
     */
    public static double recall(int[] ranked, Set<Integer> relevant, int k) {
        if (relevant.isEmpty()) {
            return 0;
        }
        int hits = 0;
        for (int i = 0; i < Math.min(k, ranked.length); i++) {
            hits += relevant.contains(ranked[i]) ? 1 : 0;
        }
        return hits / (double) relevant.size();
    }

    /**
     * @param ranked a ranked list
     * @param target the one relevant candidate
     * @param k the cut-off
     * @return 1 if the target is in the first {@code k}, else 0
     */
    public static double hit(int[] ranked, int target, int k) {
        for (int i = 0; i < Math.min(k, ranked.length); i++) {
            if (ranked[i] == target) {
                return 1;
            }
        }
        return 0;
    }

    /**
     * @param ranked a ranked list
     * @param target the one relevant candidate
     * @param k the cut-off
     * @return {@code 1 / rank} of the target within the first {@code k}, else 0
     */
    public static double reciprocalRank(int[] ranked, int target, int k) {
        for (int i = 0; i < Math.min(k, ranked.length); i++) {
            if (ranked[i] == target) {
                return 1.0 / (i + 1);
            }
        }
        return 0;
    }

    /**
     * NDCG@k with binary relevance.
     *
     * @param ranked a ranked list
     * @param relevant the relevant candidates
     * @param k the cut-off
     * @return the normalized discounted cumulative gain
     */
    public static double ndcg(int[] ranked, Set<Integer> relevant, int k) {
        double dcg = 0;
        for (int i = 0; i < Math.min(k, ranked.length); i++) {
            dcg += relevant.contains(ranked[i]) ? 1 / log2(i + 2) : 0;
        }
        double ideal = 0;
        for (int i = 0; i < Math.min(k, relevant.size()); i++) {
            ideal += 1 / log2(i + 2);
        }
        return ideal == 0 ? 0 : dcg / ideal;
    }

    /**
     * NDCG@k with graded relevance and gains {@code 2^grade - 1}.
     *
     * @param scores the model's scores of the candidates
     * @param grades the true grades of the same candidates
     * @param k the cut-off
     * @return the normalized discounted cumulative gain, 1 when no candidate is relevant
     */
    public static double ndcg(float[] scores, int[] grades, int k) {
        int[] ranked = topK(scores, Math.min(k, scores.length), null);
        double dcg = 0;
        for (int i = 0; i < ranked.length; i++) {
            dcg += (Math.pow(2, grades[ranked[i]]) - 1) / log2(i + 2);
        }
        int[] sorted = grades.clone();
        Arrays.sort(sorted);
        double ideal = 0;
        for (int i = 0; i < Math.min(k, sorted.length); i++) {
            ideal += (Math.pow(2, sorted[sorted.length - 1 - i]) - 1) / log2(i + 2);
        }
        return ideal == 0 ? 1 : dcg / ideal;
    }

    private static double log2(double x) {
        return Math.log(x) / Math.log(2);
    }
}
