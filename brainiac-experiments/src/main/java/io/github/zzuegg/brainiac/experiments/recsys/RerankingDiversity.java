package io.github.zzuegg.brainiac.experiments.recsys;

import io.github.zzuegg.brainiac.experimental.data.ClickLogs;
import io.github.zzuegg.brainiac.experimental.nn.TwoTowerRetrieval;
import io.github.zzuegg.brainiac.experimental.recsys.DiverseReranker;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Diversity re-ranking (W9 1.4) on {@link ClickLogs#topics}: items belong to latent topics and each
 * user likes two or three topics, one strongly, and is satisfied once per covered topic. A
 * relevance model ({@link TwoTowerRetrieval} trained on sampled positives, or a noisy view of the
 * true relevance standing in for a collaborative-filtering model) proposes its top 30 candidates;
 * re-rankers pick 10 of them: relevance order, random, {@link DiverseReranker#mmr} and
 * {@link DiverseReranker#dpp} along their trade-off parameters. Metrics: NDCG@10 on the true
 * graded relevance, alpha-NDCG@10 (alpha 0.5) with topics as subtopics, and intra-list diversity
 * (mean pairwise cosine distance).
 */
public final class RerankingDiversity {

    private RerankingDiversity() {
    }

    private static final int POOL = 30;
    private static final int K = 10;

    /**
     * Runs the comparison.
     *
     * @param useTwoTower whether candidates come from a trained two-tower model (else a noisy
     *     relevance oracle)
     * @param steps two-tower training steps
     * @param seed the seed
     * @return re-ranker name to {@code {NDCG@10, alpha-NDCG@10, intra-list diversity}}
     */
    public static Map<String, double[]> run(boolean useTwoTower, int steps, long seed) {
        int users = 150;
        int items = 400;
        int dim = 16;
        ClickLogs.Topics data = ClickLogs.topics(users, items, 10, dim, seed);
        double[][] scores = useTwoTower ? twoTower(data, users, items, dim, steps, seed) : noisy(data, users, items,
                seed);
        Map<String, double[]> out = new LinkedHashMap<>();
        SplittableRandom r = new SplittableRandom(seed);
        out.put("relevance only", evaluate(data, scores, (rel, emb) -> firstK(rel.length)));
        out.put("random from top 30", evaluate(data, scores, (rel, emb) -> randomK(rel.length, r)));
        for (double lambda : new double[] {0.9, 0.7, 0.5, 0.3}) {
            out.put("MMR lambda " + lambda, evaluate(data, scores, (rel, emb) -> DiverseReranker.mmr(rel, emb, K,
                    lambda)));
        }
        for (double theta : new double[] {0.95, 0.9, 0.7, 0.5}) {
            out.put("DPP theta " + theta, evaluate(data, scores, (rel, emb) -> DiverseReranker.dpp(rel, emb, K,
                    theta)));
        }
        return out;
    }

    private interface Reranker {
        int[] pick(double[] relevance, double[][] embeddings);
    }

    private static int[] firstK(int n) {
        int[] out = new int[Math.min(K, n)];
        for (int i = 0; i < out.length; i++) {
            out[i] = i;
        }
        return out;
    }

    private static int[] randomK(int n, SplittableRandom r) {
        int[] p = new int[n];
        for (int i = 0; i < n; i++) {
            p[i] = i;
        }
        for (int i = n - 1; i > 0; i--) {
            int j = r.nextInt(i + 1);
            int t = p[i];
            p[i] = p[j];
            p[j] = t;
        }
        return java.util.Arrays.copyOf(p, K);
    }

    private static double[][] noisy(ClickLogs.Topics data, int users, int items, long seed) {
        SplittableRandom r = new SplittableRandom(seed + 3);
        double[][] s = new double[users][items];
        for (int u = 0; u < users; u++) {
            for (int i = 0; i < items; i++) {
                s[u][i] = data.relevance(u, i) + 0.15 * r.nextGaussian();
            }
        }
        return s;
    }

    private static double[][] twoTower(ClickLogs.Topics data, int users, int items, int dim, int steps, long seed) {
        Model userTower = Model.sequential().input(dim).dense(32, Activation.RELU).dense(16).seed(seed).build();
        Model itemTower = Model.sequential().input(dim).dense(32, Activation.RELU).dense(16).seed(seed + 1).build();
        TwoTowerRetrieval tt = TwoTowerRetrieval.of(userTower, itemTower);
        SplittableRandom r = new SplittableRandom(seed + 2);
        int batch = 64;
        double[][] cdf = new double[users][items];
        for (int u = 0; u < users; u++) {
            double acc = 0;
            for (int i = 0; i < items; i++) {
                acc += Math.pow(data.relevance(u, i), 4);
                cdf[u][i] = acc;
            }
            for (int i = 0; i < items; i++) {
                cdf[u][i] /= acc;
            }
        }
        for (int s = 0; s < steps; s++) {
            float[] ub = new float[batch * dim];
            float[] ib = new float[batch * dim];
            for (int b = 0; b < batch; b++) {
                int u = r.nextInt(users);
                int k = java.util.Arrays.binarySearch(cdf[u], r.nextDouble());
                int i = Math.min(items - 1, k < 0 ? -k - 1 : k);
                for (int j = 0; j < dim; j++) {
                    ub[b * dim + j] = (float) data.userFeatures()[u][j];
                    ib[b * dim + j] = (float) data.embedding()[i][j];
                }
            }
            tt.trainStep(Tensor.of(Shape.of(batch, dim), ub), Tensor.of(Shape.of(batch, dim), ib));
        }
        float[] ue = tt.embedUsers(matrix(data.userFeatures())).toArray();
        float[] ie = tt.embedItems(matrix(data.embedding())).toArray();
        int e = 16;
        double[][] s = new double[users][items];
        for (int u = 0; u < users; u++) {
            for (int i = 0; i < items; i++) {
                double v = 0;
                for (int j = 0; j < e; j++) {
                    v += ue[u * e + j] * ie[i * e + j];
                }
                s[u][i] = v;
            }
        }
        return s;
    }

    private static Tensor matrix(double[][] x) {
        int d = x[0].length;
        float[] f = new float[x.length * d];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < d; j++) {
                f[i * d + j] = (float) x[i][j];
            }
        }
        return Tensor.of(Shape.of(x.length, d), f);
    }

    private static double[] evaluate(ClickLogs.Topics data, double[][] scores, Reranker reranker) {
        int users = scores.length;
        int items = scores[0].length;
        double ndcg = 0;
        double alpha = 0;
        double ild = 0;
        for (int u = 0; u < users; u++) {
            double[] row = scores[u];
            int[] pool = java.util.stream.IntStream.range(0, items).boxed()
                    .sorted((a, b) -> Double.compare(row[b], row[a])).limit(POOL).mapToInt(Integer::intValue)
                    .toArray();
            double[] rel = new double[POOL];
            double[][] emb = new double[POOL][];
            for (int p = 0; p < POOL; p++) {
                rel[p] = row[pool[p]];
                emb[p] = data.embedding()[pool[p]];
            }
            int[] pick = reranker.pick(rel, emb);
            int[] list = new int[pick.length];
            for (int p = 0; p < pick.length; p++) {
                list[p] = pool[pick[p]];
            }
            ndcg += ndcg(data, u, list, items);
            alpha += alphaNdcg(data, u, list, items, 0.5);
            ild += diversity(data, list);
        }
        return new double[] {ndcg / users, alpha / users, ild / users};
    }

    private static double ndcg(ClickLogs.Topics data, int u, int[] list, int items) {
        double dcg = 0;
        for (int p = 0; p < list.length; p++) {
            dcg += data.relevance(u, list[p]) / log2(p + 2);
        }
        double[] all = new double[items];
        for (int i = 0; i < items; i++) {
            all[i] = data.relevance(u, i);
        }
        java.util.Arrays.sort(all);
        double ideal = 0;
        for (int p = 0; p < K; p++) {
            ideal += all[items - 1 - p] / log2(p + 2);
        }
        return dcg / ideal;
    }

    /** alpha-DCG with topics as subtopics, normalised by the greedy ideal list over all items. */
    private static double alphaNdcg(ClickLogs.Topics data, int u, int[] list, int items, double a) {
        int topics = data.taste()[u].length;
        double dcg = alphaDcg(data, u, list, topics, a);
        int[] ideal = new int[K];
        int[] seen = new int[topics];
        boolean[] used = new boolean[items];
        for (int p = 0; p < K; p++) {
            int best = -1;
            double bestGain = -1;
            for (int i = 0; i < items; i++) {
                if (used[i]) {
                    continue;
                }
                double g = data.taste()[u][data.topic()[i]] * data.quality()[i]
                        * Math.pow(1 - a, seen[data.topic()[i]]);
                if (g > bestGain) {
                    bestGain = g;
                    best = i;
                }
            }
            ideal[p] = best;
            used[best] = true;
            seen[data.topic()[best]]++;
        }
        return dcg / alphaDcg(data, u, ideal, topics, a);
    }

    private static double alphaDcg(ClickLogs.Topics data, int u, int[] list, int topics, double a) {
        int[] seen = new int[topics];
        double dcg = 0;
        for (int p = 0; p < list.length; p++) {
            int t = data.topic()[list[p]];
            dcg += data.taste()[u][t] * data.quality()[list[p]] * Math.pow(1 - a, seen[t]) / log2(p + 2);
            seen[t]++;
        }
        return dcg;
    }

    private static double diversity(ClickLogs.Topics data, int[] list) {
        double s = 0;
        int pairs = 0;
        for (int i = 0; i < list.length; i++) {
            for (int j = i + 1; j < list.length; j++) {
                double[] a = data.embedding()[list[i]];
                double[] b = data.embedding()[list[j]];
                double dot = 0;
                double na = 0;
                double nb = 0;
                for (int k = 0; k < a.length; k++) {
                    dot += a[k] * b[k];
                    na += a[k] * a[k];
                    nb += b[k] * b[k];
                }
                s += 1 - dot / Math.sqrt(na * nb);
                pairs++;
            }
        }
        return pairs == 0 ? 0 : s / pairs;
    }

    private static double log2(double x) {
        return Math.log(x) / Math.log(2);
    }
}
