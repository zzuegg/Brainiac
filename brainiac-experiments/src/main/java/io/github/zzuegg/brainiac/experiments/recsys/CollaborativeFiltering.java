package io.github.zzuegg.brainiac.experiments.recsys;

import io.github.zzuegg.brainiac.experimental.data.SyntheticRecsys;
import io.github.zzuegg.brainiac.experimental.nn.BprMatrixFactorization;
import io.github.zzuegg.brainiac.experimental.recsys.EaseRecommender;
import io.github.zzuegg.brainiac.experimental.recsys.LightGcn;
import io.github.zzuegg.brainiac.experimental.recsys.MultVae;
import io.github.zzuegg.brainiac.experimental.recsys.NeuralCollaborativeFiltering;
import io.github.zzuegg.brainiac.experimental.recsys.RankingMetrics;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.SplittableRandom;
import java.util.function.IntFunction;

/**
 * Collaborative filtering from implicit feedback (W7 2.1 LightGCN, 2.3 Mult-VAE and EASE)
 * against BPR matrix factorization, item-kNN with cosine similarity and popularity.
 *
 * <p>Data: {@link SyntheticRecsys#preferences}: every user's 15 most relevant items under a
 * hidden low-rank taste model; 10 of them, chosen at random, are observed and the other 5 are
 * held out. Every model scores all unobserved items and is measured by recall@10 and NDCG@10 of
 * the held-out items. LightGCN is run at depths {@code K = 0..4} ({@code K = 0} is BPR-MF with the
 * same code path) and NGCF-lite (transforms and non-linearities kept) is the ablation.
 */
public final class CollaborativeFiltering {

    /**
     * One model's result.
     *
     * @param recall recall@10 of the held-out items
     * @param ndcg NDCG@10 of the held-out items
     * @param millis fit time in milliseconds
     */
    public record Score(double recall, double ndcg, long millis) {
    }

    private final int users;
    private final int items;
    private final List<Set<Integer>> observed = new ArrayList<>();
    private final List<Set<Integer>> held = new ArrayList<>();
    private final int[] edgeUsers;
    private final int[] edgeItems;
    private final double[][] matrix;
    private final long seed;

    /**
     * Creates the task.
     *
     * @param users the number of users
     * @param items the number of items
     * @param seed the seed
     */
    public CollaborativeFiltering(int users, int items, long seed) {
        this.users = users;
        this.items = items;
        this.seed = seed;
        SyntheticRecsys.Preferences data = SyntheticRecsys.preferences(users, items, 6, 8, 0.1, 15, seed);
        SplittableRandom r = new SplittableRandom(seed);
        matrix = new double[users][items];
        List<int[]> edges = new ArrayList<>();
        for (int u = 0; u < users; u++) {
            int[] rel = data.relevant()[u].clone();
            for (int i = rel.length - 1; i > 0; i--) {
                int j = r.nextInt(i + 1);
                int t = rel[i];
                rel[i] = rel[j];
                rel[j] = t;
            }
            Set<Integer> obs = new HashSet<>();
            Set<Integer> out = new HashSet<>();
            for (int k = 0; k < rel.length; k++) {
                if (k < 10) {
                    obs.add(rel[k]);
                    matrix[u][rel[k]] = 1;
                    edges.add(new int[] {u, rel[k]});
                } else {
                    out.add(rel[k]);
                }
            }
            observed.add(obs);
            held.add(out);
        }
        edgeUsers = edges.stream().mapToInt(e -> e[0]).toArray();
        edgeItems = edges.stream().mapToInt(e -> e[1]).toArray();
    }

    /**
     * Measures a scorer.
     *
     * @param scorer the score of every item for a user
     * @param millis the fit time to report
     * @return recall@10 and NDCG@10
     */
    public Score evaluate(IntFunction<float[]> scorer, long millis) {
        double recall = 0;
        double ndcg = 0;
        for (int u = 0; u < users; u++) {
            int[] top = RankingMetrics.topK(scorer.apply(u), 10, observed.get(u));
            recall += RankingMetrics.recall(top, held.get(u), 10);
            ndcg += RankingMetrics.ndcg(top, held.get(u), 10);
        }
        return new Score(recall / users, ndcg / users, millis);
    }

    private interface BprStep {
        void step(int[] u, int[] p, int[] n);
    }

    private void trainBpr(BprStep model, int steps) {
        SplittableRandom r = new SplittableRandom(seed + 7);
        int batch = 256;
        for (int s = 0; s < steps; s++) {
            int[] u = new int[batch];
            int[] p = new int[batch];
            int[] n = new int[batch];
            for (int b = 0; b < batch; b++) {
                int e = r.nextInt(edgeUsers.length);
                u[b] = edgeUsers[e];
                p[b] = edgeItems[e];
                do {
                    n[b] = r.nextInt(items);
                } while (matrix[u[b]][n[b]] != 0);
            }
            model.step(u, p, n);
        }
    }

    /**
     * @param steps BPR steps of batch 256
     * @return BPR matrix factorization
     */
    public Score bprMf(int steps) {
        long t = System.currentTimeMillis();
        BprMatrixFactorization mf = new BprMatrixFactorization(users, items, 32, seed);
        trainBpr(mf::trainStep, steps);
        return evaluate(mf::scores, System.currentTimeMillis() - t);
    }

    /**
     * @param layers the depth K
     * @param ngcf whether to keep transforms (NGCF-lite)
     * @param steps BPR steps of batch 256
     * @return LightGCN or NGCF-lite
     */
    public Score lightGcn(int layers, boolean ngcf, int steps) {
        long t = System.currentTimeMillis();
        LightGcn g = ngcf ? LightGcn.ngcf(users, items, edgeUsers, edgeItems, 32, layers, seed)
                : LightGcn.of(users, items, edgeUsers, edgeItems, 32, layers, seed);
        trainBpr(g::trainStep, steps);
        return evaluate(g::scores, System.currentTimeMillis() - t);
    }

    /**
     * NeuMF (W7 2.4), trained with binary cross-entropy against 4 sampled negatives per pair.
     *
     * @param steps steps of 128 observed pairs
     * @return NeuMF
     */
    public Score neuMf(int steps) {
        long t = System.currentTimeMillis();
        NeuralCollaborativeFiltering ncf = new NeuralCollaborativeFiltering(users, items, 16, seed);
        SplittableRandom r = new SplittableRandom(seed + 11);
        for (int s = 0; s < steps; s++) {
            int[] u = new int[128];
            int[] p = new int[128];
            for (int b = 0; b < 128; b++) {
                int e = r.nextInt(edgeUsers.length);
                u[b] = edgeUsers[e];
                p[b] = edgeItems[e];
            }
            ncf.trainStep(u, p, 4);
        }
        return evaluate(ncf::scores, System.currentTimeMillis() - t);
    }

    /**
     * @param lambda the L2 regularisation
     * @return EASE
     */
    public Score ease(double lambda) {
        long t = System.currentTimeMillis();
        EaseRecommender ease = EaseRecommender.fit(matrix, lambda);
        return evaluate(ease::scores, System.currentTimeMillis() - t);
    }

    /**
     * @param steps training steps of 64 users
     * @return Mult-VAE
     */
    public Score multVae(int steps) {
        long t = System.currentTimeMillis();
        MultVae vae = new MultVae(items, 64, 16, 0.5f, 0.2f, steps / 2, seed);
        SplittableRandom r = new SplittableRandom(seed + 3);
        int batch = 64;
        for (int s = 0; s < steps; s++) {
            float[] rows = new float[batch * items];
            for (int b = 0; b < batch; b++) {
                int u = r.nextInt(users);
                for (int i = 0; i < items; i++) {
                    rows[b * items + i] = (float) matrix[u][i];
                }
            }
            vae.trainStep(Tensor.of(Shape.of(batch, items), rows));
        }
        return evaluate(u -> vae.scores(matrix[u]), System.currentTimeMillis() - t);
    }

    /** @return item-kNN: items scored by their summed cosine similarity to the user's items */
    public Score itemKnn() {
        long t = System.currentTimeMillis();
        double[][] sim = new double[items][items];
        double[] norm = new double[items];
        for (double[] row : matrix) {
            for (int i = 0; i < items; i++) {
                norm[i] += row[i];
            }
        }
        for (double[] row : matrix) {
            for (int i = 0; i < items; i++) {
                if (row[i] == 0) {
                    continue;
                }
                for (int j = 0; j < items; j++) {
                    sim[i][j] += row[j];
                }
            }
        }
        for (int i = 0; i < items; i++) {
            for (int j = 0; j < items; j++) {
                sim[i][j] = i == j ? 0 : sim[i][j] / Math.sqrt(Math.max(1, norm[i] * norm[j]));
            }
        }
        long ms = System.currentTimeMillis() - t;
        return evaluate(u -> {
            float[] s = new float[items];
            for (int i : observed.get(u)) {
                for (int j = 0; j < items; j++) {
                    s[j] += (float) sim[i][j];
                }
            }
            return s;
        }, ms);
    }

    /** @return items ranked by how many users interacted with them */
    public Score popularity() {
        float[] pop = new float[items];
        for (double[] row : matrix) {
            for (int i = 0; i < items; i++) {
                pop[i] += (float) row[i];
            }
        }
        return evaluate(u -> pop, 0);
    }

    /**
     * Runs every model.
     *
     * @param steps BPR steps for the embedding models
     * @param maxLayers the deepest LightGCN
     * @return model name to score
     */
    public Map<String, Score> run(int steps, int maxLayers) {
        Map<String, Score> out = new LinkedHashMap<>();
        out.put("popularity", popularity());
        out.put("item-kNN", itemKnn());
        out.put("BPR-MF", bprMf(steps));
        for (int k = 0; k <= maxLayers; k++) {
            out.put("LightGCN K=" + k, lightGcn(k, false, steps));
        }
        out.put("NGCF-lite K=2", lightGcn(2, true, steps));
        out.put("EASE", ease(50));
        out.put("Mult-VAE", multVae(steps));
        return out;
    }
}
