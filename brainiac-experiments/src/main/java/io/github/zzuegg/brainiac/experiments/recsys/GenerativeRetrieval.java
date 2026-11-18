package io.github.zzuegg.brainiac.experiments.recsys;

import io.github.zzuegg.brainiac.data.VectorIndex;
import io.github.zzuegg.brainiac.experimental.data.SyntheticRecsys;
import io.github.zzuegg.brainiac.experimental.nn.TwoTowerRetrieval;
import io.github.zzuegg.brainiac.experimental.recsys.GenerativeRetriever;
import io.github.zzuegg.brainiac.experimental.recsys.RankingMetrics;
import io.github.zzuegg.brainiac.experimental.recsys.SemanticIdTokenizer;
import io.github.zzuegg.brainiac.experimental.recsys.SequentialRecommender;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Generative retrieval with semantic IDs (W7 2.5, TIGER) against SASRec (item ids) and a two-tower
 * model over content features served from an HNSW index ({@link VectorIndex#hnsw}).
 *
 * <p>Data: {@link SyntheticRecsys#sequences}, whose item content vectors place each planted
 * transition chain on a smooth ring. Every tenth item is <em>cold</em>: no training example
 * contains it. Each user's last event is held out; the metric is recall@10 of it (one relevant
 * item, so recall equals the hit rate), over all users and over the users whose held-out item is
 * cold. TIGER is run at two codebook depths, with the share of items whose codes collide.
 */
public final class GenerativeRetrieval {

    /**
     * One model's result.
     *
     * @param recall recall@10 over all users
     * @param coldRecall recall@10 over users whose held-out item is cold
     * @param collisions the semantic-ID collision rate (0 for other models)
     */
    public record Score(double recall, double coldRecall, double collisions) {
    }

    private final SyntheticRecsys.Sequences data;
    private final int history;
    private final long seed;
    private final List<int[]> windows = new ArrayList<>();

    /**
     * Creates the task.
     *
     * @param users users
     * @param items items
     * @param length events per user
     * @param history items of history the models read
     * @param seed the seed
     */
    public GenerativeRetrieval(int users, int items, int length, int history, long seed) {
        this.data = SyntheticRecsys.sequences(users, items, length, 6, 8, 0.7, 0.2, seed);
        this.history = history;
        this.seed = seed;
        for (int[] h : data.histories()) {
            for (int s = 0; s + history + 1 <= length - 1; s++) {
                int[] w = Arrays.copyOfRange(h, s, s + history + 1);
                if (Arrays.stream(w).noneMatch(GenerativeRetrieval::cold)) {
                    windows.add(w);
                }
            }
        }
    }

    private static boolean cold(int item) {
        return item % 10 == 0;
    }

    private int[][] testHistories() {
        int users = data.users();
        int last = data.histories()[0].length - 1;
        int[][] h = new int[users][];
        for (int u = 0; u < users; u++) {
            h[u] = Arrays.copyOfRange(data.histories()[u], last - history, last);
        }
        return h;
    }

    private Score score(int[][] top, double collisions) {
        int users = data.users();
        int last = data.histories()[0].length - 1;
        double all = 0;
        double cold = 0;
        int colds = 0;
        for (int u = 0; u < users; u++) {
            int target = data.histories()[u][last];
            double hit = RankingMetrics.hit(top[u], target, 10);
            all += hit;
            if (cold(target)) {
                cold += hit;
                colds++;
            }
        }
        return new Score(all / users, colds == 0 ? 0 : cold / colds, collisions);
    }

    private int[][] batch(SplittableRandom r, int size) {
        int[][] b = new int[size][];
        for (int i = 0; i < size; i++) {
            b[i] = windows.get(r.nextInt(windows.size()));
        }
        return b;
    }

    /**
     * @param depth codebooks of the semantic IDs
     * @param codes codes per codebook
     * @param steps training steps of batch 64
     * @return TIGER
     */
    public Score tiger(int depth, int codes, int steps) {
        SemanticIdTokenizer ids = SemanticIdTokenizer.fit(data.content(), depth, codes, 100, seed);
        GenerativeRetriever g = new GenerativeRetriever(ids, history, 32, 2, 1, seed);
        SplittableRandom r = new SplittableRandom(seed + 1);
        for (int s = 0; s < steps; s++) {
            int[][] b = batch(r, 64);
            int[][] h = new int[b.length][];
            int[] next = new int[b.length];
            for (int i = 0; i < b.length; i++) {
                h[i] = Arrays.copyOf(b[i], history);
                next[i] = b[i][history];
            }
            g.trainStep(h, next);
        }
        return score(g.recommend(testHistories(), 10), ids.collisionRate());
    }

    /**
     * @param steps training steps of batch 64
     * @return SASRec over item ids
     */
    public Score sasRec(int steps) {
        SequentialRecommender m = SequentialRecommender.sasRec(data.items(), history, 32, 2, 1, seed);
        SplittableRandom r = new SplittableRandom(seed + 1);
        for (int s = 0; s < steps; s++) {
            m.trainStep(batch(r, 64), null, null);
        }
        float[][] scores = m.scores(testHistories(), null, null);
        int[][] top = new int[scores.length][];
        for (int u = 0; u < scores.length; u++) {
            top[u] = RankingMetrics.topK(scores[u], 10, null);
        }
        return score(top, 0);
    }

    private Tensor userFeatures(int[][] h) {
        int d = data.content().shape().dim(1);
        float[] c = data.content().toArray();
        float[] out = new float[h.length * history * d];
        for (int i = 0; i < h.length; i++) {
            for (int j = 0; j < history; j++) {
                System.arraycopy(c, h[i][h[i].length - history + j] * d, out, (i * history + j) * d, d);
            }
        }
        return Tensor.of(Shape.of(h.length, history * d), out);
    }

    /**
     * @param steps training steps of batch 64
     * @return a two-tower model over content (the last items' content for users), served from HNSW
     */
    public Score twoTower(int steps) {
        int d = data.content().shape().dim(1);
        Model user = Model.sequential().input(history * d).dense(64, Activation.RELU).dense(32).seed(seed).build();
        Model item = Model.sequential().input(d).dense(64, Activation.RELU).dense(32).seed(seed + 1).build();
        TwoTowerRetrieval tt = TwoTowerRetrieval.of(user, item).temperature(0.1f);
        SplittableRandom r = new SplittableRandom(seed + 1);
        float[] c = data.content().toArray();
        for (int s = 0; s < steps; s++) {
            int[][] b = batch(r, 64);
            float[] it = new float[b.length * d];
            for (int i = 0; i < b.length; i++) {
                System.arraycopy(c, b[i][history] * d, it, i * d, d);
            }
            tt.trainStep(userFeatures(b), Tensor.of(Shape.of(b.length, d), it));
        }
        Tensor items = tt.embedItems(data.content());
        VectorIndex index = VectorIndex.hnsw(32, VectorIndex.Metric.DOT, 16, 100, 50, seed);
        index.add(items);
        Tensor users = tt.embedUsers(userFeatures(testHistories()));
        int[][] top = new int[data.users()][];
        for (int u = 0; u < top.length; u++) {
            top[u] = index.search(users.slice(0, u, u + 1).reshape(32), 10).stream().mapToInt(VectorIndex.Hit::id)
                    .toArray();
        }
        return score(top, 0);
    }

    /**
     * @param steps training steps per model
     * @return model name to score
     */
    public Map<String, Score> run(int steps) {
        Map<String, Score> out = new LinkedHashMap<>();
        out.put("SASRec", sasRec(steps));
        out.put("two-tower+HNSW", twoTower(steps));
        out.put("TIGER depth 1", tiger(1, 32, steps));
        out.put("TIGER depth 2", tiger(2, 16, steps));
        return out;
    }
}
