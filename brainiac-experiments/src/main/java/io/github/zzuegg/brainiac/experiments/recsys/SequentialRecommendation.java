package io.github.zzuegg.brainiac.experiments.recsys;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.SyntheticRecsys;
import io.github.zzuegg.brainiac.experimental.nn.BprMatrixFactorization;
import io.github.zzuegg.brainiac.experimental.nn.FactorizationMachine;
import io.github.zzuegg.brainiac.experimental.recsys.RankingMetrics;
import io.github.zzuegg.brainiac.experimental.recsys.SequentialRecommender;
import io.github.zzuegg.brainiac.experimental.recsys.TargetAttentionPooling;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.Recurrent;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.function.Function;

/**
 * Sequential recommendation (W7 2.2: SASRec, BERT4Rec, DIN; W7 2.6: HSTU) on
 * {@link SyntheticRecsys#sequences}: users follow planted item-transition chains inside drifting
 * interest clusters. Every user's last event is held out; models see the events before it.
 *
 * <p>Next-item prediction is measured by HR@10 and NDCG@10 over all items against a most-popular
 * baseline, a first-order Markov chain (item-to-item counts), BPR-MF (order-free) and a GRU over
 * item embeddings (GRU4Rec-lite). Click prediction (will the user click the next item?) is
 * measured by ROC-AUC: DIN's target attention against mean pooling of the same history and a
 * factorization machine over the candidate and the clicked and skipped history items.
 */
public final class SequentialRecommendation {

    /**
     * Next-item metrics.
     *
     * @param hr hit rate at 10
     * @param ndcg NDCG at 10
     */
    public record Score(double hr, double ndcg) {
    }

    private final SyntheticRecsys.Sequences data;
    private final int window;
    private final long seed;
    private final List<int[]> trainWindows = new ArrayList<>();
    private final List<long[]> trainTimes = new ArrayList<>();
    private final List<int[]> trainClicks = new ArrayList<>();

    /**
     * Creates the task.
     *
     * @param users the number of users
     * @param items the number of items
     * @param length events per user
     * @param window the history window of the models
     * @param seed the seed
     */
    public SequentialRecommendation(int users, int items, int length, int window, long seed) {
        this.data = SyntheticRecsys.sequences(users, items, length, 6, 8, 0.6, 0.2, seed);
        this.window = window;
        this.seed = seed;
        for (int u = 0; u < users; u++) {
            int[] h = data.histories()[u];
            for (int s = 0; s + window + 1 <= length - 1; s++) {
                trainWindows.add(java.util.Arrays.copyOfRange(h, s, s + window + 1));
                trainTimes.add(java.util.Arrays.copyOfRange(data.times()[u], s, s + window + 1));
                trainClicks.add(java.util.Arrays.copyOfRange(data.clicks()[u], s, s + window + 1));
            }
        }
    }

    /** @return the sequences */
    public SyntheticRecsys.Sequences data() {
        return data;
    }

    private int length() {
        return data.histories()[0].length;
    }

    /**
     * Measures a next-item scorer on every user's held-out last event.
     *
     * @param scorer histories (all events before the last), times and actions to item scores
     * @return HR@10 and NDCG@10
     */
    public Score evaluate(Scorer scorer) {
        int users = data.users();
        int last = length() - 1;
        int[][] hist = new int[users][];
        long[][] times = new long[users][];
        int[][] acts = new int[users][];
        for (int u = 0; u < users; u++) {
            hist[u] = java.util.Arrays.copyOf(data.histories()[u], last);
            times[u] = java.util.Arrays.copyOf(data.times()[u], last);
            acts[u] = java.util.Arrays.copyOf(data.clicks()[u], last);
        }
        float[][] scores = scorer.scores(hist, times, acts);
        double hr = 0;
        double ndcg = 0;
        for (int u = 0; u < users; u++) {
            int[] top = RankingMetrics.topK(scores[u], 10, null);
            int target = data.histories()[u][last];
            hr += RankingMetrics.hit(top, target, 10);
            ndcg += RankingMetrics.ndcg(top, java.util.Set.of(target), 10);
        }
        return new Score(hr / users, ndcg / users);
    }

    /** Scores every item for a batch of histories. */
    @FunctionalInterface
    public interface Scorer {
        /**
         * @param histories item histories
         * @param times their timestamps
         * @param actions their click flags
         * @return {@code [batch][items]} scores
         */
        float[][] scores(int[][] histories, long[][] times, int[][] actions);
    }

    /** @return items ranked by training frequency */
    public Score popularity() {
        float[] pop = new float[data.items()];
        for (int[] w : trainWindows) {
            pop[w[w.length - 1]]++;
        }
        return evaluate((h, t, a) -> {
            float[][] out = new float[h.length][];
            java.util.Arrays.fill(out, pop);
            return out;
        });
    }

    /** @return a first-order Markov chain of item-to-item transition counts */
    public Score markov() {
        int n = data.items();
        float[][] counts = new float[n][n];
        for (int[] w : trainWindows) {
            for (int i = 0; i + 1 < w.length; i++) {
                counts[w[i]][w[i + 1]]++;
            }
        }
        return evaluate((h, t, a) -> {
            float[][] out = new float[h.length][];
            for (int i = 0; i < h.length; i++) {
                out[i] = counts[h[i][h[i].length - 1]].clone();
            }
            return out;
        });
    }

    /**
     * @param steps BPR steps of batch 256
     * @return BPR-MF over (user, item) pairs of the training events, order-free
     */
    public Score bprMf(int steps) {
        int users = data.users();
        int items = data.items();
        BprMatrixFactorization mf = new BprMatrixFactorization(users, items, 32, seed);
        SplittableRandom r = new SplittableRandom(seed + 1);
        int last = length() - 1;
        for (int s = 0; s < steps; s++) {
            int[] u = new int[256];
            int[] p = new int[256];
            int[] q = new int[256];
            for (int b = 0; b < 256; b++) {
                u[b] = r.nextInt(users);
                p[b] = data.histories()[u[b]][r.nextInt(last)];
                q[b] = r.nextInt(items);
            }
            mf.trainStep(u, p, q);
        }
        return evaluate((h, t, a) -> {
            float[][] out = new float[h.length][];
            for (int i = 0; i < h.length; i++) {
                out[i] = mf.scores(i);
            }
            return out;
        });
    }

    private void train(int steps, int batch, Function<int[], Double> step) {
        SplittableRandom r = new SplittableRandom(seed + 2);
        for (int s = 0; s < steps; s++) {
            int[] pick = new int[batch];
            for (int b = 0; b < batch; b++) {
                pick[b] = r.nextInt(trainWindows.size());
            }
            step.apply(pick);
        }
    }

    /**
     * Trains and measures a transformer recommender.
     *
     * @param model the model, with window {@code window} (SASRec, HSTU) or {@code window + 1}
     *     (BERT4Rec)
     * @param steps steps of batch 64
     * @param withTimesAndActions whether the model reads timestamps and click actions
     * @return HR@10 and NDCG@10
     */
    public Score sequential(SequentialRecommender model, int steps, boolean withTimesAndActions) {
        train(steps, 64, pick -> {
            int[][] w = new int[pick.length][];
            long[][] t = new long[pick.length][];
            int[][] c = new int[pick.length][];
            for (int b = 0; b < pick.length; b++) {
                w[b] = trainWindows.get(pick[b]);
                t[b] = trainTimes.get(pick[b]);
                c[b] = trainClicks.get(pick[b]);
            }
            return model.trainStep(w, withTimesAndActions ? t : null, withTimesAndActions ? c : null);
        });
        return evaluate((h, t, a) -> model.scores(h, withTimesAndActions ? t : null, withTimesAndActions ? a : null));
    }

    /**
     * @param steps steps of batch 64
     * @param dim the embedding and hidden width
     * @return GRU4Rec-lite: a core GRU over item embeddings, tied output, next-item cross-entropy
     *     at every position
     */
    public Score gru(int steps, int dim) {
        int items = data.items();
        SplittableRandom r = new SplittableRandom(seed + 3);
        Var emb = Var.param(Tensor.normal(Shape.of(items, dim), 0, 0.1f, r));
        Layer gru = Recurrent.gru(dim).returnSequences(true).create(Shape.of(window, dim), r);
        List<Var> params = new ArrayList<>(gru.parameters());
        params.add(emb);
        Optimizer opt = Optimizer.adam(5e-3f);
        Function<int[][], Var> hidden = in -> {
            int b = in.length;
            float[] ids = new float[b * window];
            for (int i = 0; i < b; i++) {
                for (int j = 0; j < window; j++) {
                    ids[i * window + j] = in[i][j];
                }
            }
            return gru.forward(emb.gather(0, Tensor.of(Shape.of(b * window), ids)).reshape(b, window, dim), true);
        };
        train(steps, 64, pick -> {
            int b = pick.length;
            int[][] in = new int[b][window];
            float[] target = new float[b * window];
            for (int i = 0; i < b; i++) {
                int[] w = trainWindows.get(pick[i]);
                System.arraycopy(w, 0, in[i], 0, window);
                for (int j = 0; j < window; j++) {
                    target[i * window + j] = w[j + 1];
                }
            }
            Var h = hidden.apply(in).reshape(b * window, dim);
            Var loss = Functional.crossEntropy(h.matmul(emb.transpose()), Tensor.of(Shape.of(b * window), target));
            loss.backward();
            opt.step(params);
            return (double) loss.value().item();
        });
        return evaluate((h, t, a) -> Var.noGrad(() -> {
            int b = h.length;
            int[][] in = new int[b][window];
            for (int i = 0; i < b; i++) {
                System.arraycopy(h[i], h[i].length - window, in[i], 0, window);
            }
            Var last = hidden.apply(in).slice(1, window - 1, window).reshape(b, dim);
            float[] s = last.matmul(emb.transpose()).value().toArray();
            float[][] out = new float[b][items];
            for (int i = 0; i < b; i++) {
                System.arraycopy(s, i * items, out[i], 0, items);
            }
            return out;
        }));
    }

    /**
     * Runs the next-item comparison.
     *
     * @param steps training steps per neural model
     * @return model name to HR@10 and NDCG@10
     */
    public Map<String, Score> nextItem(int steps) {
        Map<String, Score> out = new LinkedHashMap<>();
        int items = data.items();
        out.put("popularity", popularity());
        out.put("Markov", markov());
        out.put("BPR-MF", bprMf(steps));
        out.put("GRU4Rec", gru(steps, 32));
        out.put("SASRec", sequential(SequentialRecommender.sasRec(items, window, 32, 2, 1, seed), steps, false));
        out.put("BERT4Rec", sequential(SequentialRecommender.bert4Rec(items, window, 32, 2, 1, seed), steps,
                false));
        return out;
    }

    /**
     * Retrieval and ranking with one model (W7 2.6): SASRec and HSTU recommenders that read the
     * click/skip action of every event (HSTU also the timestamps) and are trained on next-item
     * cross-entropy plus a click head. Measured by HR@10 / NDCG@10 of the held-out last event and
     * the ROC-AUC of predicting whether it was clicked.
     *
     * @param steps steps of batch 64 per model
     * @param widths the HSTU model widths (a small scaling sweep)
     * @return model name to {@code {HR@10, NDCG@10, click AUC, parameters}}
     */
    public Map<String, double[]> transduction(int steps, int... widths) {
        Map<String, double[]> out = new LinkedHashMap<>();
        int items = data.items();
        out.put("SASRec d=32", unified(SequentialRecommender.sasRec(items, window, 32, 2, 1, seed), steps));
        for (int w : widths) {
            out.put("HSTU d=" + w, unified(SequentialRecommender.hstu(items, window, w, 2, 1, seed), steps));
        }
        return out;
    }

    private double[] unified(SequentialRecommender model, int steps) {
        Score s = sequential(model, steps, true);
        int users = data.users();
        int last = length() - 1;
        int[][] hist = new int[users][];
        long[][] times = new long[users][];
        int[][] acts = new int[users][];
        int[] cand = new int[users];
        long[] candTimes = new long[users];
        float[] labels = new float[users];
        for (int u = 0; u < users; u++) {
            hist[u] = java.util.Arrays.copyOf(data.histories()[u], last);
            times[u] = java.util.Arrays.copyOf(data.times()[u], last);
            acts[u] = java.util.Arrays.copyOf(data.clicks()[u], last);
            cand[u] = data.histories()[u][last];
            candTimes[u] = data.times()[u][last];
            labels[u] = data.clicks()[u][last];
        }
        double auc = BinaryBench.score(model.clickLogits(hist, times, acts, cand, candTimes), labels, 0).auc();
        return new double[] {s.hr(), s.ndcg(), auc, model.size()};
    }

    /**
     * Click prediction: will the user click the next item? Trains on the first 80% of users and
     * tests on the rest.
     *
     * @param steps gradient steps of batch 64
     * @return model name to test ROC-AUC
     */
    public Map<String, Double> clickAuc(int steps) {
        int users = data.users();
        int items = data.items();
        int split = users * 4 / 5;
        List<int[]> trainRows = new ArrayList<>();
        List<int[]> testRows = new ArrayList<>();
        for (int u = 0; u < users; u++) {
            for (int t = window; t < length(); t++) {
                (u < split ? trainRows : testRows).add(new int[] {u, t});
            }
        }
        Map<String, Double> out = new LinkedHashMap<>();
        out.put("FM", fmClicks(trainRows, testRows, steps, items));
        out.put("mean pooling", dinClicks(trainRows, testRows, steps, items, false));
        out.put("DIN", dinClicks(trainRows, testRows, steps, items, true));
        return out;
    }

    private Tensor fmRows(List<int[]> rows, int[] pick, int items) {
        int width = 3 * items;
        float[] x = new float[pick.length * width];
        for (int b = 0; b < pick.length; b++) {
            int[] row = rows.get(pick[b]);
            int u = row[0];
            int t = row[1];
            x[b * width + data.histories()[u][t]] = 1;
            for (int j = t - window; j < t; j++) {
                int off = data.clicks()[u][j] == 1 ? items : 2 * items;
                x[b * width + off + data.histories()[u][j]] = 1f / window;
            }
        }
        return Tensor.of(Shape.of(pick.length, width), x);
    }

    private double fmClicks(List<int[]> trainRows, List<int[]> testRows, int steps, int items) {
        FactorizationMachine fm = new FactorizationMachine(3 * items, 8, new SplittableRandom(seed));
        return clickTraining(trainRows, testRows, steps, fm.parameters(),
                pick -> fm.forward(Var.of(fmRows(pick.rows(), pick.ids(), items))), 0.01f);
    }

    private record Pick(List<int[]> rows, int[] ids) {
    }

    private double dinClicks(List<int[]> trainRows, List<int[]> testRows, int steps, int items, boolean din) {
        int d = 16;
        SplittableRandom r = new SplittableRandom(seed + 5);
        Var emb = Var.param(Tensor.normal(Shape.of(items, d), 0, 0.1f, r));
        Var clickEmb = Var.param(Tensor.normal(Shape.of(2, d), 0, 0.1f, r));
        TargetAttentionPooling pool = new TargetAttentionPooling(d, 16, r);
        Var w1 = Var.param(Tensor.normal(Shape.of(3 * d, 32), 0, 0.2f, r));
        Var b1 = Var.param(Tensor.zeros(Shape.of(32)));
        Var w2 = Var.param(Tensor.normal(Shape.of(32, 1), 0, 0.2f, r));
        List<Var> params = new ArrayList<>(List.of(emb, clickEmb, w1, b1, w2));
        if (din) {
            params.addAll(pool.parameters());
        }
        Function<Pick, Var> model = pick -> {
            int b = pick.ids().length;
            float[] hist = new float[b * window];
            float[] flags = new float[b * window];
            float[] cand = new float[b];
            for (int i = 0; i < b; i++) {
                int[] row = pick.rows().get(pick.ids()[i]);
                for (int j = 0; j < window; j++) {
                    hist[i * window + j] = data.histories()[row[0]][row[1] - window + j];
                    flags[i * window + j] = data.clicks()[row[0]][row[1] - window + j];
                }
                cand[i] = data.histories()[row[0]][row[1]];
            }
            Var h = emb.gather(0, Tensor.of(Shape.of(b * window), hist))
                    .add(clickEmb.gather(0, Tensor.of(Shape.of(b * window), flags))).reshape(b, window, d);
            Var c = emb.gather(0, Tensor.of(Shape.of(b), cand));
            Var user = din ? pool.forward(List.of(h, c), true) : h.mean(1, false);
            return Var.concat(1, user, c, user.mul(c)).matmul(w1).add(b1).relu().matmul(w2);
        };
        return clickTraining(trainRows, testRows, steps, params, model, 5e-3f);
    }

    private double clickTraining(List<int[]> trainRows, List<int[]> testRows, int steps, List<Var> params,
            Function<Pick, Var> model, float lr) {
        Optimizer opt = Optimizer.adam(lr);
        SplittableRandom r = new SplittableRandom(seed + 9);
        for (int s = 0; s < steps; s++) {
            int[] ids = new int[64];
            float[] y = new float[64];
            for (int b = 0; b < 64; b++) {
                ids[b] = r.nextInt(trainRows.size());
                int[] row = trainRows.get(ids[b]);
                y[b] = data.clicks()[row[0]][row[1]];
            }
            Var z = model.apply(new Pick(trainRows, ids));
            z.softplus().sub(z.mul(Tensor.of(Shape.of(64, 1), y))).mean().backward();
            opt.step(params);
        }
        int[] all = new int[testRows.size()];
        float[] labels = new float[all.length];
        for (int i = 0; i < all.length; i++) {
            all[i] = i;
            int[] row = testRows.get(i);
            labels[i] = data.clicks()[row[0]][row[1]];
        }
        float[] logits = Var.noGrad(() -> model.apply(new Pick(testRows, all))).value().toArray();
        return BinaryBench.score(logits, labels, 0).auc();
    }
}
