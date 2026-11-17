package io.github.zzuegg.brainiac.experiments.graphs;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.SyntheticRecsys;
import io.github.zzuegg.brainiac.experimental.nn.BprMatrixFactorization;
import io.github.zzuegg.brainiac.experimental.nn.CrossNetwork;
import io.github.zzuegg.brainiac.experimental.nn.FactorizationMachine;
import io.github.zzuegg.brainiac.experimental.nn.TabNetLite;
import io.github.zzuegg.brainiac.experimental.nn.TwoTowerRetrieval;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.SplittableRandom;
import java.util.function.UnaryOperator;

/**
 * Recommendation and tabular models against their simple baselines.
 * <ul>
 *   <li>{@link #retrieval}: two-tower retrieval from features versus BPR matrix
 *       factorization from interaction ids versus random, measured by recall@10 of held-out
 *       interactions; the towers also serve users never seen in training (cold start),
 *       which an id-based factorization cannot;</li>
 *   <li>{@link #clicks}: click prediction on one-hot fields where a field-pair interaction
 *       drives clicks: logistic regression versus a factorization machine and a
 *       DCN-V2 cross network;</li>
 *   <li>{@link #tabular}: numeric features with multiplicative interactions and
 *       distractor columns: logistic regression versus an MLP, a cross network and
 *       TabNet-lite, whose sparse masks also reveal the informative columns.</li>
 * </ul>
 */
public final class RecsysTabularBenchmarks {

    private RecsysTabularBenchmarks() {
    }

    /**
     * Trains a logit model with binary cross-entropy and measures test accuracy.
     *
     * @param model features to one logit
     * @param params its parameters
     * @param train training rows
     * @param test test rows
     * @param steps gradient steps of batch 64
     * @param lr the Adam learning rate
     * @param seed the seed
     * @return the test accuracy
     */
    public static double accuracy(UnaryOperator<Var> model, List<Var> params, SyntheticRecsys.Table train,
            SyntheticRecsys.Table test, int steps, float lr, long seed) {
        Optimizer opt = Optimizer.adam(lr);
        SplittableRandom r = new SplittableRandom(seed);
        int cols = train.features().shape().dim(1);
        float[] x = train.features().toArray();
        float[] y = train.labels().toArray();
        for (int s = 0; s < steps; s++) {
            float[] bx = new float[64 * cols];
            float[] by = new float[64];
            for (int i = 0; i < 64; i++) {
                int j = r.nextInt(train.rows());
                System.arraycopy(x, j * cols, bx, i * cols, cols);
                by[i] = y[j];
            }
            Var logit = model.apply(Var.of(Tensor.of(Shape.of(64, cols), bx)));
            logit.softplus().sub(logit.mul(Tensor.of(Shape.of(64, 1), by))).mean().backward();
            opt.step(params);
        }
        float[] p = model.apply(Var.of(test.features())).value().toArray();
        float[] ty = test.labels().toArray();
        int right = 0;
        for (int i = 0; i < p.length; i++) {
            right += (p[i] > 0) == (ty[i] > 0.5f) ? 1 : 0;
        }
        return right / (double) p.length;
    }

    private static Map<String, Double> compare(SyntheticRecsys.Table train, SyntheticRecsys.Table test, int steps,
            long seed, boolean withTabNet) {
        int cols = train.features().shape().dim(1);
        Map<String, Double> out = new LinkedHashMap<>();
        Model logistic = Model.sequential().input(cols).dense(1).seed(seed).build();
        out.put("logistic", accuracy(x -> logistic.forward(x, true), logistic.parameters(), train, test, steps, 0.03f,
                seed));
        Model mlp = Model.sequential().input(cols).dense(32, Activation.RELU).dense(32, Activation.RELU).dense(1)
                .seed(seed).build();
        out.put("MLP", accuracy(x -> mlp.forward(x, true), mlp.parameters(), train, test, steps, 0.01f, seed));
        FactorizationMachine fm = new FactorizationMachine(cols, 8, new SplittableRandom(seed));
        out.put("FM", accuracy(x -> fm.forward(x, true), fm.parameters(), train, test, steps, 0.03f, seed));
        CrossNetwork cross = new CrossNetwork(cols, 2, new SplittableRandom(seed));
        Model head = Model.sequential().input(cols).dense(1).seed(seed).build();
        List<Var> dcnParams = new ArrayList<>(cross.parameters());
        dcnParams.addAll(head.parameters());
        out.put("DCN-V2", accuracy(x -> head.forward(cross.forward(x, true), true), dcnParams, train, test, steps,
                0.01f, seed));
        if (withTabNet) {
            TabNetLite tab = new TabNetLite(cols, 1, 3, 16, 1.5f, new SplittableRandom(seed));
            out.put("TabNet-lite", accuracy(x -> tab.forward(x, true), tab.parameters(), train, test, steps, 0.01f,
                    seed));
        }
        return out;
    }

    /**
     * @param steps gradient steps per model
     * @param seed the seed
     * @return model to test accuracy on click prediction
     */
    public static Map<String, Double> clicks(int steps, long seed) {
        SyntheticRecsys.Table all = SyntheticRecsys.clicks(6000, 3, 8, seed);
        return compare(all.rows(0, 4000), all.rows(4000, 6000), steps, seed, false);
    }

    /**
     * @param steps gradient steps per model
     * @param seed the seed
     * @return model to test accuracy on the interaction table
     */
    public static Map<String, Double> tabular(int steps, long seed) {
        return compare(SyntheticRecsys.tabular(3000, 10, seed), SyntheticRecsys.tabular(1500, 10, seed + 1), steps,
                seed, true);
    }

    /**
     * @param steps training steps per model
     * @param seed the seed
     * @return model to recall@10 of held-out interactions ("two-tower (new users)" on users unseen in training)
     */
    public static Map<String, Double> retrieval(int steps, long seed) {
        int users = 200;
        int items = 300;
        int f = 12;
        SyntheticRecsys.Preferences data = SyntheticRecsys.preferences(users, items, 6, f, 0.1, 10, seed);
        SplittableRandom r = new SplittableRandom(seed);
        BprMatrixFactorization mf = new BprMatrixFactorization(users, items, 16, seed);
        for (int s = 0; s < steps; s++) {
            int[] u = new int[128];
            int[] pos = new int[128];
            int[] neg = new int[128];
            for (int i = 0; i < 128; i++) {
                u[i] = r.nextInt(users);
                pos[i] = data.relevant()[u[i]][r.nextInt(7)];
                neg[i] = r.nextInt(items);
            }
            mf.trainStep(u, pos, neg);
        }
        Model user = Model.sequential().input(f).dense(32, Activation.RELU).dense(16).seed(seed).build();
        Model item = Model.sequential().input(f).dense(32, Activation.RELU).dense(16).seed(seed + 1).build();
        TwoTowerRetrieval tt = TwoTowerRetrieval.of(user, item);
        float[] uf = data.userFeatures().toArray();
        float[] itf = data.itemFeatures().toArray();
        int seen = 150;
        for (int s = 0; s < steps; s++) {
            float[] bu = new float[64 * f];
            float[] bi = new float[64 * f];
            for (int i = 0; i < 64; i++) {
                int u = r.nextInt(seen);
                int it = data.relevant()[u][r.nextInt(7)];
                System.arraycopy(uf, u * f, bu, i * f, f);
                System.arraycopy(itf, it * f, bi, i * f, f);
            }
            tt.trainStep(Tensor.of(Shape.of(64, f), bu), Tensor.of(Shape.of(64, f), bi));
        }
        int[][] towerTop = tt.topK(data.userFeatures(), data.itemFeatures(), 20);
        Map<String, Double> out = new LinkedHashMap<>();
        double mfHits = 0;
        double towerHits = 0;
        double coldHits = 0;
        for (int u = 0; u < users; u++) {
            Set<Integer> observed = new HashSet<>();
            Set<Integer> held = new HashSet<>();
            for (int k = 0; k < 10; k++) {
                (k < 7 ? observed : held).add(data.relevant()[u][k]);
            }
            float[] sc = mf.scores(u);
            for (int it : observed) {
                sc[it] = Float.NEGATIVE_INFINITY;
            }
            final float[] score = sc;
            int[] mfTop = java.util.stream.IntStream.range(0, items).boxed()
                    .sorted((a, b) -> Float.compare(score[b], score[a])).limit(10).mapToInt(Integer::intValue)
                    .toArray();
            int[] towerList = java.util.Arrays.stream(towerTop[u]).filter(it -> !observed.contains(it)).limit(10)
                    .toArray();
            for (int it : mfTop) {
                mfHits += u < seen && held.contains(it) ? 1 : 0;
            }
            for (int it : towerList) {
                if (held.contains(it)) {
                    if (u < seen) {
                        towerHits++;
                    } else {
                        coldHits++;
                    }
                }
            }
        }
        out.put("random", 10.0 / (items - 7));
        out.put("BPR-MF", mfHits / (seen * 3.0));
        out.put("two-tower", towerHits / (seen * 3.0));
        out.put("two-tower (new users)", coldHits / ((users - seen) * 3.0));
        return out;
    }
}
