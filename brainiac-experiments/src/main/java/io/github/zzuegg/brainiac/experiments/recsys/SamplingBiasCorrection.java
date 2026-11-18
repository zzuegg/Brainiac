package io.github.zzuegg.brainiac.experiments.recsys;

import io.github.zzuegg.brainiac.experimental.data.ClickLogs;
import io.github.zzuegg.brainiac.experimental.nn.BprMatrixFactorization;
import io.github.zzuegg.brainiac.experimental.nn.TwoTowerRetrieval;
import io.github.zzuegg.brainiac.experimental.recsys.CorrectedInBatchSoftmax;
import io.github.zzuegg.brainiac.experimental.recsys.RankingMetrics;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;
import java.util.SplittableRandom;
import java.util.function.IntFunction;

/**
 * Sampling-bias correction of in-batch negatives (9.1.8) on {@link ClickLogs#interactions}: 400
 * users, 1000 items with Zipf popularity, 30 interactions per user. Models: plain in-batch
 * softmax, streaming logQ correction, mixed negatives (in-batch plus 128 uniform) and the full
 * softmax oracle ({@link CorrectedInBatchSoftmax}), against the core-style
 * {@link TwoTowerRetrieval} on one-hot ID towers (uncorrected in-batch) and BPR-MF with uniform
 * negatives. Every model gets the same number of batches of 256 pairs.
 * <p>
 * Metric: Recall@20 against each user's true top 20 items under the generating affinity, overall
 * and split into head items (the 10% most popular) and tail items; for the logQ models also the
 * total-variation distance between the estimator's normalised frequencies and the empirical item
 * frequencies of the training stream.
 */
public final class SamplingBiasCorrection {

    private SamplingBiasCorrection() {
    }

    private static double[] recall(ClickLogs.Interactions data, IntFunction<float[]> scores, int k) {
        int n = data.items();
        Integer[] order = new Integer[n];
        for (int i = 0; i < n; i++) {
            order[i] = i;
        }
        java.util.Arrays.sort(order, (a, b) -> Double.compare(data.popularity()[b], data.popularity()[a]));
        boolean[] head = new boolean[n];
        for (int i = 0; i < n / 10; i++) {
            head[order[i]] = true;
        }
        double[] hits = new double[3];
        double[] total = new double[3];
        for (int u = 0; u < data.users(); u++) {
            float[] truth = new float[n];
            for (int i = 0; i < n; i++) {
                truth[i] = (float) data.affinity()[u][i];
            }
            int[] relevant = RankingMetrics.topK(truth, k, Set.of());
            Set<Integer> got = new HashSet<>();
            for (int i : RankingMetrics.topK(scores.apply(u), k, Set.of())) {
                got.add(i);
            }
            for (int i : relevant) {
                int part = head[i] ? 1 : 2;
                double h = got.contains(i) ? 1 : 0;
                hits[0] += h;
                total[0]++;
                hits[part] += h;
                total[part]++;
            }
        }
        return new double[] {hits[0] / total[0], total[1] == 0 ? Double.NaN : hits[1] / total[1],
            total[2] == 0 ? Double.NaN : hits[2] / total[2]};
    }

    private static double frequencyError(CorrectedInBatchSoftmax m, ClickLogs.Interactions data) {
        int n = data.items();
        double[] f = new double[n];
        for (int i : data.item()) {
            f[i]++;
        }
        double[] p = new double[n];
        double sp = 0;
        for (int i = 0; i < n; i++) {
            p[i] = m.estimator().probability(i, 0);
            sp += p[i];
        }
        double tv = 0;
        for (int i = 0; i < n; i++) {
            tv += Math.abs(p[i] / sp - f[i] / data.size());
        }
        return tv / 2;
    }

    private static Tensor oneHot(int[] ids, int width) {
        float[] f = new float[ids.length * width];
        for (int i = 0; i < ids.length; i++) {
            f[i * width + ids[i]] = 1;
        }
        return Tensor.of(Shape.of(ids.length, width), f);
    }

    /**
     * @param exponent the Zipf exponent of item popularity, such as 0.8 or 1.2
     * @param steps batches of 256 pairs per model
     * @param seed the seed
     * @return model name to {@code {Recall@20, head Recall@20, tail Recall@20, frequency TV
     *     error (NaN without an estimator)}}
     */
    public static Map<String, double[]> run(double exponent, int steps, long seed) {
        ClickLogs.Interactions data = ClickLogs.interactions(400, 1000, 8, exponent, 30, seed);
        int batch = 256;
        Map<String, double[]> out = new LinkedHashMap<>();
        String[] names = {"in-batch softmax", "logQ-corrected in-batch", "mixed negatives + logQ",
            "full softmax (oracle)"};
        CorrectedInBatchSoftmax.Mode[] modes = CorrectedInBatchSoftmax.Mode.values();
        for (int k = 0; k < modes.length; k++) {
            CorrectedInBatchSoftmax m = new CorrectedInBatchSoftmax(data.users(), data.items(), 16, modes[k], 128,
                    seed);
            SplittableRandom r = new SplittableRandom(seed + 7);
            for (int s = 0; s < steps; s++) {
                int[] u = new int[batch];
                int[] i = new int[batch];
                for (int j = 0; j < batch; j++) {
                    int row = r.nextInt(data.size());
                    u[j] = data.user()[row];
                    i[j] = data.item()[row];
                }
                m.trainStep(u, i);
            }
            double[] rec = recall(data, m::scores, 20);
            boolean est = modes[k] == CorrectedInBatchSoftmax.Mode.LOG_Q
                    || modes[k] == CorrectedInBatchSoftmax.Mode.MIXED;
            out.put(names[k], new double[] {rec[0], rec[1], rec[2], est ? frequencyError(m, data) : Double.NaN});
        }
        Model userTower = Model.sequential().input(data.users()).dense(16).seed(seed).build();
        Model itemTower = Model.sequential().input(data.items()).dense(16).seed(seed + 1).build();
        TwoTowerRetrieval towers = TwoTowerRetrieval.of(userTower, itemTower).temperature(0.1f);
        SplittableRandom r = new SplittableRandom(seed + 7);
        for (int s = 0; s < steps; s++) {
            int[] u = new int[batch];
            int[] i = new int[batch];
            for (int j = 0; j < batch; j++) {
                int row = r.nextInt(data.size());
                u[j] = data.user()[row];
                i[j] = data.item()[row];
            }
            towers.trainStep(oneHot(u, data.users()), oneHot(i, data.items()));
        }
        int[] allUsers = new int[data.users()];
        int[] allItems = new int[data.items()];
        for (int j = 0; j < allUsers.length; j++) {
            allUsers[j] = j;
        }
        for (int j = 0; j < allItems.length; j++) {
            allItems[j] = j;
        }
        Tensor ue = towers.embedUsers(oneHot(allUsers, data.users()));
        Tensor ie = towers.embedItems(oneHot(allItems, data.items()));
        float[] all = ue.matmul(ie.transpose()).toArray();
        int n = data.items();
        double[] rec = recall(data, u -> java.util.Arrays.copyOfRange(all, u * n, (u + 1) * n), 20);
        out.put("TwoTowerRetrieval (one-hot towers)", new double[] {rec[0], rec[1], rec[2], Double.NaN});
        BprMatrixFactorization bpr = new BprMatrixFactorization(data.users(), data.items(), 16, seed);
        r = new SplittableRandom(seed + 7);
        for (int s = 0; s < steps; s++) {
            int[] u = new int[batch];
            int[] i = new int[batch];
            int[] neg = new int[batch];
            for (int j = 0; j < batch; j++) {
                int row = r.nextInt(data.size());
                u[j] = data.user()[row];
                i[j] = data.item()[row];
                neg[j] = r.nextInt(n);
            }
            bpr.trainStep(u, i, neg);
        }
        rec = recall(data, bpr::scores, 20);
        out.put("BPR-MF (uniform negatives)", new double[] {rec[0], rec[1], rec[2], Double.NaN});
        return out;
    }
}
