package io.github.zzuegg.brainiac.experiments.recsys;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.ClickLogs;
import io.github.zzuegg.brainiac.experimental.recsys.ClickDebiasing;
import io.github.zzuegg.brainiac.experimental.recsys.RankingLosses;
import io.github.zzuegg.brainiac.experimental.recsys.RankingMetrics;
import io.github.zzuegg.brainiac.experimental.train.LambdaMart;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Loss;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Unbiased learning to rank from click logs (W9 1.3) on {@link ClickLogs#rankingLog}: a noisy
 * logging ranker shows the top 10 of 20 documents per query, and clicks come from a position-based
 * model with position decay {@code eta} (or from a cascade model). Every ranker is the same small
 * MLP trained on the sessions of 80 % of the queries, and is scored by NDCG@5 against the true
 * graded relevance of all 20 documents of the held-out queries:
 * <ul>
 *   <li>naive clicks: RankNet and LambdaRank ({@link RankingLosses}), a listwise softmax, and
 *       {@link LambdaMart} on per-document click counts;</li>
 *   <li>debiased ({@link ClickDebiasing}): IPW with the true propensities, regression EM and the
 *       dual learning algorithm (both estimate the propensities themselves; their error against
 *       the true examination curve is reported), and the cascade-model correction;</li>
 *   <li>an oracle trained by LambdaRank on the true grades of the shown lists.</li>
 * </ul>
 */
public final class ClickLogLearning {

    private ClickLogLearning() {
    }

    private static final int DOCS = 20;
    private static final int K = 10;

    private static Model ranker(int d, long seed) {
        return Model.sequential().input(d).dense(16, Activation.RELU).dense(1).seed(seed).build();
    }

    /**
     * Runs every method.
     *
     * @param eta the position-bias exponent
     * @param clickModel the click model generating the log
     * @param steps Adam steps per neural ranker
     * @param seed the seed
     * @return method name to {@code {NDCG@5, propensity-curve error (NaN if not estimated)}}
     */
    public static Map<String, double[]> run(double eta, ClickLogs.ClickModel clickModel, int steps, long seed) {
        int queries = 250;
        int trainQueries = 200;
        int sessions = 20;
        int d = 6;
        ClickLogs.RankingLog log = ClickLogs.rankingLog(queries, DOCS, d, K, sessions, eta, clickModel, 1.0, seed);
        int trainSessions = trainQueries * sessions;
        double[][][] feats = new double[trainSessions][K][];
        int[][] clicks = new int[trainSessions][];
        int[][] grades = new int[trainSessions][K];
        for (int s = 0; s < trainSessions; s++) {
            int q = log.query()[s];
            for (int p = 0; p < K; p++) {
                feats[s][p] = log.features()[q][log.shown()[s][p]];
                grades[s][p] = log.relevance()[q][log.shown()[s][p]];
            }
            clicks[s] = log.clicks()[s];
        }
        ClickDebiasing.Impressions imp = new ClickDebiasing.Impressions(feats, clicks);
        Map<String, double[]> out = new LinkedHashMap<>();
        double[] ones = new double[K];
        java.util.Arrays.fill(ones, 1);
        double nan = Double.NaN;
        out.put("naive RankNet", new double[] {ndcg(listwise(imp, null, RankingLosses.rankNet(), d, steps, seed), log,
                trainQueries), nan});
        out.put("naive LambdaRank", new double[] {ndcg(listwise(imp, null, RankingLosses.lambdaRank(5), d, steps,
                seed), log, trainQueries), nan});
        Model naive = ranker(d, seed);
        ClickDebiasing.trainIpw(naive, imp, ones, steps, 32, 3e-3f, seed);
        out.put("naive softmax", new double[] {ndcg(naive, log, trainQueries), nan});
        out.put("naive LambdaMART", new double[] {lambdaMart(log, trainQueries, sessions), nan});
        Model ipw = ranker(d, seed);
        ClickDebiasing.trainIpw(ipw, imp, log.examination(), steps, 32, 3e-3f, seed);
        out.put("IPW (true propensities)", new double[] {ndcg(ipw, log, trainQueries), 0});
        Model em = ranker(d, seed);
        double[] emTheta = ClickDebiasing.regressionEm(em, imp, 10, Math.max(1, steps / 10), 32, 3e-3f, seed);
        out.put("regression EM", new double[] {ndcg(em, log, trainQueries), curveError(emTheta, log.examination())});
        Model dla = ranker(d, seed);
        double[] dlaTheta = ClickDebiasing.dualLearning(dla, imp, steps, 32, 3e-3f, seed);
        out.put("DLA", new double[] {ndcg(dla, log, trainQueries), curveError(dlaTheta, log.examination())});
        Model cascade = ranker(d, seed);
        Optimizer opt = Optimizer.adam(3e-3f);
        SplittableRandom r = new SplittableRandom(seed);
        for (int s = 0; s < steps; s++) {
            int[] b = sample(trainSessions, 32, r);
            ClickDebiasing.cascadeLoss(ClickDebiasing.score(cascade, imp, b, true), ClickDebiasing.clicks(imp, b))
                    .backward();
            opt.step(cascade.parameters());
        }
        out.put("cascade correction", new double[] {ndcg(cascade, log, trainQueries), nan});
        out.put("oracle (true grades)", new double[] {ndcg(listwise(imp, grades, RankingLosses.lambdaRank(5), d,
                steps, seed), log, trainQueries), nan});
        return out;
    }

    private static int[] sample(int n, int size, SplittableRandom r) {
        int[] b = new int[size];
        for (int i = 0; i < size; i++) {
            b[i] = r.nextInt(n);
        }
        return b;
    }

    /** Trains on session lists with a core ranking loss, against clicks or (oracle) true grades. */
    private static Model listwise(ClickDebiasing.Impressions imp, int[][] grades, Loss loss, int d, int steps,
            long seed) {
        Model m = ranker(d, seed);
        Optimizer opt = Optimizer.adam(3e-3f);
        SplittableRandom r = new SplittableRandom(seed);
        for (int s = 0; s < steps; s++) {
            int[] b = sample(imp.sessions(), 32, r);
            Tensor labels = ClickDebiasing.clicks(imp, b);
            if (grades != null) {
                float[] g = new float[b.length * K];
                for (int i = 0; i < b.length; i++) {
                    for (int p = 0; p < K; p++) {
                        g[i * K + p] = grades[b[i]][p];
                    }
                }
                labels = Tensor.of(Shape.of(b.length, K), g);
            }
            loss.compute(ClickDebiasing.score(m, imp, b, true), labels).backward();
            opt.step(m.parameters());
        }
        return m;
    }

    private static double lambdaMart(ClickLogs.RankingLog log, int trainQueries, int sessions) {
        Map<Long, int[]> counts = new LinkedHashMap<>();
        for (int s = 0; s < trainQueries * sessions; s++) {
            int q = log.query()[s];
            for (int p = 0; p < K; p++) {
                long key = (long) q * DOCS + log.shown()[s][p];
                counts.computeIfAbsent(key, x -> new int[1])[0] += log.clicks()[s][p];
            }
        }
        List<double[]> x = new ArrayList<>();
        List<Integer> g = new ArrayList<>();
        List<Integer> qs = new ArrayList<>();
        for (Map.Entry<Long, int[]> e : counts.entrySet()) {
            int q = (int) (e.getKey() / DOCS);
            int doc = (int) (e.getKey() % DOCS);
            x.add(log.features()[q][doc]);
            g.add(Math.min(4, (int) Math.round(8.0 * e.getValue()[0] / sessions)));
            qs.add(q);
        }
        LambdaMart lm = LambdaMart.fit(x.toArray(double[][]::new), g.stream().mapToInt(Integer::intValue).toArray(),
                qs.stream().mapToInt(Integer::intValue).toArray(), new LambdaMart.Settings(60, 3, 0.1, 5, 32, 5));
        double total = 0;
        int count = 0;
        for (int q = trainQueries; q < log.features().length; q++) {
            float[] scores = new float[DOCS];
            for (int i = 0; i < DOCS; i++) {
                scores[i] = (float) lm.predict(log.features()[q][i]);
            }
            total += RankingMetrics.ndcg(scores, log.relevance()[q], 5);
            count++;
        }
        return total / count;
    }

    private static double ndcg(Model m, ClickLogs.RankingLog log, int trainQueries) {
        int queries = log.features().length;
        int d = log.features()[0][0].length;
        double total = 0;
        for (int q = trainQueries; q < queries; q++) {
            float[] f = new float[DOCS * d];
            for (int i = 0; i < DOCS; i++) {
                for (int j = 0; j < d; j++) {
                    f[i * d + j] = (float) log.features()[q][i][j];
                }
            }
            float[] scores = Var.noGrad(() -> m.forward(Var.of(Tensor.of(Shape.of(DOCS, d), f)), false)).value()
                    .toArray();
            total += RankingMetrics.ndcg(scores, log.relevance()[q], 5);
        }
        return total / (queries - trainQueries);
    }

    private static double curveError(double[] estimate, double[] truth) {
        double e = 0;
        for (int p = 0; p < truth.length; p++) {
            e += Math.abs(estimate[p] - truth[p] / truth[0]) / truth.length;
        }
        return e;
    }
}
