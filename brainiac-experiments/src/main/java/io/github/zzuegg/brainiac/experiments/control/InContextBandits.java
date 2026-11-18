package io.github.zzuegg.brainiac.experiments.control;

import io.github.zzuegg.brainiac.experimental.rl.AlgorithmDistillation;
import io.github.zzuegg.brainiac.rl.Bandit;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * W6 3.17: algorithm distillation on the one-good-arm bandits of {@link MetaRlBandits}. A
 * causal transformer is trained on learning histories of UCB1 (the source algorithm) and
 * then plays new tasks in context with frozen weights. Compared with the source itself and
 * with the same transformer trained only on expert histories (expert distillation).
 * Metric: the in-context learning curve, reported as mean reward per pull over the first
 * and the last third of the horizon.
 *
 * <p>Paper: Laskin et al., <a href="https://arxiv.org/abs/2210.14215">arXiv:2210.14215</a>.
 */
public final class InContextBandits {

    private InContextBandits() {
    }

    /** UCB1: pull every arm once, then the arm with the highest mean plus bonus. */
    static int ucb(int[] arms, float[] rewards, int t, Bandit task, SplittableRandom random) {
        int k = MetaRlBandits.ARMS;
        double[] sum = new double[k];
        int[] n = new int[k];
        for (int j = 0; j < t; j++) {
            sum[arms[j]] += rewards[j];
            n[arms[j]]++;
        }
        int best = -1;
        double bestScore = Double.NEGATIVE_INFINITY;
        for (int a = 0; a < k; a++) {
            double score = n[a] == 0 ? 1e9 + random.nextDouble() : sum[a] / n[a] + Math.sqrt(2 * Math.log(t) / n[a]);
            if (score > bestScore) {
                bestScore = score;
                best = a;
            }
        }
        return best;
    }

    /**
     * @param horizon pulls per task
     * @param histories source histories to train on
     * @param steps gradient steps
     * @param evalTasks fresh tasks to evaluate on
     * @param seed the seed
     * @return method name to {first third, last third, whole} mean reward per pull
     */
    public static Map<String, double[]> run(int horizon, int histories, int steps, int evalTasks, long seed) {
        Map<String, double[]> out = new LinkedHashMap<>();
        List<AlgorithmDistillation.History> source = AlgorithmDistillation.collect(MetaRlBandits::task,
                InContextBandits::ucb, histories, horizon, seed);
        List<AlgorithmDistillation.History> expert = AlgorithmDistillation.collect(MetaRlBandits::task,
                (a, r, t, task, rnd) -> task.bestArm(), histories, horizon, seed);
        AlgorithmDistillation ad = AlgorithmDistillation.of(MetaRlBandits.ARMS, horizon, 32, 2, seed)
                .train(source, steps);
        out.put("AD (UCB histories)", summary(ad.evaluate(MetaRlBandits::task, evalTasks, seed + 100)));
        double[] src = new double[horizon];
        for (AlgorithmDistillation.History h : AlgorithmDistillation.collect(MetaRlBandits::task,
                InContextBandits::ucb, evalTasks, horizon, seed + 100)) {
            for (int t = 0; t < horizon; t++) {
                src[t] += h.rewards()[t] / evalTasks;
            }
        }
        out.put("source UCB1", summary(src));
        AlgorithmDistillation ed = AlgorithmDistillation.of(MetaRlBandits.ARMS, horizon, 32, 2, seed)
                .train(expert, steps);
        out.put("expert distillation", summary(ed.evaluate(MetaRlBandits::task, evalTasks, seed + 100)));
        return out;
    }

    private static double[] summary(double[] curve) {
        int third = curve.length / 3;
        double first = 0;
        double last = 0;
        double all = 0;
        for (int t = 0; t < curve.length; t++) {
            all += curve[t] / curve.length;
            if (t < third) {
                first += curve[t] / third;
            }
            if (t >= curve.length - third) {
                last += curve[t] / third;
            }
        }
        return new double[] {first, last, all};
    }
}
