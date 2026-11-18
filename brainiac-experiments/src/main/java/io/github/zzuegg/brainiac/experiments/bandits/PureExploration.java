package io.github.zzuegg.brainiac.experiments.bandits;

import io.github.zzuegg.brainiac.experimental.bandit.BestArmIdentification;
import io.github.zzuegg.brainiac.experimental.bandit.DuelingBandits;
import io.github.zzuegg.brainiac.experimental.data.BanditSuites;
import io.github.zzuegg.brainiac.train.HyperparameterSearch;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Wave 11 items 11.6.3-11.6.4: pure exploration and dueling.
 *
 * <ul>
 *   <li>{@link #fixedConfidence} (11.6.3): mean samples to stop at confidence {@code 1 - delta}
 *       and the error rate of every best-arm rule, uniform allocation included.</li>
 *   <li>{@link #fixedBudget} (11.6.3): error probability against the budget for uniform
 *       allocation, successive halving, Track-and-Stop, top-two and the anytime Almost
 *       Tracking (which is never told the budget).</li>
 *   <li>{@link #hyperparameters} (11.6.3): choosing the learning rate of a tiny logistic
 *       regression where each pull is one short noisy training run: Almost Tracking at the
 *       same number of runs as core {@code HyperparameterSearch.hyperband} (successive
 *       halving brackets), scored by the true error of the chosen rate.</li>
 *   <li>{@link #dueling} (11.6.4): RUCB, double Thompson sampling, the independent-pull
 *       reduction, a refitted Bradley-Terry model and uniform duels, on Bradley-Terry and
 *       cyclic preferences.</li>
 * </ul>
 */
public final class PureExploration {

    private PureExploration() {
    }

    /** The rules of the fixed-confidence table. */
    public static final List<BestArmIdentification.Method> FIXED_CONFIDENCE = List.of(
            BestArmIdentification.Method.UNIFORM, BestArmIdentification.Method.SUCCESSIVE_ELIMINATION,
            BestArmIdentification.Method.LUCB, BestArmIdentification.Method.TRACK_AND_STOP,
            BestArmIdentification.Method.TOP_TWO_TS, BestArmIdentification.Method.TOP_TWO_EB,
            BestArmIdentification.Method.ALMOST_TRACKING);

    /** The rules of the fixed-budget table. */
    public static final List<BestArmIdentification.Method> FIXED_BUDGET = List.of(
            BestArmIdentification.Method.UNIFORM, BestArmIdentification.Method.SUCCESSIVE_HALVING,
            BestArmIdentification.Method.TRACK_AND_STOP, BestArmIdentification.Method.TOP_TWO_EB,
            BestArmIdentification.Method.ALMOST_TRACKING);

    /**
     * 11.6.3, fixed confidence.
     *
     * @param instance the bandit
     * @param model its reward model
     * @param delta the confidence parameter
     * @param maxSamples the sample cap per run
     * @param seeds seeds
     * @return rule name to {@code {mean samples, error rate, share stopped}}
     */
    public static Map<String, double[]> fixedConfidence(BanditSuites.Instance instance,
            BestArmIdentification.Model model, double delta, long maxSamples, int seeds) {
        int best = instance.bestArm();
        Map<String, double[]> out = new LinkedHashMap<>();
        for (BestArmIdentification.Method m : FIXED_CONFIDENCE) {
            double samples = 0;
            double errors = 0;
            double stopped = 0;
            for (int s = 0; s < seeds; s++) {
                BestArmIdentification.Outcome o = BestArmIdentification.fixedConfidence(m, instance, model, delta,
                        maxSamples, 100 + s);
                samples += o.samples();
                errors += o.recommendation() == best ? 0 : 1;
                stopped += o.stopped() ? 1 : 0;
            }
            out.put(m.toString(), new double[] {samples / seeds, errors / seeds, stopped / seeds});
        }
        return out;
    }

    /**
     * 11.6.3, fixed budget.
     *
     * @param instance the bandit
     * @param model its reward model
     * @param budgets the budgets
     * @param seeds seeds
     * @return rule name to the error probability at each budget
     */
    public static Map<String, double[]> fixedBudget(BanditSuites.Instance instance, BestArmIdentification.Model model,
            long[] budgets, int seeds) {
        int best = instance.bestArm();
        Map<String, double[]> out = new LinkedHashMap<>();
        for (BestArmIdentification.Method m : FIXED_BUDGET) {
            double[] err = new double[budgets.length];
            for (int b = 0; b < budgets.length; b++) {
                for (int s = 0; s < seeds; s++) {
                    err[b] += BestArmIdentification.fixedBudget(m, instance, model, budgets[b], 300 + s) == best ? 0 : 1;
                }
                err[b] /= seeds;
            }
            out.put(m.toString(), err);
        }
        return out;
    }

    // ---------------------------------------------------------------- hyperparameter use

    private static final double[] RATES = {0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10};

    /** One short training run: 15 SGD steps of logistic regression, returns validation accuracy. */
    private static double trainingRun(double lr, SplittableRandom r) {
        double w0 = 0;
        double w1 = 0;
        double bias = 0;
        for (int step = 0; step < 15; step++) {
            int y = r.nextInt(2);
            double x0 = (y == 1 ? 0.6 : -0.6) + r.nextGaussian();
            double x1 = (y == 1 ? 0.3 : -0.3) + r.nextGaussian();
            double p = 1 / (1 + Math.exp(-(w0 * x0 + w1 * x1 + bias)));
            double g = p - y;
            w0 -= lr * g * x0;
            w1 -= lr * g * x1;
            bias -= lr * g;
        }
        int correct = 0;
        int n = 40;
        for (int i = 0; i < n; i++) {
            int y = r.nextInt(2);
            double x0 = (y == 1 ? 0.6 : -0.6) + r.nextGaussian();
            double x1 = (y == 1 ? 0.3 : -0.3) + r.nextGaussian();
            correct += ((w0 * x0 + w1 * x1 + bias) > 0 ? 1 : 0) == y ? 1 : 0;
        }
        return correct / (double) n;
    }

    /**
     * The hyperparameter-selection result.
     *
     * @param pulls training runs used by both searches
     * @param almostTrackingRegret mean true accuracy gap of Almost Tracking's choice to the best rate
     * @param hyperbandRegret the same for Hyperband
     * @param bestAccuracy the best rate's true accuracy
     */
    public record HyperRow(long pulls, double almostTrackingRegret, double hyperbandRegret, double bestAccuracy) {
    }

    /**
     * 11.6.3, hyperparameter use: Almost Tracking against core Hyperband at equal training runs.
     *
     * @param maxBudget Hyperband's largest per-configuration budget (runs averaged)
     * @param seeds seeds
     * @return the row
     */
    public static HyperRow hyperparameters(int maxBudget, int seeds) {
        SplittableRandom truth = new SplittableRandom(9);
        double[] mean = new double[RATES.length];
        for (int a = 0; a < RATES.length; a++) {
            for (int i = 0; i < 400; i++) {
                mean[a] += trainingRun(RATES[a], truth) / 400;
            }
        }
        double best = java.util.Arrays.stream(mean).max().orElse(0);
        BanditSuites.Instance inst = BanditSuites.experiments("logistic lr", mean, (a, r) -> trainingRun(RATES[a], r));
        double atRegret = 0;
        double hbRegret = 0;
        long pulls = 0;
        for (int s = 0; s < seeds; s++) {
            SplittableRandom r = new SplittableRandom(500 + s);
            long[] used = {0};
            HyperparameterSearch search = new HyperparameterSearch(Map.of("lr",
                    new HyperparameterSearch.Choice(java.util.Arrays.stream(RATES).boxed().toList())), 600 + s);
            HyperparameterSearch.Result res = search.hyperband((config, budget) -> {
                used[0] += budget;
                double acc = 0;
                for (int i = 0; i < budget; i++) {
                    acc += trainingRun((Double) config.get("lr"), r);
                }
                return -acc / budget;
            }, maxBudget, 3);
            double chosen = (Double) res.best().config().get("lr");
            int idx = java.util.Arrays.stream(RATES).boxed().toList().indexOf(chosen);
            hbRegret += best - mean[idx];
            int at = BestArmIdentification.fixedBudget(BestArmIdentification.Method.ALMOST_TRACKING, inst,
                    BestArmIdentification.Model.gaussian(0.1), used[0], 700 + s);
            atRegret += best - mean[at];
            pulls += used[0];
        }
        return new HyperRow(pulls / seeds, atRegret / seeds, hbRegret / seeds, best);
    }

    // ---------------------------------------------------------------- dueling

    /**
     * 11.6.4: dueling policies on one preference matrix.
     *
     * @param prefs the preferences
     * @param horizon duels
     * @param seeds seeds
     * @return policy name to {@code {mean cumulative regret, share of late duels led by a winner}}
     */
    public static Map<String, double[]> dueling(BanditSuites.Preferences prefs, int horizon, int seeds) {
        Map<String, double[]> out = new LinkedHashMap<>();
        for (DuelingBandits.Method m : DuelingBandits.Method.values()) {
            double regret = 0;
            double share = 0;
            for (int s = 0; s < seeds; s++) {
                DuelingBandits.Run run = DuelingBandits.run(prefs, DuelingBandits.of(m, prefs.arms(), s + 1), horizon,
                        900 + s);
                regret += run.regret();
                share += run.winnerShare();
            }
            out.put(m.toString(), new double[] {regret / seeds, share / seeds});
        }
        return out;
    }

    /**
     * Prints the tables at a larger budget.
     *
     * @param args unused
     */
    public static void main(String[] args) {
        BanditSuites.Instance g = BanditSuites.gaussian(1, 1, 0.5, 0.5, 0.5, 0.4, 0.3);
        System.out.println(fixedConfidence(g, BestArmIdentification.Model.gaussian(1), 0.05, 100_000, 50));
        System.out.println(dueling(BanditSuites.cyclic(6, 0.3, 1), 20_000, 20));
    }
}
