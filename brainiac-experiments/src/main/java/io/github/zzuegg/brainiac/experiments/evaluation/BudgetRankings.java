package io.github.zzuegg.brainiac.experiments.evaluation;

import io.github.zzuegg.brainiac.experimental.eval.BudgetRankingStudy;
import io.github.zzuegg.brainiac.experimental.eval.CurveExtrapolator;
import io.github.zzuegg.brainiac.experiments.attention.AttentionShootout;
import io.github.zzuegg.brainiac.experiments.sequence.MixerShootout;
import io.github.zzuegg.brainiac.experiments.timeseries.LongHorizonForecasting;
import io.github.zzuegg.brainiac.nn.LayerSpec;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Rankings against the training budget (9.3.16) and early prediction of the final ranking
 * (9.3.17).
 * <ul>
 *   <li>{@link #forecasting}: the univariate {@code LongHorizonForecasting} lineup (DLinear,
 *       PatchTST, the vanilla transformer, Autoformer, FEDformer, TimesNet, N-HiTS, N-BEATS,
 *       iTransformer, TiDE) at lookback 48, horizon 24, logged densely up to 32x the test
 *       budget of 40 steps;</li>
 *   <li>{@link #mixers}: a lineup from {@code MixerShootout} and {@code AttentionShootout} on
 *       associative recall, logged the same way;</li>
 *   <li>{@link #study}: {@link BudgetRankingStudy} on the ladder 1x..32x of those curves:
 *       Kendall's τ of each budget's ranking with the 32x ranking, and crossover budgets;</li>
 *   <li>{@link #earlyRanking}: {@link CurveExtrapolator} on the curves truncated at a fraction of
 *       the 32x budget, against ranking by the last observed value and a least-squares
 *       power law, plus successive halving's survivor.</li>
 * </ul>
 */
public final class BudgetRankings {

    private BudgetRankings() {
    }

    /**
     * Densely logged learning curves.
     *
     * @param methods the lineup
     * @param checkpoints the logged budgets, increasing, starting at 0
     * @param scores score per {@code [method][seed][checkpoint]}
     * @param base the test budget
     * @param lowerIsBetter whether lower scores are better
     */
    public record Curves(List<String> methods, int[] checkpoints, double[][][] scores, int base,
            boolean lowerIsBetter) {

        /**
         * Restricts the curves to some budgets.
         *
         * @param budgets budgets that are among the checkpoints
         * @return score per {@code [method][seed][budget]}
         */
        public double[][][] at(int[] budgets) {
            double[][][] out = new double[scores.length][][];
            for (int m = 0; m < scores.length; m++) {
                out[m] = new double[scores[m].length][budgets.length];
                for (int s = 0; s < scores[m].length; s++) {
                    for (int b = 0; b < budgets.length; b++) {
                        out[m][s][b] = scores[m][s][index(budgets[b])];
                    }
                }
            }
            return out;
        }

        /**
         * The checkpoint index of a budget.
         *
         * @param budget a logged budget
         * @return its index
         */
        public int index(int budget) {
            for (int i = 0; i < checkpoints.length; i++) {
                if (checkpoints[i] == budget) {
                    return i;
                }
            }
            throw new IllegalArgumentException("budget " + budget + " was not logged");
        }
    }

    /**
     * Checkpoints every {@code base / perBase} steps from 0 to {@code base * 2^doublings}.
     *
     * @param base the test budget
     * @param doublings ladder doublings
     * @param perBase checkpoints per test budget, dividing {@code base}
     * @return the checkpoints
     */
    public static int[] denseCheckpoints(int base, int doublings, int perBase) {
        int last = base << doublings;
        int every = base / perBase;
        int[] c = new int[last / every + 1];
        for (int i = 0; i < c.length; i++) {
            c[i] = i * every;
        }
        return c;
    }

    /**
     * Learning curves of the long-horizon forecasting lineup (test MSE, lower is better).
     *
     * @param models names from {@code LongHorizonForecasting.CURVE_LINEUP}
     * @param base the test budget (40 steps in {@code LongHorizonForecastingTest})
     * @param doublings ladder doublings (5 for 32x)
     * @param perBase checkpoints per test budget
     * @param seeds the seeds (the published row used 11)
     * @return the curves
     */
    public static Curves forecasting(List<String> models, int base, int doublings, int perBase, long[] seeds) {
        int[] cps = denseCheckpoints(base, doublings, perBase);
        double[][][] scores = new double[models.size()][seeds.length][];
        for (int s = 0; s < seeds.length; s++) {
            Map<String, double[]> c = LongHorizonForecasting.horizonCurves(48, 24, cps, models, seeds[s]);
            for (int m = 0; m < models.size(); m++) {
                scores[m][s] = c.get(models.get(m));
            }
        }
        return new Curves(List.copyOf(models), cps, scores, base, true);
    }

    /**
     * The mixer and attention lineups of waves 4 and 5 merged, in their default sizes.
     *
     * @return name to spec
     */
    public static Map<String, LayerSpec> mixerLineup() {
        Map<String, LayerSpec> all = new LinkedHashMap<>(MixerShootout.lineup());
        all.putAll(AttentionShootout.lineup());
        return all;
    }

    /**
     * Learning curves of mixers on associative recall (test accuracy, higher is better).
     *
     * @param lineup name to spec
     * @param task the task
     * @param base the test budget
     * @param doublings ladder doublings
     * @param perBase checkpoints per test budget
     * @param seeds the seeds
     * @return the curves
     */
    public static Curves mixers(Map<String, LayerSpec> lineup, MixerShootout.Task task, int base, int doublings,
            int perBase, long[] seeds) {
        int[] cps = denseCheckpoints(base, doublings, perBase);
        List<String> names = new ArrayList<>(lineup.keySet());
        double[][][] scores = new double[names.size()][seeds.length][];
        for (int m = 0; m < names.size(); m++) {
            for (int s = 0; s < seeds.length; s++) {
                scores[m][s] = MixerShootout.curve(lineup.get(names.get(m)), task, 16, cps, 0.01f, seeds[s]);
            }
        }
        return new Curves(names, cps, scores, base, false);
    }

    /**
     * The budget-ladder study of logged curves.
     *
     * @param c the curves
     * @param doublings ladder doublings
     * @param bootstrap bootstrap resamples
     * @return the study
     */
    public static BudgetRankingStudy.Result study(Curves c, int doublings, int bootstrap) {
        int[] ladder = BudgetRankingStudy.ladder(c.base(), doublings);
        return BudgetRankingStudy.analyse(c.methods(), ladder, c.at(ladder), c.lowerIsBetter(), bootstrap, 7);
    }

    /**
     * Kendall's τ with the final ranking of each early-ranking rule.
     *
     * @param fraction the observed fraction of the largest budget
     * @param lastValue τ of ranking by the last observed value
     * @param powerLaw τ of a least-squares power-law extrapolation
     * @param ensemble τ of the parametric ensemble's posterior-mean extrapolation
     * @param pfn τ of the LC-PFN extrapolation, NaN when no PFN was given
     * @param halvingSurvivorCorrect whether successive halving's survivor is the true best
     * @param computeSaved the share of the full lineup's training budget not spent by extrapolation
     * @param halvingSaved the share not spent by successive halving
     */
    public record EarlyRanking(double fraction, double lastValue, double powerLaw, double ensemble, double pfn,
            boolean halvingSurvivorCorrect, double computeSaved, double halvingSaved) {
        @Override
        public String toString() {
            return String.format("at %.0f%%: tau last value %.2f, power law %.2f, ensemble %.2f, LC-PFN %.2f; "
                    + "%.0f%% compute saved; successive halving survivor %s, %.0f%% saved", 100 * fraction,
                    lastValue, powerLaw, ensemble, pfn, 100 * computeSaved,
                    halvingSurvivorCorrect ? "correct" : "wrong", 100 * halvingSaved);
        }
    }

    /**
     * Predicts the final (largest-budget) ranking from curves truncated at a fraction of it.
     * Every method's seed-mean curve is extrapolated; τ is against the seed-mean final scores.
     *
     * @param c the curves
     * @param fraction the observed share of the largest budget
     * @param pfn a trained LC-PFN, or null
     * @return the τ of each rule
     */
    public static EarlyRanking earlyRanking(Curves c, double fraction, CurveExtrapolator.Pfn pfn) {
        int last = c.checkpoints()[c.checkpoints().length - 1];
        int cut = 0;
        while (cut + 1 < c.checkpoints().length && c.checkpoints()[cut + 1] <= fraction * last) {
            cut++;
        }
        int m = c.methods().size();
        double[] truth = new double[m];
        double[] lastValue = new double[m];
        double[] power = new double[m];
        double[] ensemble = new double[m];
        double[] pfnPred = new double[m];
        double sign = c.lowerIsBetter() ? 1 : -1;
        double[][] means = new double[m][];
        for (int i = 0; i < m; i++) {
            double[] mean = new double[c.checkpoints().length];
            for (double[] s : c.scores()[i]) {
                for (int k = 0; k < mean.length; k++) {
                    mean[k] += s[k] / c.scores()[i].length;
                }
            }
            means[i] = mean;
            truth[i] = sign * mean[mean.length - 1];
            double[] x = new double[cut];
            double[] y = new double[cut];
            for (int k = 1; k <= cut; k++) {
                x[k - 1] = c.checkpoints()[k] / (double) last;
                y[k - 1] = sign * mean[k];
            }
            lastValue[i] = y[cut - 1];
            power[i] = CurveExtrapolator.powerLawLeastSquares(x, y).predict(1.0);
            ensemble[i] = CurveExtrapolator.ensemble(x, y, 1.0, 400, 31 + i).mean();
            pfnPred[i] = pfn == null ? Double.NaN : pfn.predict(x, y, 1.0).mean();
        }
        double[] halvingScores = new double[m];
        double saved = successiveHalving(means, c.checkpoints(), sign, cut, halvingScores);
        int best = 0;
        int survivor = 0;
        for (int i = 1; i < m; i++) {
            best = truth[i] < truth[best] ? i : best;
            survivor = halvingScores[i] < halvingScores[survivor] ? i : survivor;
        }
        return new EarlyRanking(fraction, BudgetRankingStudy.kendallTau(lastValue, truth),
                BudgetRankingStudy.kendallTau(power, truth), BudgetRankingStudy.kendallTau(ensemble, truth),
                pfn == null ? Double.NaN : BudgetRankingStudy.kendallTau(pfnPred, truth), best == survivor,
                1 - (double) cut / (c.checkpoints().length - 1), saved);
    }

    /**
     * Successive halving over the logged curves: the first rung observes every method up to the
     * truncation point, each later rung keeps the better half and doubles the budget. Returns the
     * share of compute saved against training everything to the end; {@code out} receives each
     * method's last observed score (survivors have the furthest observation).
     */
    private static double successiveHalving(double[][] means, int[] cps, double sign, int firstRung,
            double[] out) {
        int m = means.length;
        boolean[] alive = new boolean[m];
        java.util.Arrays.fill(alive, true);
        int rung = Math.max(1, firstRung);
        long spent = 0;
        int lastIndex = cps.length - 1;
        int prev = 0;
        while (true) {
            int count = 0;
            for (int i = 0; i < m; i++) {
                if (alive[i]) {
                    out[i] = sign * means[i][rung];
                    spent += rung - prev;
                    count++;
                }
            }
            if (count == 1 || rung == lastIndex) {
                break;
            }
            Integer[] order = new Integer[m];
            for (int i = 0; i < m; i++) {
                order[i] = i;
            }
            java.util.Arrays.sort(order, (a, b) -> Double.compare(alive[a] ? out[a] : Double.MAX_VALUE,
                    alive[b] ? out[b] : Double.MAX_VALUE));
            int keep = (count + 1) / 2;
            for (int k = keep; k < m; k++) {
                if (alive[order[k]]) {
                    alive[order[k]] = false;
                    out[order[k]] += 1e6;
                }
            }
            prev = rung;
            rung = Math.min(lastIndex, 2 * rung);
        }
        for (int i = 0; i < m; i++) {
            if (!alive[i]) {
                out[i] = Math.max(out[i], 1e6);
            }
        }
        return 1 - spent / ((double) lastIndex * m);
    }

    /**
     * The full study: the forecasting lineup and the merged mixer lineup on the 1x..32x ladder
     * with three seeds, then early ranking at 5, 10 and 25% of 32x.
     *
     * @param args {@code [forecasting|mixers|both] [seeds]}
     */
    public static void main(String[] args) {
        String which = args.length > 0 ? args[0] : "both";
        int seeds = args.length > 1 ? Integer.parseInt(args[1]) : 3;
        long[] forecastSeeds = new long[seeds];
        long[] mixerSeeds = new long[seeds];
        for (int s = 0; s < seeds; s++) {
            forecastSeeds[s] = 11 + s;
            mixerSeeds[s] = 1 + s;
        }
        CurveExtrapolator.Pfn pfn = new CurveExtrapolator.Pfn(12, 48, 32, 2, 0.002f,
                new java.util.SplittableRandom(5));
        System.out.printf("LC-PFN pretraining loss %.3f%n", pfn.pretrain(1500, 32, 9));
        List<Curves> all = new ArrayList<>();
        if (!which.equals("mixers")) {
            long t0 = System.nanoTime();
            all.add(forecasting(LongHorizonForecasting.CURVE_LINEUP, 40, 5, 4, forecastSeeds));
            System.out.printf("forecasting curves in %.0f s%n", (System.nanoTime() - t0) / 1e9);
        }
        if (!which.equals("forecasting")) {
            long t0 = System.nanoTime();
            all.add(mixers(mixerLineup(), MixerShootout.recall(8, 8, 4), 25, 5, 5, mixerSeeds));
            System.out.printf("mixer curves in %.0f s%n", (System.nanoTime() - t0) / 1e9);
        }
        for (Curves c : all) {
            BudgetRankingStudy.Result r = study(c, 5, 1000);
            System.out.print(BudgetRankingStudy.table(r));
            for (int b = 0; b < r.budgets().length; b++) {
                System.out.println(r.budgets()[b] + " steps: " + r.ranking(b));
            }
            for (BudgetRankingStudy.Crossover x : r.crossovers()) {
                System.out.println("crossover: " + x);
            }
            System.out.println("curves " + java.util.Arrays.deepToString(c.scores()));
            for (double f : new double[] {0.05, 0.1, 0.25}) {
                System.out.println(earlyRanking(c, f, pfn));
            }
        }
    }
}
