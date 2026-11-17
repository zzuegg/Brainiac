package io.github.zzuegg.brainiac.experiments.classic;

import io.github.zzuegg.brainiac.experimental.train.ModelBasedSearch;
import io.github.zzuegg.brainiac.train.HyperparameterSearch;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * GP Bayesian optimisation and BOHB ({@link ModelBasedSearch}) against core
 * {@link HyperparameterSearch}'s random search, TPE and Hyperband, on the objective of the core
 * hyperparameter tests: a log-scale learning rate, a continuous knob, a layer count and an
 * optimizer choice, where more budget lowers the score as {@code 1 / budget}.
 * <ul>
 *   <li>{@link #fixedBudget}: the best score after a number of full-budget trials, averaged
 *       over seeds;</li>
 *   <li>{@link #multiFidelity}: the best full-budget score of Hyperband and BOHB, which spend
 *       the same total budget in the same brackets, averaged over seeds; random search with
 *       the same total budget is the reference.</li>
 * </ul>
 */
public final class HyperparameterSearchComparison {

    private HyperparameterSearchComparison() {
    }

    /**
     * The objective of core {@code HyperparameterSearchTest}: best at lr 1e-3, x 0.3, 4 layers and
     * "adam".
     *
     * @param c the configuration
     * @param budget the budget
     * @return the score, lower is better
     */
    public static double objective(Map<String, Object> c, int budget) {
        double lr = Math.log10((double) c.get("lr")) + 3;
        double x = (double) c.get("x") - 0.3;
        double layers = ((int) c.get("layers") - 4) / 4.0;
        double opt = c.get("opt").equals("adam") ? 0 : 0.5;
        return lr * lr + x * x + layers * layers + opt + 1.0 / budget;
    }

    /** @return the search space of {@link #objective} */
    public static Map<String, HyperparameterSearch.Range> space() {
        Map<String, HyperparameterSearch.Range> s = new LinkedHashMap<>();
        s.put("lr", new HyperparameterSearch.LogUniform(1e-5, 1e-1));
        s.put("x", new HyperparameterSearch.Uniform(0, 1));
        s.put("layers", new HyperparameterSearch.IntRange(1, 8));
        s.put("opt", new HyperparameterSearch.Choice(List.of("sgd", "adam")));
        return s;
    }

    /**
     * Fixed-budget search.
     *
     * @param trials trials per method
     * @param seeds how many seeds to average
     * @return method to the mean best score (the {@code 1 / budget} floor is 0.01)
     */
    public static Map<String, Double> fixedBudget(int trials, int seeds) {
        Map<String, Double> out = new LinkedHashMap<>();
        int budget = 100;
        for (long seed = 0; seed < seeds; seed++) {
            add(out, "random", new HyperparameterSearch(space(), seed)
                    .random(HyperparameterSearchComparison::objective, trials, budget).best().score() / seeds);
            add(out, "tpe", new HyperparameterSearch(space(), seed)
                    .tpe(HyperparameterSearchComparison::objective, trials, budget, 8, 24).best().score() / seeds);
            add(out, "GP-BO (EI)", new ModelBasedSearch(space(), seed)
                    .bayesOpt(HyperparameterSearchComparison::objective, trials, budget, 8,
                            ModelBasedSearch.Acquisition.EXPECTED_IMPROVEMENT).best().score() / seeds);
            add(out, "GP-BO (LCB)", new ModelBasedSearch(space(), seed)
                    .bayesOpt(HyperparameterSearchComparison::objective, trials, budget, 8,
                            ModelBasedSearch.Acquisition.LOWER_CONFIDENCE_BOUND).best().score() / seeds);
        }
        return out;
    }

    /**
     * Multi-fidelity search at equal total budget.
     *
     * @param maxBudget the largest budget
     * @param eta the reduction factor
     * @param iterations how often each method runs all its brackets
     * @param seeds how many seeds to average
     * @return method to the mean best full-budget score, plus "total budget" (per seed)
     */
    public static Map<String, Double> multiFidelity(int maxBudget, int eta, int iterations, int seeds) {
        Map<String, Double> out = new LinkedHashMap<>();
        double total = 0;
        for (long seed = 0; seed < seeds; seed++) {
            HyperparameterSearch core = new HyperparameterSearch(space(), seed);
            double hb = Double.POSITIVE_INFINITY;
            int spent = 0;
            for (int i = 0; i < iterations; i++) {
                HyperparameterSearch.Result r = core.hyperband(HyperparameterSearchComparison::objective, maxBudget, eta);
                hb = Math.min(hb, r.best().score());
                spent += r.trials().stream().mapToInt(HyperparameterSearch.Trial::budget).sum();
            }
            add(out, "hyperband", hb / seeds);
            HyperparameterSearch.Result b = new ModelBasedSearch(space(), seed)
                    .bohb(HyperparameterSearchComparison::objective, maxBudget, eta, iterations, 1.0 / 3);
            add(out, "BOHB", b.best().score() / seeds);
            int fullTrials = Math.max(1, spent / maxBudget);
            add(out, "random (same budget)", new HyperparameterSearch(space(), seed)
                    .random(HyperparameterSearchComparison::objective, fullTrials, maxBudget).best().score() / seeds);
            total = spent;
        }
        out.put("total budget", total);
        return out;
    }

    private static void add(Map<String, Double> m, String k, double v) {
        m.merge(k, v, Double::sum);
    }
}
